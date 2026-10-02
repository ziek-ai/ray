"""Recognising video topics, codecs and keyframes from MCAP payload bytes.

A window row has to carry, for every video topic, the frames back to the last
keyframe before the window starts, or the clip cannot be decoded. MCAP is
codec-agnostic and the caller is never asked to describe the encoding: a topic
counts as video when its schema name is one recorders use for compressed
images or video, or when its first payload sniffs as one of the codecs below;
the codec and each keyframe are then read from the bytes.

Payloads are the serialized message (CDR, protobuf, ...) with the frame bytes
embedded after a small header, so signatures are searched for rather than
expected at offset 0. Recognised: JPEG and PNG (every still image is a
keyframe), H.264 and H.265 in Annex-B byte-stream form, VP9 and AV1, which
are every ``format`` Foxglove's ``CompressedVideo`` message allows. VP9 and
AV1 carry no start codes, so their frame bytes are located by parsing the
protobuf or CDR framing of the message, and the message's ``format`` string
is used as a hint. A recognised video topic whose bytes fit none of these
still gets a lead-in: the whole look-back cap rather than a keyframe-exact
one (see ``mcap_reader``).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

_JPEG_SOI = b"\xff\xd8\xff"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ANNEXB_START = b"\x00\x00\x01"
# How far into a payload a still-image signature may sit: a ROS 2
# ``sensor_msgs/CompressedImage`` puts a header and a format string before it.
_SIGNATURE_SEARCH_WINDOW = 256

# H.264 (ITU-T H.264 table 7-1): nal_unit_type is the low five bits of the
# one-byte header; 5 is an IDR slice, the only slice type a decoder can start
# on. H.265 (ITU-T H.265 table 7-1): nal_unit_type is bits 1-6 of a two-byte
# header; 16-23 are the IRAP pictures (BLA, IDR, CRA and reserved IRAP types).
_H264_IDR = 5
_H265_IRAP_TYPES = range(16, 24)

# VP9 (bitstream spec 6.2): the uncompressed header opens with frame_marker
# ``0b10``, two profile bits, show_existing_frame and frame_type (0 = key
# frame); a key frame then carries frame_sync_code 0x49 0x83 0x42.
_VP9_SYNC_CODE = 0x498342

# AV1 (spec 5.3): an OBU header is a forbidden bit, a 4-bit obu_type, an
# extension flag, a has_size flag and a reserved bit. A temporal unit that
# holds a sequence header OBU is one a decoder can start on.
_AV1_SEQUENCE_HEADER = 1
_AV1_TEMPORAL_DELIMITER = 2
_AV1_OBU_TYPES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 15})

# Schema names under which recorders commonly log compressed video or images.
VIDEO_SCHEMA_NAMES = frozenset(
    {
        "foxglove_msgs/msg/CompressedVideo",
        "foxglove_msgs/CompressedVideo",
        "foxglove.CompressedVideo",
        "foxglove_msgs/msg/CompressedImage",
        "foxglove_msgs/CompressedImage",
        "foxglove.CompressedImage",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    }
)


class VideoCodec(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"

    @property
    def every_frame_is_a_keyframe(self) -> bool:
        return self in (VideoCodec.JPEG, VideoCodec.PNG)

    @property
    def is_annex_b(self) -> bool:
        return self in (VideoCodec.H264, VideoCodec.H265)


# ``format`` strings Foxglove's ``CompressedVideo`` / ``CompressedImage`` and
# ROS 2's ``CompressedImage`` carry, as a hint for the codec.
_FORMAT_TOKENS: Dict[bytes, VideoCodec] = {
    b"h264": VideoCodec.H264,
    b"h265": VideoCodec.H265,
    b"hevc": VideoCodec.H265,
    b"vp9": VideoCodec.VP9,
    b"av1": VideoCodec.AV1,
    b"jpeg": VideoCodec.JPEG,
    b"jpg": VideoCodec.JPEG,
    b"png": VideoCodec.PNG,
}


def is_video_schema(schema_name: Optional[str]) -> bool:
    """Whether a channel with this schema name carries compressed video or images."""
    return schema_name is not None and schema_name in VIDEO_SCHEMA_NAMES


@dataclass(frozen=True)
class VideoTopics:
    """Which topics carry video, settled at planning from the sample files.

    Planning sniffs the first message of every selected channel of the sample
    files: ``video`` holds the topics it recognised, ``probed`` every topic it
    looked at. A reader meeting a topic planning never saw sniffs the first
    payload itself.
    """

    video: FrozenSet[str] = frozenset()
    probed: FrozenSet[str] = frozenset()

    def recognises(
        self, topic: str, schema_name: Optional[str], payload: Optional[bytes] = None
    ) -> Optional[bool]:
        """Whether ``topic`` is video; ``None`` when nothing settles it yet.

        A video schema name or a topic planning recognised says yes; a topic
        planning probed and did not recognise says no; otherwise the payload
        decides when one is at hand.
        """
        if is_video_schema(schema_name) or topic in self.video:
            return True
        if topic in self.probed:
            return False
        if payload is not None:
            return detect_codec(payload) is not None
        return None


# -- Annex-B (H.264 / H.265) ---------------------------------------------------


def _nal_headers(payload: bytes) -> Iterator[Tuple[int, int]]:
    """Yield the first two bytes after every Annex-B start code in ``payload``.

    A four-byte start code (``00 00 00 01``) contains the three-byte one, so
    searching for the latter finds both.
    """
    position = payload.find(_ANNEXB_START)
    while position != -1:
        header = position + len(_ANNEXB_START)
        if header + 1 < len(payload):
            yield payload[header], payload[header + 1]
        position = payload.find(_ANNEXB_START, header)


def _annex_b_codec(payload: bytes) -> Optional[VideoCodec]:
    """H.264 or H.265 from the NAL header layouts, ``None`` if neither fits.

    An H.265 header is two bytes: a zero forbidden bit, a six-bit type, a
    six-bit layer id (zero in a single-layer stream) and a three-bit temporal
    id plus one (so at least one). An H.264 header is one byte: a zero
    forbidden bit, a two-bit reference indicator and a five-bit type between
    1 and 23 for anything carried in a stream.
    """
    headers = list(_nal_headers(payload))
    if not headers:
        return None
    hevc_like = all(
        (first & 0x80) == 0
        and (((first & 0x01) << 5) | (second >> 3)) == 0
        and (second & 0x07) >= 1
        for first, second in headers
    )
    if hevc_like:
        # The second byte of an H.264 NAL is payload: a profile for a parameter
        # set, slice-header bits for a slice, both practically never 1..7 for
        # every NAL of a stream. A stream that fits the H.265 layout throughout
        # is H.265, even where the H.264 layout would also fit.
        return VideoCodec.H265
    if all((first & 0x80) == 0 and 1 <= (first & 0x1F) <= 23 for first, _ in headers):
        return VideoCodec.H264
    return None


# -- message framing (where the frame bytes sit) ------------------------------


def _varint(data: bytes, pos: int) -> Tuple[Optional[int], int]:
    """A protobuf varint at ``pos``; ``(None, pos)`` when malformed."""
    value, shift = 0, 0
    while pos < len(data) and shift <= 63:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    return None, pos


def _protobuf_field(payload: bytes, number: int) -> Optional[bytes]:
    """Value of length-delimited field ``number`` if ``payload`` parses as protobuf."""
    pos, size = 0, len(payload)
    while pos < size:
        tag, pos = _varint(payload, pos)
        if tag is None:
            return None
        field, wire = tag >> 3, tag & 0x07
        if wire == 0:
            value, pos = _varint(payload, pos)
            if value is None:
                return None
        elif wire == 1:
            pos += 8
        elif wire == 2:
            length, pos = _varint(payload, pos)
            if length is None or pos + length > size:
                return None
            if field == number:
                return payload[pos : pos + length]
            pos += length
        elif wire == 5:
            pos += 4
        else:
            return None
    return None


def _cdr_video_data(payload: bytes) -> Optional[bytes]:
    """The ``data`` of a CDR-encoded Foxglove ``CompressedVideo`` message.

    ROS 2 CDR: a four-byte encapsulation header, ``timestamp`` (eight bytes),
    ``frame_id`` (a length-prefixed string, padded to four bytes), then the
    length-prefixed ``data`` bytes.
    """
    if len(payload) < 16 or payload[0] != 0 or payload[1] > 1:
        return None
    order = "little" if payload[1] == 1 else "big"
    pos = 12
    length = int.from_bytes(payload[pos : pos + 4], order)
    pos += 4 + length
    pos = (pos + 3) // 4 * 4
    if pos + 4 > len(payload):
        return None
    length = int.from_bytes(payload[pos : pos + 4], order)
    pos += 4
    if length == 0 or pos + length > len(payload):
        return None
    return payload[pos : pos + length]


def _embedded_frames(payload: bytes) -> List[bytes]:
    """Where a VP9 or AV1 frame may start: the message's ``data`` field, then the payload."""
    candidates: List[bytes] = []
    if payload[:1] in (b"\x0a", b"\x12", b"\x1a", b"\x22"):
        data = _protobuf_field(payload, 3)
        if data:
            candidates.append(data)
    elif payload[:1] == b"\x00":
        data = _cdr_video_data(payload)
        if data:
            candidates.append(data)
    candidates.append(payload)
    return candidates


def _format_hint(payload: bytes) -> Optional[VideoCodec]:
    """The codec a message's ``format`` string names, when one is found.

    Protobuf ``CompressedVideo`` keeps it in field 4; the CDR messages keep it
    as a length-prefixed string, after the frame in ``CompressedVideo`` and
    before it in ``sensor_msgs/CompressedImage``.
    """
    if payload[:1] in (b"\x0a", b"\x12", b"\x1a", b"\x22"):
        value = _protobuf_field(payload, 4)
        if value:
            return _FORMAT_TOKENS.get(value.strip().lower())
    for region in (payload[-32:], payload[:96]):
        for token, codec in _FORMAT_TOKENS.items():
            index = region.find(token)
            if index < 4 or index + len(token) >= len(region):
                continue
            length = int.from_bytes(region[index - 4 : index], "little")
            if length == len(token) + 1 and region[index + len(token)] == 0:
                return codec
    return None


# -- VP9 ---------------------------------------------------------------------


class _Bits:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, count: int) -> Optional[int]:
        if self._pos + count > len(self._data) * 8:
            return None
        value = 0
        for _ in range(count):
            byte = self._data[self._pos >> 3]
            value = (value << 1) | ((byte >> (7 - (self._pos & 7))) & 1)
            self._pos += 1
        return value


def _vp9_keyframe(frame: bytes, strict: bool) -> Optional[bool]:
    """Whether ``frame`` is a VP9 key frame; ``None`` if it is not VP9 at all.

    With ``strict`` only a key frame, whose sync code cannot occur by chance,
    counts as VP9: an inter frame's header is two bits and a few flags.
    """
    bits = _Bits(frame)
    if bits.read(2) != 0b10:
        return None
    low, high = bits.read(1), bits.read(1)
    if low is None or high is None:
        return None
    profile = (high << 1) | low
    if profile == 3 and bits.read(1) != 0:
        return None
    show_existing = bits.read(1)
    if show_existing is None:
        return None
    if show_existing:
        return None if strict else False
    frame_type = bits.read(1)
    if frame_type is None:
        return None
    if frame_type == 0:
        bits.read(2)  # show_frame, error_resilient_mode
        if bits.read(24) != _VP9_SYNC_CODE:
            return None
        return True
    return None if strict else False


# -- AV1 ---------------------------------------------------------------------


def _leb128(data: bytes, pos: int) -> Tuple[Optional[int], int]:
    value = 0
    for index in range(8):
        if pos >= len(data):
            return None, pos
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << (7 * index)
        if not byte & 0x80:
            return value, pos
    return None, pos


def _av1_obu_types(frame: bytes) -> Optional[List[int]]:
    """The OBU types of an AV1 temporal unit, or ``None`` if it does not parse.

    Tries the low-overhead format (every OBU carries its size) and then the
    Annex-B length-delimited one.
    """
    types = _av1_sized_obus(frame)
    if types is None:
        types = _av1_annex_b_obus(frame)
    return types or None


def _av1_obu_header(data: bytes, pos: int) -> Tuple[Optional[int], bool, int]:
    """``(obu_type, has_size, position after the header)``; type ``None`` if invalid."""
    if pos >= len(data):
        return None, False, pos
    header = data[pos]
    obu_type = (header >> 3) & 0x0F
    if header & 0x80 or header & 0x01 or obu_type not in _AV1_OBU_TYPES:
        return None, False, pos
    pos += 1 + ((header >> 2) & 1)
    return obu_type, bool((header >> 1) & 1), pos


def _av1_sized_obus(frame: bytes) -> Optional[List[int]]:
    pos, types = 0, []
    while pos < len(frame):
        obu_type, has_size, pos = _av1_obu_header(frame, pos)
        if obu_type is None:
            return None
        types.append(obu_type)
        if not has_size:
            # Only the last OBU of a unit may run to the end.
            return types
        size, pos = _leb128(frame, pos)
        if size is None or pos + size > len(frame):
            return None
        pos += size
    return types


def _av1_annex_b_obus(frame: bytes) -> Optional[List[int]]:
    unit_size, pos = _leb128(frame, 0)
    if unit_size is None or pos + unit_size != len(frame):
        return None
    types = []
    while pos < len(frame):
        frame_size, pos = _leb128(frame, pos)
        if frame_size is None or pos + frame_size > len(frame):
            return None
        frame_end = pos + frame_size
        while pos < frame_end:
            obu_size, pos = _leb128(frame, pos)
            if obu_size is None or pos + obu_size > frame_end:
                return None
            obu_type, _, _ = _av1_obu_header(frame, pos)
            if obu_type is None:
                return None
            types.append(obu_type)
            pos += obu_size
    return types


def _av1_keyframe(frame: bytes, strict: bool) -> Optional[bool]:
    """Whether ``frame`` is an AV1 temporal unit a decoder can start on.

    ``None`` if it does not parse as AV1. With ``strict`` the unit must open
    with a temporal delimiter or a sequence header, as encoders write them.
    """
    types = _av1_obu_types(frame)
    if types is None:
        return None
    if strict and types[0] not in (_AV1_TEMPORAL_DELIMITER, _AV1_SEQUENCE_HEADER):
        return None
    return _AV1_SEQUENCE_HEADER in types


# -- public ------------------------------------------------------------------


def detect_codec(payload: bytes) -> Optional[VideoCodec]:
    """Guess a payload's codec from its bytes, or ``None`` if nothing is recognised.

    Still images by signature and H.264 / H.265 by their NAL headers are
    settled from the payload alone. VP9 and AV1 are parsed at the frame bytes
    the message framing points to; without a ``format`` hint naming them, only
    a key frame (VP9's sync code, an AV1 unit opening with a delimiter or
    sequence header) is accepted, since an inter frame's header is too short
    to be telling.
    """
    head = payload[:_SIGNATURE_SEARCH_WINDOW]
    if _JPEG_SOI in head:
        return VideoCodec.JPEG
    if _PNG_SIGNATURE in head:
        return VideoCodec.PNG
    codec = _annex_b_codec(payload)
    if codec is not None:
        return codec
    hint = _format_hint(payload)
    for frame in _embedded_frames(payload):
        if (
            hint in (None, VideoCodec.AV1)
            and _av1_keyframe(frame, hint is None) is not None
        ):
            return VideoCodec.AV1
        if (
            hint in (None, VideoCodec.VP9)
            and _vp9_keyframe(frame, hint is None) is not None
        ):
            return VideoCodec.VP9
    return None


def is_keyframe(payload: bytes, codec: VideoCodec) -> bool:
    """Whether a decoder can start on this payload."""
    if codec.every_frame_is_a_keyframe:
        return True
    if codec is VideoCodec.H264:
        return any((first & 0x1F) == _H264_IDR for first, _ in _nal_headers(payload))
    if codec is VideoCodec.H265:
        return any(
            ((first >> 1) & 0x3F) in _H265_IRAP_TYPES
            for first, _ in _nal_headers(payload)
        )
    for frame in _embedded_frames(payload):
        found = (
            _vp9_keyframe(frame, False)
            if codec is VideoCodec.VP9
            else _av1_keyframe(frame, False)
        )
        if found is not None:
            return found
    return False
