"""Unit tests for in-task video decoding on the V2 MCAP reader (``VideoOptions``)."""

import dataclasses
import importlib.util
import io
import os

import pyarrow as pa
import pytest
from pyarrow.fs import LocalFileSystem

from ray.data._internal.datasource_v2.common.listing_utils import sample_files
from ray.data._internal.datasource_v2.formats.mcap import mcap_reader
from ray.data._internal.datasource_v2.formats.mcap.mcap_datasource_v2 import (
    MCAPDatasourceV2,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_decode import (
    FrameDecoder,
    FrameThinner,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_options import (
    TimeRange,
    VideoOptions,
    WindowSpec,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_summary import read_summary
from ray.data._internal.datasource_v2.formats.mcap.mcap_video import (
    VideoCodec,
    detect_codec,
    is_keyframe,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("mcap") is None,
        reason="mcap module not available. Install with: pip install mcap",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("av") is None,
        reason="av not available. Install with: pip install av",
    ),
]

GOP = 10
FRAME_NS = 33_000_000


def encode_h264(num_frames, width=64, height=48):
    """H.264 Annex-B access units, one per frame, a keyframe every ``GOP``."""
    import av
    import numpy as np

    container = av.open(io.BytesIO(), mode="w", format="h264")
    # A fixed GOP, no B-frames and no scene-cut keyframes keep the keyframe
    # positions predictable; without a global header libx264 repeats SPS/PPS
    # before every keyframe, as recorders do.
    stream = container.add_stream(
        "libx264",
        rate=30,
        options={"g": str(GOP), "bf": "0", "sc_threshold": "0", "tune": "zerolatency"},
    )
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    packets = []
    for i in range(num_frames):
        array = np.full((height, width, 3), i * 8 % 256, dtype=np.uint8)
        array[:, :, 1] = (i * 3) % 256
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        packets.extend(bytes(p) for p in stream.encode(frame))
    packets.extend(bytes(p) for p in stream.encode())
    return packets


def write_payloads(
    path,
    payloads,
    *,
    schema_name,
    topic="/camera",
    chunk_size=1 << 20,
    log_times=None,
    extra_topic=None,
):
    """One channel of ``payloads`` at 30 fps (or ``log_times``), plus an optional
    ``extra_topic`` as ``(topic, schema_name, payload)`` written at every step."""
    from mcap.writer import CompressionType, Writer

    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=chunk_size, compression=CompressionType.ZSTD)
        writer.start(profile="", library="ray-test")
        schema_id = writer.register_schema(
            name=schema_name, encoding="ros2msg", data=b"video\n"
        )
        channel_id = writer.register_channel(
            schema_id=schema_id, topic=topic, message_encoding="cdr"
        )
        extra_channel = None
        extra_payload = b""
        if extra_topic is not None:
            extra_schema = writer.register_schema(
                name=extra_topic[1], encoding="ros2msg", data=b"x\n"
            )
            extra_channel = writer.register_channel(
                schema_id=extra_schema, topic=extra_topic[0], message_encoding="cdr"
            )
            extra_payload = extra_topic[2]
        for i, payload in enumerate(payloads):
            log_time = log_times[i] if log_times is not None else i * FRAME_NS
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                publish_time=log_time,
                data=payload,
                sequence=i,
            )
            if extra_channel is not None:
                writer.add_message(
                    channel_id=extra_channel,
                    log_time=log_time,
                    publish_time=log_time,
                    data=extra_payload,
                    sequence=i,
                )
        writer.finish()


@pytest.fixture
def h264_file(tmp_path):
    """30 H.264 access units in several chunks, so chunk cuts fall mid-GOP."""
    path = os.path.join(tmp_path, "h264.mcap")
    write_payloads(
        path, encode_h264(30), schema_name="foxglove.CompressedVideo", chunk_size=400
    )
    summary = read_summary(LocalFileSystem(), path)
    assert summary is not None and len(summary.chunk_indexes) >= 4
    return path


@pytest.fixture
def jpeg_file(tmp_path):
    """10 JPEGs behind a header, the ROS 2 ``sensor_msgs/CompressedImage`` layout."""
    import numpy as np
    from PIL import Image

    payloads = []
    for i in range(10):
        array = np.full((48, 64, 3), i * 20 % 256, dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG")
        payloads.append(b"\x00\x01\x00\x00" + b"header" * 4 + buffer.getvalue())
    path = os.path.join(tmp_path, "jpeg.mcap")
    write_payloads(path, payloads, schema_name="sensor_msgs/msg/CompressedImage")
    return path


def list_manifests(datasource):
    indexer = datasource._get_file_indexer()
    return list(
        indexer.list_files(pa.array(datasource.paths), filesystem=datasource.filesystem)
    )


def read_rows(datasource, manifests, columns=None):
    indexer = datasource._get_file_indexer()
    sample = sample_files(indexer, datasource.paths, datasource.filesystem, [])
    scanner = datasource.create_scanner(
        datasource.infer_schema(sample), datasource.filesystem
    )
    if columns is not None:
        scanner = scanner.prune_columns(columns)
    rows = []
    for manifest in manifests:
        for table in scanner.create_reader().read(manifest):
            assert table.schema.names == scanner.read_schema().names
            rows.extend(table.to_pylist())
    return rows, scanner


def test_encoder_keyframes_are_detected():
    packets = encode_h264(30)
    assert all(detect_codec(p) is VideoCodec.H264 for p in packets)
    keyframes = [i for i, p in enumerate(packets) if is_keyframe(p, VideoCodec.H264)]
    assert keyframes == list(range(0, 30, GOP))


def test_decode_yields_one_frame_per_message(h264_file):
    datasource = MCAPDatasourceV2(
        [h264_file], video=VideoOptions(), include_row_id=True
    )
    rows, scanner = read_rows(datasource, list_manifests(datasource))
    assert len(rows) == 30
    assert [row["sequence"] for row in rows] == list(range(30))
    frame = rows[0]["frame"]
    assert frame.shape == (48, 64, 3) and frame.dtype.name == "uint8"
    assert "data" not in rows[0]
    assert {"topic", "log_time", "publish_time", "channel_id", "row_id"} <= set(rows[0])
    assert str(scanner.read_schema().field("frame").type).startswith(
        "ArrowTensorTypeV2"
    )


def test_decode_primes_a_task_that_starts_mid_gop(h264_file):
    datasource = MCAPDatasourceV2([h264_file], video=VideoOptions())
    (manifest,) = list_manifests(datasource)
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    n = len(manifest)
    assert n >= 3
    # Three tasks; the second and third start inside a group of pictures and
    # must read back to the previous keyframe to decode their first frames.
    parts = [
        FileManifest(block.slice(0, n // 3)),
        FileManifest(block.slice(n // 3, n // 3)),
        FileManifest(block.slice(2 * (n // 3))),
    ]
    whole, _ = read_rows(datasource, [manifest])
    split, _ = read_rows(datasource, parts)
    assert sorted(r["sequence"] for r in split) == sorted(r["sequence"] for r in whole)
    assert len(split) == 30
    by_seq = {r["sequence"]: r["frame"] for r in whole}
    for row in split:
        assert (row["frame"] == by_seq[row["sequence"]]).all()


def test_decode_attributes_frames_of_messages_sharing_a_log_time(tmp_path):
    """Two messages stamped alike each get their own frame, in order."""
    log_times = [i * FRAME_NS for i in range(30)]
    log_times[6] = log_times[5]  # frames 5 and 6 share a timestamp
    log_times[29] = log_times[28]  # and so do the last two, drained by flush
    path = os.path.join(tmp_path, "dup.mcap")
    write_payloads(
        path,
        encode_h264(30),
        schema_name="foxglove.CompressedVideo",
        chunk_size=400,
        log_times=log_times,
    )
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    rows, _ = read_rows(datasource, list_manifests(datasource))
    assert sorted(row["sequence"] for row in rows) == list(range(30))
    assert [row["log_time"] for row in rows] == sorted(log_times)
    by_seq = {row["sequence"]: row["frame"] for row in rows}
    reference = MCAPDatasourceV2([path], video=VideoOptions(), include_row_id=True)
    (manifest,) = list_manifests(reference)
    whole, _ = read_rows(reference, [manifest])
    for row in whole:
        assert (by_seq[row["sequence"]] == row["frame"]).all()


def test_planning_checks_every_selected_topic(tmp_path):
    """A non-video topic after the video one, or with ``resize`` set, still fails
    at planning rather than in a read task."""
    path = os.path.join(tmp_path, "mixed.mcap")
    write_payloads(
        path,
        encode_h264(10),
        schema_name="foxglove.CompressedVideo",
        extra_topic=("/imu", "sensor_msgs/msg/Imu", b"\x01\x02\x03\x04" * 8),
    )
    for video in (
        VideoOptions(),
        VideoOptions(resize=(24, 32)),
    ):
        datasource = MCAPDatasourceV2([path], video=video)
        with pytest.raises(ValueError, match="'/imu'.*not a video topic"):
            infer_frame_schema(datasource)
    only_camera = MCAPDatasourceV2([path], topics=["/camera"], video=VideoOptions())
    rows, _ = read_rows(only_camera, list_manifests(only_camera))
    assert len(rows) == 10


def infer_frame_schema(datasource):
    indexer = datasource._get_file_indexer()
    sample = sample_files(indexer, datasource.paths, datasource.filesystem, [])
    return datasource.infer_schema(sample)


def test_frame_thinner_never_rewinds():
    thinner = FrameThinner(100)
    thinner.observe(250)  # the lead-in already kept a frame in interval 2
    assert thinner.keep(260) is False
    # A frame released late from an earlier interval neither survives nor
    # reopens interval 2 for the frame after it.
    assert thinner.keep(150) is False
    assert thinner.keep(270) is False
    assert thinner.keep(310) is True
    assert FrameThinner(None).keep(5) is True


def write_two_cameras(path, offset_frames):
    """Two H.264 cameras, one chunk per message; ``/b`` runs ``offset_frames``
    behind ``/a`` with the same encoded stream."""
    from mcap.writer import CompressionType, Writer

    packets = encode_h264(30)
    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=1, compression=CompressionType.ZSTD)
        writer.start(profile="", library="ray-test")
        schema_id = writer.register_schema(
            name="foxglove.CompressedVideo", encoding="ros2msg", data=b"video\n"
        )
        channels = {
            topic: writer.register_channel(
                schema_id=schema_id, topic=topic, message_encoding="cdr"
            )
            for topic in ("/a", "/b")
        }
        for i, payload in enumerate(packets):
            for topic, shift in (("/a", 0), ("/b", offset_frames)):
                log_time = (i + shift) * FRAME_NS
                writer.add_message(
                    channel_id=channels[topic],
                    log_time=log_time,
                    publish_time=log_time,
                    data=payload,
                    sequence=i,
                )
        writer.finish()
    return channels


def test_each_video_channel_gets_its_own_lead_in(tmp_path):
    """A task owning frames 26-29 of both cameras, with ``/b`` eight frames
    behind: ``/b``'s keyframe (frame 20) lies after ``/a``'s first owned frame
    and in chunks the task does not own, and must still prime ``/b``."""
    path = os.path.join(tmp_path, "two.mcap")
    channels = write_two_cameras(path, offset_frames=8)
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    (manifest,) = list_manifests(datasource)
    summary = read_summary(LocalFileSystem(), path)
    assert summary is not None and len(summary.chunk_indexes) == 60

    def frame_of(chunk):  # one message per chunk
        (channel_id,) = chunk.message_index_offsets
        shift = 8 if channel_id == channels["/b"] else 0
        return channel_id, chunk.message_start_time // FRAME_NS - shift

    owned_offsets = {
        c.chunk_start_offset for c in summary.chunk_indexes if frame_of(c)[1] >= 26
    }
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    rows = [
        i
        for i, md in enumerate(manifest.file_chunk_metadatas)
        if md is not None and int(md["unit_ids"][0]) in owned_offsets
    ]
    assert len(rows) == 8
    part = FileManifest(block.take(rows))

    split, _ = read_rows(datasource, [part])
    assert sorted((r["topic"], r["sequence"]) for r in split) == [
        (topic, seq) for topic in ("/a", "/b") for seq in range(26, 30)
    ]
    whole, _ = read_rows(datasource, [manifest])
    reference = {(r["topic"], r["sequence"]): r["frame"] for r in whole}
    for row in split:
        assert (row["frame"] == reference[(row["topic"], row["sequence"])]).all()


def manifest_for_frames(datasource, path, channels, offset_frames, frames):
    """A manifest owning exactly ``frames`` (``(topic, index)`` pairs) of a file
    written with one chunk per message."""
    (manifest,) = list_manifests(datasource)
    summary = read_summary(LocalFileSystem(), path)
    assert summary is not None
    by_channel = {cid: topic for topic, cid in channels.items()}
    wanted_offsets = set()
    for chunk in summary.chunk_indexes:
        (channel_id,) = chunk.message_index_offsets
        topic = by_channel[channel_id]
        shift = offset_frames if topic == "/b" else 0
        if (topic, chunk.message_start_time // FRAME_NS - shift) in frames:
            wanted_offsets.add(chunk.chunk_start_offset)
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    rows = [
        i
        for i, md in enumerate(manifest.file_chunk_metadatas)
        if md is not None and int(md["unit_ids"][0]) in wanted_offsets
    ]
    assert len(rows) == len(frames)
    return FileManifest(block.take(rows)), manifest


class HeldBackDecoder(FrameDecoder):
    """A decoder that releases every frame one packet late, as a real one may."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._held = []

    def decode(self, message):
        ready, self._held = self._held, list(super().decode(message))
        yield from ready

    def flush(self):
        yield from self._held
        self._held = []
        yield from super().flush()


@pytest.mark.parametrize("held_back", [False, True], ids=["prompt", "held-back"])
def test_gaps_between_owned_chunks_are_fed_to_the_decoder(
    tmp_path, monkeypatch, held_back
):
    """Frames another task owns (or a checkpoint excluded) in the middle of a
    channel's stream are decoded for state so the frames after them are right,
    and a frame of ours the decoder held back across the gap is still emitted."""
    if held_back:
        monkeypatch.setattr(mcap_reader, "FrameDecoder", HeldBackDecoder)
    path = os.path.join(tmp_path, "gaps.mcap")
    channels = write_two_cameras(path, offset_frames=0)
    datasource = MCAPDatasourceV2([path], topics=["/a"], video=VideoOptions())
    # A gap inside a GOP (frames 15-16 missing) and one spanning a keyframe
    # (frames 10-21 missing, keyframe 20 among them).
    for owned_frames in (
        list(range(0, 15)) + list(range(17, 30)),
        list(range(0, 10)) + list(range(22, 30)),
    ):
        part, whole_manifest = manifest_for_frames(
            datasource, path, channels, 0, {("/a", i) for i in owned_frames}
        )
        rows, _ = read_rows(datasource, [part])
        assert [r["sequence"] for r in rows] == owned_frames
        whole, _ = read_rows(datasource, [whole_manifest])
        reference = {r["sequence"]: r["frame"] for r in whole}
        for row in rows:
            assert (row["frame"] == reference[row["sequence"]]).all()


def split_at_first_idr(packet):
    """Cut an access unit before its IDR NAL unit (type 5), start code included."""
    i = 0
    while True:
        j = packet.find(b"\x00\x00\x01", i)
        assert j >= 0, "no IDR NAL unit"
        if packet[j + 3] & 0x1F == 5:
            cut = j - 1 if j > 0 and packet[j - 1] == 0 else j
            return packet[:cut], packet[cut:]
        i = j + 3


def test_planning_decodes_parameter_sets_written_separately(tmp_path):
    """SPS and PPS in a message of their own ahead of the keyframe: planning
    plays the stream head in order instead of the lone keyframe."""
    packets = encode_h264(10)
    parameter_sets, keyframe = split_at_first_idr(packets[0])
    assert detect_codec(parameter_sets) is VideoCodec.H264
    assert not is_keyframe(parameter_sets, VideoCodec.H264)
    assert is_keyframe(keyframe, VideoCodec.H264)
    path = os.path.join(tmp_path, "split.mcap")
    write_payloads(
        path,
        [parameter_sets, keyframe] + packets[1:],
        schema_name="foxglove.CompressedVideo",
        log_times=[0, 0] + [i * FRAME_NS for i in range(1, 10)],
    )
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    schema = infer_frame_schema(datasource)
    assert "shape=(48, 64, 3)" in str(schema.field("frame").type)
    rows, _ = read_rows(datasource, list_manifests(datasource))
    assert all(row["frame"].shape == (48, 64, 3) for row in rows)
    # The keyframe's frame belongs to the keyframe message (sequence 1), not to
    # the parameter-set message stamped alike (sequence 0), which is no row.
    assert [row["sequence"] for row in rows] == list(range(1, 11))
    pruned, _ = read_rows(datasource, list_manifests(datasource), columns=["sequence"])
    assert [row["sequence"] for row in pruned] == list(range(1, 11))


def test_pruned_projection_counts_the_frames_a_decoder_would_emit(
    h264_file, monkeypatch
):
    """``count()`` (no ``frame`` column) and a full read agree on the rows: a
    channel that starts without a keyframe in reach yields nothing until its
    next keyframe either way."""
    monkeypatch.setenv("RAY_DATA_MCAP_MAX_LEAD_IN_S", "0.05")
    datasource = MCAPDatasourceV2([h264_file], video=VideoOptions())
    (manifest,) = list_manifests(datasource)
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    summary = read_summary(LocalFileSystem(), h264_file)
    assert summary is not None
    # Chunks starting mid-GOP (frames 8 and 15 in this fixture): the first
    # GOP's keyframe (frame 0) is further back than 50 ms, so decoding can only
    # start at the keyframe at frame 10.
    rows = [
        i
        for i, c in enumerate(summary.chunk_indexes)
        if 2 <= c.message_start_time // FRAME_NS < 20
    ]
    assert rows
    part = FileManifest(block.take(rows))
    decoded, _ = read_rows(datasource, [part])
    counted, _ = read_rows(datasource, [part], columns=["sequence"])
    assert [r["sequence"] for r in decoded] == [r["sequence"] for r in counted]
    assert decoded and all(r["sequence"] >= 10 for r in decoded)


def test_lead_in_parameter_sets_do_not_take_an_fps_interval(tmp_path):
    """A parameter-set message in the lead-in is no frame, so it must not use
    up the interval of the first owned frame stamped just after it."""
    packets = encode_h264(10)
    parameter_sets, keyframe = split_at_first_idr(packets[0])
    path = os.path.join(tmp_path, "ps.mcap")
    # Parameter sets at 1 ms, the keyframe at 50 ms: the same 100 ms interval.
    log_times = [1_000_000, 50_000_000] + [i * 100_000_000 for i in range(1, 10)]
    write_payloads(
        path,
        [parameter_sets, keyframe] + packets[1:],
        schema_name="foxglove.CompressedVideo",
        chunk_size=1,
        log_times=log_times,
    )
    datasource = MCAPDatasourceV2([path], video=VideoOptions(fps=10))
    (manifest,) = list_manifests(datasource)
    whole, _ = read_rows(datasource, [manifest])
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    # A task owning everything but the parameter-set message: it is lead-in.
    split, _ = read_rows(datasource, [FileManifest(block.slice(1))])
    assert [r["sequence"] for r in whole] == list(range(1, 11))
    assert [r["sequence"] for r in split] == [r["sequence"] for r in whole]


def test_gap_longer_than_the_lead_in_goes_cold_until_a_keyframe(tmp_path, monkeypatch):
    """A gap that cannot be fed whole and holds no keyframe in its tail leaves
    the decoder without references: the frames after it are skipped until the
    next keyframe rather than decoded against the wrong pictures."""
    path = os.path.join(tmp_path, "longgap.mcap")
    channels = write_two_cameras(path, offset_frames=0)
    monkeypatch.setenv("RAY_DATA_MCAP_MAX_LEAD_IN_S", "0.1")
    datasource = MCAPDatasourceV2([path], topics=["/a"], video=VideoOptions())
    # Frames 5-17 belong to another task; only 15-17 fit in a 100 ms lead-in and
    # none of them is a keyframe (10 lies outside it), so 18 and 19 are skipped.
    owned_frames = list(range(0, 5)) + list(range(18, 30))
    part, whole_manifest = manifest_for_frames(
        datasource, path, channels, 0, {("/a", i) for i in owned_frames}
    )
    rows, _ = read_rows(datasource, [part])
    expected = list(range(0, 5)) + list(range(20, 30))
    assert [r["sequence"] for r in rows] == expected
    whole, _ = read_rows(datasource, [whole_manifest])
    reference = {r["sequence"]: r["frame"] for r in whole}
    for row in rows:
        assert (row["frame"] == reference[row["sequence"]]).all()
    counted, _ = read_rows(datasource, [part], columns=["sequence"])
    assert [r["sequence"] for r in counted] == expected


def test_zero_look_back_still_goes_cold_after_a_skipped_span(tmp_path):
    """With the look-back cap at zero nothing is read back, but frames another
    task owns between two of ours still leave the decoder without references:
    cold until the next keyframe, in both projections."""
    path = os.path.join(tmp_path, "nolookback.mcap")
    channels = write_two_cameras(path, offset_frames=0)
    datasource = MCAPDatasourceV2([path], topics=["/a"], video=VideoOptions())
    owned_frames = list(range(0, 5)) + list(range(17, 30))
    part, _ = manifest_for_frames(
        datasource, path, channels, 0, {("/a", i) for i in owned_frames}
    )
    scanner = datasource.create_scanner(
        infer_frame_schema(datasource), datasource.filesystem
    )
    scanner = dataclasses.replace(scanner, max_lead_in_ns=0)
    expected = list(range(0, 5)) + list(range(20, 30))  # 17-19 follow a skipped span

    def sequences(s):
        return [
            row["sequence"]
            for table in s.create_reader().read(part)
            for row in table.to_pylist()
        ]

    assert sequences(scanner) == expected
    assert sequences(scanner.prune_columns(["sequence"])) == expected


@pytest.mark.parametrize(
    "schema_name",
    ["foxglove.CompressedVideo", "custom_msgs/msg/Frame"],
    ids=["video-schema", "custom-schema"],
)
def test_planning_tells_the_codec_from_the_stream_head(tmp_path, schema_name):
    """A sample file that starts mid-GOP on VP9 inter frames still plans: the
    codec comes from the first keyframe further on."""
    payloads = [VP9_KEYFRAME if i % 10 == 2 else VP9_PFRAME for i in range(30)]
    path = os.path.join(tmp_path, "midgop.mcap")
    write_payloads(path, payloads, schema_name=schema_name, chunk_size=1)
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    schema = infer_frame_schema(datasource)
    assert "ArrowVariableShapedTensorType" in str(schema.field("frame").type)
    assert datasource._video_topics.recognises("/camera", None) is True


def test_cold_frames_still_hold_their_fps_interval(h264_file):
    """Frames skipped while a channel is cold were kept by the whole-file read,
    so they still take their ``fps`` interval: the keyframe that warms the
    channel does not open an interval the whole-file read already spent."""
    datasource = MCAPDatasourceV2([h264_file], video=VideoOptions(fps=5))
    (manifest,) = list_manifests(datasource)
    whole, _ = read_rows(datasource, [manifest])
    assert [r["sequence"] for r in whole] == [0, 7, 13, 19, 25]
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    scanner = datasource.create_scanner(
        infer_frame_schema(datasource), datasource.filesystem
    )
    scanner = dataclasses.replace(scanner, max_lead_in_ns=50_000_000)
    # Chunks from frame 8 on, with the keyframe at 0 out of reach: frames 8
    # and 9 are cold, 10 warms the channel but shares its 200 ms interval with
    # them (and with frame 7, which the whole-file read kept), so the first
    # frame this task keeps is 13.
    part = FileManifest(block.slice(2))
    expected = [13, 19, 25]
    for projection in (scanner, scanner.prune_columns(["sequence"])):
        rows = [
            row["sequence"]
            for table in projection.create_reader().read(part)
            for row in table.to_pylist()
        ]
        assert rows == expected


def test_time_gap_without_skipped_messages_keeps_decoding(tmp_path):
    """A camera that pauses for longer than the look-back cap skipped nothing:
    the decoder keeps its state and the frames after the pause decode."""
    log_times = [i * FRAME_NS for i in range(30)]
    for i in range(15, 30):
        log_times[i] += 20 * 1_000_000_000  # a 20 s pause mid-GOP
    path = os.path.join(tmp_path, "pause.mcap")
    write_payloads(
        path,
        encode_h264(30),
        schema_name="foxglove.CompressedVideo",
        chunk_size=1,
        log_times=log_times,
    )
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    (manifest,) = list_manifests(datasource)
    whole, _ = read_rows(datasource, [manifest])
    assert [r["sequence"] for r in whole] == list(range(30))
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    # A task owning frames 12-19, one chunk per frame, straddles the pause.
    rows, _ = read_rows(datasource, [FileManifest(block.slice(12, 8))])
    assert [r["sequence"] for r in rows] == list(range(12, 20))
    reference = {r["sequence"]: r["frame"] for r in whole}
    for row in rows:
        assert (row["frame"] == reference[row["sequence"]]).all()


VP9_KEYFRAME = b"\x82\x49\x83\x42" + bytes(16)
VP9_PFRAME = b"\x86" + bytes(16)


@pytest.mark.parametrize(
    "schema_name",
    ["foxglove.CompressedVideo", "custom_msgs/msg/Frame"],
    ids=["video-schema", "custom-schema"],
)
def test_mid_gop_task_tells_the_codec_from_its_lead_in(tmp_path, schema_name):
    """A VP9 inter frame does not name its codec; a task starting on one takes
    it from the keyframe in its lead-in, and goes cold rather than failing
    when no payload in reach tells, also for a topic planning recognised from
    its bytes rather than its schema name."""
    payloads = [VP9_KEYFRAME if i % 10 == 0 else VP9_PFRAME for i in range(30)]
    assert detect_codec(VP9_PFRAME) is None
    path = os.path.join(tmp_path, "vp9.mcap")
    write_payloads(path, payloads, schema_name=schema_name, chunk_size=1)
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    (manifest,) = list_manifests(datasource)
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    # The synthetic frames carry no picture data a decoder accepts, so the
    # projection without ``frame`` is what shows which rows the task yields.
    counted, _ = read_rows(
        datasource, [FileManifest(block.slice(12, 8))], columns=["sequence"]
    )
    assert [r["sequence"] for r in counted] == list(range(12, 20))
    read_rows(datasource, [FileManifest(block.slice(12, 8))])  # must not raise

    # With the keyframe out of reach the codec is unknown: cold, not an error,
    # until the keyframe at frame 20 tells it.
    cold = MCAPDatasourceV2([path], video=VideoOptions())
    scanner = cold.create_scanner(infer_frame_schema(cold), cold.filesystem)
    scanner = dataclasses.replace(scanner, max_lead_in_ns=50_000_000)
    rows = [
        row
        for table in scanner.prune_columns(["sequence"])
        .create_reader()
        .read(FileManifest(block.slice(12, 14)))
        for row in table.to_pylist()
    ]
    assert [r["sequence"] for r in rows] == list(range(20, 26))


def test_decode_time_range_starting_mid_gop(h264_file):
    start, end = GOP + GOP // 2, GOP + GOP // 2 + 5
    datasource = MCAPDatasourceV2(
        [h264_file],
        time_range=TimeRange(start * FRAME_NS, end * FRAME_NS),
        video=VideoOptions(),
    )
    rows, _ = read_rows(datasource, list_manifests(datasource))
    assert [row["sequence"] for row in rows] == list(range(start, end))


def test_decode_resize_and_fps(h264_file):
    resized = MCAPDatasourceV2([h264_file], video=VideoOptions(resize=(24, 32)))
    rows, scanner = read_rows(resized, list_manifests(resized))
    assert len(rows) == 30 and rows[0]["frame"].shape == (24, 32, 3)
    assert "shape=(24, 32, 3)" in str(scanner.read_schema().field("frame").type)

    # 30 frames at 33 ms span ten 100 ms intervals: one frame survives in each.
    thinned = MCAPDatasourceV2([h264_file], video=VideoOptions(fps=10))
    rows, _ = read_rows(thinned, list_manifests(thinned))
    assert len(rows) == 10
    assert [row["log_time"] // 100_000_000 for row in rows] == list(range(10))
    # The same frames survive however the read is split, and whether or not
    # ``frame`` is projected.
    (manifest,) = list_manifests(thinned)
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    parts = [FileManifest(block.slice(0, 2)), FileManifest(block.slice(2))]
    split, _ = read_rows(thinned, parts)
    assert sorted(r["sequence"] for r in split) == [r["sequence"] for r in rows]
    pruned, _ = read_rows(thinned, [manifest], columns=["sequence"])
    assert [r["sequence"] for r in pruned] == [r["sequence"] for r in rows]


def test_decode_embedded_jpeg(jpeg_file):
    datasource = MCAPDatasourceV2([jpeg_file], video=VideoOptions())
    rows, _ = read_rows(datasource, list_manifests(datasource))
    assert len(rows) == 10
    assert rows[0]["frame"].shape == (48, 64, 3)
    assert rows[3]["frame"][0, 0, 0] in range(55, 66)  # JPEG is lossy

    resized = MCAPDatasourceV2([jpeg_file], video=VideoOptions(resize=(24, 32)))
    rows, _ = read_rows(resized, list_manifests(resized))
    assert [row["frame"].shape for row in rows] == [(24, 32, 3)] * 10


def test_corrupt_still_is_skipped_not_fatal(tmp_path):
    """One truncated JPEG costs one frame, not the task."""
    import numpy as np
    from PIL import Image

    payloads = []
    for i in range(6):
        buffer = io.BytesIO()
        Image.fromarray(np.full((48, 64, 3), i * 20, dtype=np.uint8)).save(
            buffer, format="JPEG"
        )
        payloads.append(buffer.getvalue())
    good = list(payloads)
    payloads[2] = payloads[2][:40]  # still sniffs as JPEG, cannot be decoded
    assert detect_codec(payloads[2]) is VideoCodec.JPEG
    path = os.path.join(tmp_path, "corrupt.mcap")
    write_payloads(path, payloads, schema_name="sensor_msgs/msg/CompressedImage")
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    rows, _ = read_rows(datasource, list_manifests(datasource))
    assert [row["sequence"] for row in rows] == [0, 1, 3, 4, 5]

    # Under ``fps`` a corrupt still must not take the interval of the good
    # still after it: frames 3-5 share a 100 ms interval, 3 is corrupt.
    payloads[2], payloads[3] = good[2], payloads[3][:40]
    path = os.path.join(tmp_path, "corrupt_fps.mcap")
    write_payloads(path, payloads, schema_name="sensor_msgs/msg/CompressedImage")
    thinned = MCAPDatasourceV2([path], video=VideoOptions(fps=10))
    rows, _ = read_rows(thinned, list_manifests(thinned))
    assert [row["sequence"] for row in rows] == [0, 4]


def test_decoded_builder_refuses_a_row_without_its_frame():
    from ray.data._internal.datasource_v2.formats.mcap.mcap_reader import (
        _MessageTableBuilder,
    )

    class Stub:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    selected = (
        None,
        Stub(id=1, topic="/cam", message_encoding="cdr", metadata={}),
        Stub(channel_id=1, log_time=0, publish_time=0, sequence=0, data=b"x"),
        "row",
    )
    wanted = _MessageTableBuilder(
        columns=None,
        include_metadata=False,
        include_row_id=False,
        decode_json=False,
        decoded=True,
    )
    with pytest.raises(AssertionError, match="without its frame"):
        wanted.add(selected)  # pyrefly: ignore[bad-argument-type]
    pruned = _MessageTableBuilder(
        columns={"topic"},
        include_metadata=False,
        include_row_id=False,
        decode_json=False,
        decoded=True,
    )
    pruned.add(selected)  # pyrefly: ignore[bad-argument-type]
    assert pruned.build().column_names == ["topic"]


def test_frame_pruned_away_skips_decoding(h264_file):
    datasource = MCAPDatasourceV2([h264_file], video=VideoOptions())
    rows, _ = read_rows(
        datasource, list_manifests(datasource), columns=["topic", "sequence"]
    )
    assert len(rows) == 30 and set(rows[0]) == {"topic", "sequence"}


def test_decode_rejects_non_video_topics(tmp_path):
    path = os.path.join(tmp_path, "imu.mcap")
    write_payloads(
        path,
        [b"\x01\x02\x03\x04" * 8] * 5,
        schema_name="sensor_msgs/msg/Imu",
        topic="/imu",
    )
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    with pytest.raises(ValueError, match="'/imu'.*not a video topic"):
        read_rows(datasource, list_manifests(datasource))

    # A topic whose schema says video but whose bytes fit no codec cannot be
    # decoded either, and says so by name.
    opaque = os.path.join(tmp_path, "opaque.mcap")
    write_payloads(
        opaque,
        [b"\x01\x02\x03\x04" * 8] * 5,
        schema_name="foxglove.CompressedVideo",
        topic="/cam",
    )
    datasource = MCAPDatasourceV2([opaque], video=VideoOptions())
    with pytest.raises(ValueError, match="'/cam'.*not JPEG, PNG, H.264"):
        read_rows(datasource, list_manifests(datasource))


def test_video_option_validation(h264_file):
    with pytest.raises(ValueError, match="positive number"):
        VideoOptions(fps=0)
    with pytest.raises(ValueError, match="height, width"):
        VideoOptions(resize=(10, 0))
    assert VideoOptions(fps=5).fps_interval_ns == 200_000_000
    assert VideoOptions(resize=(24, 32)).resize == (24, 32)
    # Decoding is offered where a row's decoded frames stay bounded: a message
    # (one frame) or a window; a topic or file row would be the whole stream.
    MCAPDatasourceV2(
        [h264_file],
        read_granularity="window",
        window=WindowSpec(length_s=1),
        video=VideoOptions(),
    )
    for granularity in ("topic", "file", "attachment", "metadata"):
        with pytest.raises(ValueError, match="cannot be cut into blocks"):
            MCAPDatasourceV2(
                [h264_file], read_granularity=granularity, video=VideoOptions()
            )


# -- decoded window rows ------------------------------------------------------


def window_datasource(path, length_s, stride_s=None, **video):
    return MCAPDatasourceV2(
        [path],
        read_granularity="window",
        window=WindowSpec(length_s=length_s, stride_s=stride_s),
        video=VideoOptions(**video),
    )


def split_manifest(manifest, parts):
    """Cut a manifest's chunk rows into ``parts`` consecutive tasks."""
    block = manifest.as_block()
    assert isinstance(block, pa.Table)
    n = len(manifest)
    assert n >= parts
    bounds = [round(i * n / parts) for i in range(parts + 1)]
    return [
        FileManifest(block.slice(lo, hi - lo)) for lo, hi in zip(bounds, bounds[1:])
    ]


def frames_by_sequence(path):
    """Ground truth: every frame of the file decoded at message granularity."""
    datasource = MCAPDatasourceV2([path], video=VideoOptions())
    rows, _ = read_rows(datasource, list_manifests(datasource))
    return {row["sequence"]: row["frame"] for row in rows}


def by_window(rows):
    return {row["window_start"]: row for row in rows}


def test_decoded_windows_match_the_whole_file_read(h264_file):
    """Window rows carry the window's frames per topic; a read split into three
    tasks, each starting mid-GOP, gives the same frames as one task, and the
    same frames as the message-granularity decode."""
    import numpy as np

    datasource = window_datasource(h264_file, length_s=0.33)
    (manifest,) = list_manifests(datasource)
    whole, scanner = read_rows(datasource, [manifest])
    split, _ = read_rows(datasource, split_manifest(manifest, 3))
    names = scanner.read_schema().names
    assert names[-2:] == ["frames:/camera", "frame_times:/camera"]
    assert str(scanner.read_schema().field("frames:/camera").type).startswith(
        "ArrowVariableShapedTensorType"
    )
    assert sorted(by_window(whole)) == [0, 330_000_000, 660_000_000]
    assert sorted(by_window(split)) == sorted(by_window(whole))
    truth = frames_by_sequence(h264_file)
    for start, row in by_window(whole).items():
        other = by_window(split)[start]
        times = row["frame_times:/camera"]
        assert (
            times
            == other["frame_times:/camera"]
            == [
                t
                for t in range(0, 30 * FRAME_NS, FRAME_NS)
                if start <= t < start + 330_000_000
            ]
        )
        frames = row["frames:/camera"]
        assert frames.shape == (10, 48, 64, 3) and frames.dtype == np.uint8
        assert np.array_equal(frames, other["frames:/camera"])
        for t, frame in zip(times, frames):
            assert np.array_equal(frame, truth[t // FRAME_NS])
        # The camera's messages left the lists; the lead-in went to the decoder.
        assert row["num_messages"] == 0 and row["num_lead_in"] == 0
        assert row["topic"] == [] and row["data"] == []


def test_decoded_window_opening_mid_gop_starts_on_its_own_first_frame(h264_file):
    """A window that opens at frame 16, inside the second GOP, is decoded from
    the keyframe at frame 10 by whichever task owns it."""
    import numpy as np

    datasource = window_datasource(h264_file, length_s=0.5)
    (manifest,) = list_manifests(datasource)
    truth = frames_by_sequence(h264_file)
    for manifests in ([manifest], split_manifest(manifest, 3)):
        rows, _ = read_rows(datasource, manifests)
        second = by_window(rows)[500_000_000]
        assert second["frame_times:/camera"][0] == 16 * FRAME_NS
        assert np.array_equal(second["frames:/camera"][0], truth[16])
        assert len(second["frame_times:/camera"]) == 14


def test_decoded_windows_overlap_and_bound_their_frames(h264_file):
    """Overlapping windows duplicate frames on purpose, and no window holds a
    frame outside its span."""
    datasource = window_datasource(h264_file, length_s=0.33, stride_s=0.165)
    (manifest,) = list_manifests(datasource)
    rows, _ = read_rows(datasource, split_manifest(manifest, 2))
    holding = {}
    for row in rows:
        start, end = row["window_start"], row["window_end"]
        for t in row["frame_times:/camera"]:
            assert start <= t < end
            holding.setdefault(t, []).append(start)
    # Frame 6 (198 ms) sits in [0, 330) and [165, 495); frame 0 only in the first.
    assert sorted(holding[6 * FRAME_NS]) == [0, 165_000_000]
    assert holding[0] == [0]
    assert sorted(by_window(rows)) == [
        0,
        165_000_000,
        330_000_000,
        495_000_000,
        660_000_000,
        825_000_000,
    ]


def write_two_cameras_and_an_imu(path):
    from mcap.writer import CompressionType, Writer

    packets = encode_h264(30)
    with open(path, "wb") as stream:
        writer = Writer(stream, chunk_size=400, compression=CompressionType.ZSTD)
        writer.start(profile="", library="ray-test")
        video_schema = writer.register_schema(
            name="foxglove.CompressedVideo", encoding="ros2msg", data=b"video\n"
        )
        imu_schema = writer.register_schema(
            name="sensor_msgs/msg/Imu", encoding="ros2msg", data=b"imu\n"
        )
        cameras = {
            topic: writer.register_channel(
                schema_id=video_schema, topic=topic, message_encoding="cdr"
            )
            for topic in ("/cam_a", "/cam_b")
        }
        imu = writer.register_channel(
            schema_id=imu_schema, topic="/imu", message_encoding="cdr"
        )
        for i, payload in enumerate(packets):
            log_time = i * FRAME_NS
            for channel in cameras.values():
                writer.add_message(
                    channel_id=channel,
                    log_time=log_time,
                    publish_time=log_time,
                    data=payload,
                    sequence=i,
                )
            writer.add_message(
                channel_id=imu,
                log_time=log_time,
                publish_time=log_time,
                data=bytes([i % 256]) * 16,
                sequence=i,
            )
        writer.finish()


def test_decoded_windows_keep_the_other_topics_in_the_lists(tmp_path):
    """Two cameras give two column pairs; the IMU stays in the message lists
    and is the only channel ``channels`` describes."""
    path = os.path.join(tmp_path, "rig.mcap")
    write_two_cameras_and_an_imu(path)
    datasource = window_datasource(path, length_s=0.33)
    (manifest,) = list_manifests(datasource)
    rows, scanner = read_rows(datasource, split_manifest(manifest, 3))
    assert scanner.read_schema().names[-4:] == [
        "frames:/cam_a",
        "frame_times:/cam_a",
        "frames:/cam_b",
        "frame_times:/cam_b",
    ]
    assert len(rows) == 3
    for row in rows:
        assert (
            row["frames:/cam_a"].shape == row["frames:/cam_b"].shape == (10, 48, 64, 3)
        )
        assert row["topic"] == ["/imu"] * 10 and row["num_messages"] == 10
        assert len(row["log_time"]) == 10 and len(row["data"]) == 10
        assert [c["topic"] for c in row["channels"]] == ["/imu"]


def test_decoded_windows_fps_and_resize_do_not_depend_on_the_split(h264_file):
    import numpy as np

    datasource = window_datasource(h264_file, length_s=0.33, fps=10, resize=(24, 32))
    (manifest,) = list_manifests(datasource)
    whole, _ = read_rows(datasource, [manifest])
    split, _ = read_rows(datasource, split_manifest(manifest, 3))
    kept = []
    for start, row in sorted(by_window(whole).items()):
        other = by_window(split)[start]
        assert row["frame_times:/camera"] == other["frame_times:/camera"]
        assert np.array_equal(row["frames:/camera"], other["frames:/camera"])
        assert row["frames:/camera"].shape[1:] == (24, 32, 3)
        kept.extend(row["frame_times:/camera"])
    # One frame per 100 ms interval of the epoch grid, over 990 ms of video.
    assert kept == sorted(kept) and len(kept) == 10
    assert len({t // 100_000_000 for t in kept}) == 10


def test_decoded_windows_pruned_frames_skip_the_decoder(h264_file, monkeypatch):
    """With ``frames:`` pruned no decoder is built, yet ``frame_times:`` still
    lists the frames a decoder would keep."""

    class NoDecoder(FrameDecoder):
        def __init__(self, *args, **kwargs):
            raise AssertionError("frames were pruned; nothing should decode")

    monkeypatch.setattr(mcap_reader, "FrameDecoder", NoDecoder)
    datasource = window_datasource(h264_file, length_s=0.33, fps=10)
    (manifest,) = list_manifests(datasource)
    rows, _ = read_rows(
        datasource,
        split_manifest(manifest, 3),
        columns=["window_start", "frame_times:/camera"],
    )
    assert sorted(by_window(rows)) == [0, 330_000_000, 660_000_000]
    assert sum(len(row["frame_times:/camera"]) for row in rows) == 10
    assert set(rows[0]) == {"window_start", "frame_times:/camera"}


def test_decoded_window_over_the_row_limit_fails_early(h264_file, monkeypatch):
    monkeypatch.setenv("RAY_DATA_MCAP_MAX_ROW_BYTES", str(48 * 64 * 3 * 2))
    datasource = window_datasource(h264_file, length_s=0.33)
    with pytest.raises(ValueError, match=r"window \[0, 330000000\).*fps"):
        read_rows(datasource, list_manifests(datasource))


def test_emitted_windows_release_their_frames(h264_file, monkeypatch):
    """A window that has been emitted keeps no frames or messages behind in the
    task's pending list: the row owns them, and memory holds only the windows
    still in flight."""
    created = []

    class Tracking(mcap_reader._PendingWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(mcap_reader, "_PendingWindow", Tracking)
    datasource = window_datasource(h264_file, length_s=0.33)
    rows, _ = read_rows(datasource, list_manifests(datasource))
    assert sorted(by_window(rows)) == [0, 330_000_000, 660_000_000]
    assert all(len(row["frame_times:/camera"]) == 10 for row in rows)
    assert len(created) == 3
    assert all(not window.messages and not window.frames for window in created)
    assert all(window.decoded_bytes == 0 for window in created)


def test_cold_channel_warns_once_per_topic(tmp_path, monkeypatch, caplog):
    """A channel with no keyframe within the look-back cap goes cold, and says
    so once per topic, naming the topic and the cap, on the message path and
    on the window path; a stream that simply begins mid-GOP at the recording's
    own start does not warn."""
    from ray.util.debug import reset_log_once

    path = os.path.join(tmp_path, "coldwarn.mcap")
    channels = write_two_cameras(path, offset_frames=0)
    monkeypatch.setenv("RAY_DATA_MCAP_MAX_LEAD_IN_S", "0.1")
    owned = {("/a", i) for i in range(15, 30)}

    def cold_warnings():
        return [
            record.message
            for record in caplog.records
            if "skipped until the next keyframe" in record.message
        ]

    # Ray's loggers do not propagate to the root logger caplog listens on.
    mcap_reader.logger.addHandler(caplog.handler)
    try:
        # Message rows: the task owns frames 15-29 and the 100 ms lead-in holds
        # 12-14, none a keyframe, so 15-19 are skipped and the topic warns once.
        reset_log_once("mcap_cold_channel:/a")
        datasource = MCAPDatasourceV2([path], topics=["/a"], video=VideoOptions())
        part, _ = manifest_for_frames(datasource, path, channels, 0, owned)
        rows, _ = read_rows(datasource, [part])
        assert [r["sequence"] for r in rows] == list(range(20, 30))
        warnings = cold_warnings()
        assert len(warnings) == 1
        assert "'/a'" in warnings[0] and "0.1 s" in warnings[0]
        assert "RAY_DATA_MCAP_MAX_LEAD_IN_S" in warnings[0]

        # Window rows, same split: the window at 0.66 s is ours, its lead-in
        # (frames 17-19) holds no keyframe, so the same warning once.
        caplog.clear()
        reset_log_once("mcap_cold_channel:/a")
        windows = MCAPDatasourceV2(
            [path],
            topics=["/a"],
            read_granularity="window",
            window=WindowSpec(length_s=0.33),
            video=VideoOptions(),
        )
        part, _ = manifest_for_frames(windows, path, channels, 0, owned)
        rows, _ = read_rows(windows, [part])
        assert [row["window_start"] for row in rows] == [660_000_000]
        assert len(rows[0]["frame_times:/a"]) == 10
        assert len(cold_warnings()) == 1

        # The recording's own start: a stream that begins on a P-frame has
        # nothing before it to look back into, so no warning.
        caplog.clear()
        reset_log_once("mcap_cold_channel:/camera")
        head = os.path.join(tmp_path, "midgop_start.mcap")
        write_payloads(
            head, encode_h264(30)[5:], schema_name="foxglove.CompressedVideo"
        )
        from_start = MCAPDatasourceV2([head], video=VideoOptions())
        rows, _ = read_rows(from_start, list_manifests(from_start))
        assert [r["sequence"] for r in rows] == list(range(5, 25))
        assert cold_warnings() == []
    finally:
        mcap_reader.logger.removeHandler(caplog.handler)


def test_empty_window_frames_keep_the_frame_shape(tmp_path):
    """A window with messages but no kept frame (here: ``fps`` keeps one frame
    per second) still gets a ``(0, height, width, 3)`` tensor, sized from a
    frame seen earlier in the task or from ``resize``."""
    path = os.path.join(tmp_path, "sparse.mcap")
    write_two_cameras_and_an_imu(path)
    for video, shape in (
        (VideoOptions(fps=1), (48, 64)),
        (VideoOptions(fps=1, resize=(24, 32)), (24, 32)),
    ):
        datasource = MCAPDatasourceV2(
            [path],
            topics=["/cam_a", "/imu"],
            read_granularity="window",
            window=WindowSpec(length_s=0.33),
            video=video,
        )
        rows, _ = read_rows(datasource, list_manifests(datasource))
        rows = by_window(rows)
        assert sorted(rows) == [0, 330_000_000, 660_000_000]
        height, width = shape
        assert rows[0]["frames:/cam_a"].shape == (1, height, width, 3)
        for start in (330_000_000, 660_000_000):
            assert rows[start]["frames:/cam_a"].shape == (0, height, width, 3)
            assert rows[start]["frame_times:/cam_a"] == []
            assert rows[start]["num_messages"] > 0


def test_decoded_windows_are_counted_without_their_frame_columns(h264_file):
    """A projection that drops both frame columns of a video-only read (a
    ``count()``, or the window bounds alone) still yields every window that
    holds a frame, as the message path keeps its rows without ``frame``."""
    datasource = window_datasource(h264_file, length_s=0.33)
    (manifest,) = list_manifests(datasource)
    for columns in (["window_start"], ["path", "window_start", "window_end"]):
        rows, _ = read_rows(datasource, split_manifest(manifest, 3), columns=columns)
        assert sorted(row["window_start"] for row in rows) == [
            0,
            330_000_000,
            660_000_000,
        ]
        assert all(set(row) == set(columns) for row in rows)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
