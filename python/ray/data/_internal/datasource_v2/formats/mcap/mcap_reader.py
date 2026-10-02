"""Reads the chunks a listing row assigned to a task and builds rows.

``MCAPReader.read`` receives a ``FileManifest`` whose rows name files and, per
file, the byte offsets of the chunks this task owns. For each file it seeks to
those chunks, decompresses them, keeps the selected messages and yields Arrow
tables of about ``target_block_size`` bytes. With ``log_time_order`` the owned
chunks are merged by log time as they are read (a heap of chunk indexes and
messages, as the mcap library does for a whole file), so memory holds only the
chunks that overlap in time.

At ``message`` granularity a row carries the same columns the legacy datasource
produces, plus ``channel_metadata`` (a ``map<string, string>`` of the channel's
metadata) and, on request, ``row_id``: a deterministic name for the message
built from the file path, the chunk's byte offset and the message's position in
the chunk.

At ``window``, ``topic`` and ``file`` granularity a row packs many messages
into parallel list columns (see ``mcap_windows``). A window belongs to the
task owning the chunk its start falls in; that task reads on to the window's
end, and back to the last keyframe of every video topic, so the row is
decodable on its own. Video topics are recognised from the schema name or the
first payload (``mcap_video``); one whose codec cannot be parsed gets the
whole look-back cap as its lead-in. With ``video`` a window row carries the
video topics decoded instead, as ``frames:<topic>`` and ``frame_times:<topic>``
columns, each topic's stream decoded once per task and sliced per window.
"""

import bisect
import heapq
import itertools
import json
import logging
from dataclasses import dataclass, field as dataclasses_field
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import pyarrow as pa
from pyarrow.fs import FileSystem, LocalFileSystem
from typing_extensions import override

from ray.data._internal.arrow_block import _BATCH_SIZE_PRESERVING_STUB_COL_NAME
from ray.data._internal.datasource_v2.formats.mcap.mcap_decode import (
    FrameDecoder,
    FrameThinner,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_options import (
    ATTACHMENT_GRANULARITY,
    DEFAULT_MAX_LEAD_IN_NS,
    MESSAGE_GRANULARITY,
    METADATA_GRANULARITY,
    ROW_ID_COLUMN,
    TOPIC_GRANULARITY,
    WINDOW_GRANULARITY,
    MCAPSelection,
    VideoOptions,
    WindowSpec,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_records import (
    RecordRowBatch,
    iter_records,
    read_record_at,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_summary import (
    attachment_unit_id,
    message_row_id,
    metadata_unit_id,
    topic_unit_id,
    unindexed_message_row_id,
    unindexed_record_row_id,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_video import (
    VideoCodec,
    VideoTopics,
    carries_picture,
    detect_codec,
    is_keyframe,
)
from ray.data._internal.datasource_v2.formats.mcap.mcap_windows import (
    FRAMES_PREFIX,
    CoarseRow,
    CoarseRowBatch,
    owner_offsets,
    place_windows,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data._internal.datasource_v2.interfaces.read_units import ReadUnit
from ray.data._internal.datasource_v2.interfaces.reader import Reader
from ray.data._internal.datasource_v2.interfaces.supports_metadata import (
    MetadataType,
    SupportsMetadata,
)
from ray.data._internal.datasource_v2.interfaces.synthesized_columns import (
    ReadUnitPosition,
    SynthesizedColumn,
)
from ray.data._internal.object_extensions.arrow import raise_on_pickle_object_columns
from ray.data._internal.tensor_extensions.arrow import convert_to_pyarrow_array
from ray.data._internal.util import GiB, iterate_with_retry
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning, PathPartitionParser
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once

if TYPE_CHECKING:
    from mcap.records import Channel, ChunkIndex, Message, Schema
    from mcap.summary import Summary

logger = logging.getLogger(__name__)

# Rough in-memory cost of one row beyond its payload: the timestamps, the
# sequence number, the topic string and the Arrow offsets around them. Only
# used to decide when a table is big enough to yield.
_ROW_OVERHEAD_BYTES = 96

# Largest payload a topic or file row may carry: a memory guardrail, not a
# format limit. A row is built in the worker, converted to Arrow and copied
# into the object store as one object, so it should fit a worker about three
# times over. The payload column uses 64-bit offsets, so the cap may be
# raised past 2 GiB where the workers have the memory.
DEFAULT_MAX_ROW_BYTES = GiB

# Columns a message row carries, in output order. Metadata columns are present
# only with ``include_metadata``; ``row_id`` only with ``include_row_id``.
DATA_COLUMNS = ("data", "topic", "log_time", "publish_time", "sequence")
METADATA_COLUMNS = (
    "channel_id",
    "message_encoding",
    "schema_name",
    "schema_encoding",
    "schema_data",
    "channel_metadata",
)

# One selected message as the reader sees it: its schema (``None`` for a
# schema-less channel), its channel, the record, and its ``row_id``.
_Selected = Tuple[Optional["Schema"], "Channel", "Message", str]
# A message inside a coarse row: schema, channel, record.
_Entry = Tuple[Optional["Schema"], "Channel", "Message"]


@dataclass
class _Declared:
    """Channel and schema records known while reading one file.

    Starts as the summary's and grows with the records met inside chunks, so a
    channel declared only in an earlier chunk serves the later ones.
    ``scanned`` holds the offsets of chunks already read back for declarations.
    """

    channels: Dict[int, "Channel"]
    schemas: Dict[int, "Schema"]
    scanned: Set[int]


def message_schema(
    *,
    include_metadata: bool,
    include_row_id: bool,
    data_type: Optional[pa.DataType] = None,
    frame_type: Optional[pa.DataType] = None,
) -> pa.Schema:
    """The Arrow schema of message rows, before partition and synthesized columns.

    Types follow what the legacy datasource's block builder infers from Python
    values (``int64`` for every integer), so a dataset read through either path
    concatenates. ``schema_data`` is dictionary-encoded: a schema definition is
    stored once per file but repeated on every row, and ROS 2 definitions run
    kilobytes each. ``channel_metadata`` is a ``map`` rather than an inferred
    ``struct`` so its type does not depend on which keys a recorder wrote.

    Args:
        include_metadata: Whether the per-channel and per-schema columns are present.
        include_row_id: Whether ``row_id`` is present.
        data_type: Type of the ``data`` column: ``binary``, or the type of the
            decoded JSON values when every selected channel is JSON-encoded (the
            caller sampled one message for it).
        frame_type: With ``VideoOptions``, the tensor type of the ``frame``
            column that replaces ``data``.

    Returns:
        The schema, columns in output order.
    """
    fields = [
        pa.field("frame", frame_type)
        if frame_type is not None
        else pa.field("data", data_type if data_type is not None else pa.binary()),
        pa.field("topic", pa.string()),
        pa.field("log_time", pa.int64()),
        pa.field("publish_time", pa.int64()),
        pa.field("sequence", pa.int64()),
    ]
    if include_metadata:
        fields += [
            pa.field("channel_id", pa.int64()),
            pa.field("message_encoding", pa.string()),
            pa.field("schema_name", pa.string()),
            pa.field("schema_encoding", pa.string()),
            pa.field("schema_data", pa.dictionary(pa.int32(), pa.binary())),
            pa.field("channel_metadata", pa.map_(pa.string(), pa.string())),
        ]
    if include_row_id:
        fields.append(pa.field(ROW_ID_COLUMN, pa.string()))
    return pa.schema(fields)


def decode_payload(channel: "Channel", data: bytes, where: str) -> Any:
    """Decode a JSON payload into Python values.

    Only called when the dataset's ``data`` column was planned as decoded JSON,
    which requires every selected channel of the sampled files to be
    JSON-encoded. Arrow has no column type for a mix of bytes and decoded
    values (the fallback is a pickled-object column, which the V2 read path
    refuses), so a channel of another encoding, or a message that is not valid
    JSON, fails the read naming the message instead of poisoning the column.

    Args:
        channel: The message's channel.
        data: The payload.
        where: What to name in the error: a ``row_id`` or a file.

    Returns:
        The decoded JSON value.
    """
    if channel.message_encoding != "json":
        raise ValueError(
            f"{where}: topic {channel.topic!r} is {channel.message_encoding!r}-encoded, "
            "but the dataset's `data` column holds decoded JSON because every "
            "selected channel of the sampled files was JSON-encoded. Select topics "
            "of one encoding per read (pass `topics=` or `message_types=`)."
        )
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"{where}: message on JSON-encoded topic {channel.topic!r} is not valid "
            f"JSON: {e}"
        ) from e


class _MessageTableBuilder:
    """Accumulates selected messages column by column and builds Arrow tables."""

    def __init__(
        self,
        *,
        columns: Optional[Set[str]],
        include_metadata: bool,
        include_row_id: bool,
        decode_json: bool,
        decoded: bool = False,
    ):
        # ``None`` means every column. The set decides what is accumulated, so a
        # pruned read never decodes a JSON payload it will not return.
        self._want = (
            (lambda name: True) if columns is None else (lambda name: name in columns)
        )
        self._include_metadata = include_metadata
        self._include_row_id = include_row_id
        # Fixed by the planned schema: ``data`` is decoded JSON values or bytes
        # for every row of the dataset, never a mix.
        self._decode_json = decode_json
        # With ``VideoOptions`` a row is a decoded frame: ``frame``
        # replaces ``data``.
        self._decoded = decoded
        self.reset()

    def reset(self) -> None:
        self.num_rows = 0
        self.estimated_bytes = 0
        self._columns: Dict[str, List[Any]] = {}

    def add(self, selected: _Selected, frame: Any = None) -> None:
        schema, channel, message, row_id = selected
        self.num_rows += 1
        self.estimated_bytes += _ROW_OVERHEAD_BYTES
        put = self._columns.setdefault
        if self._decoded:
            if self._want("frame"):
                # Every decoded row carries its frame: a caller that has none
                # (a projection without ``frame``) must build without the
                # column, or the table's columns would come out ragged.
                assert frame is not None, "a decoded row without its frame"
                self.estimated_bytes += frame.nbytes
                put("frame", []).append(frame)
        else:
            self.estimated_bytes += len(message.data)
            if self._want("data"):
                put("data", []).append(
                    decode_payload(channel, message.data, row_id)
                    if self._decode_json
                    else message.data
                )
        if self._want("topic"):
            put("topic", []).append(channel.topic)
        if self._want("log_time"):
            put("log_time", []).append(message.log_time)
        if self._want("publish_time"):
            put("publish_time", []).append(message.publish_time)
        if self._want("sequence"):
            put("sequence", []).append(message.sequence)
        if self._include_metadata:
            if self._want("channel_id"):
                put("channel_id", []).append(message.channel_id)
            if self._want("message_encoding"):
                put("message_encoding", []).append(channel.message_encoding)
            if self._want("schema_name"):
                put("schema_name", []).append(schema.name if schema else None)
            if self._want("schema_encoding"):
                put("schema_encoding", []).append(schema.encoding if schema else None)
            if self._want("schema_data"):
                put("schema_data", []).append(schema.data if schema else None)
            if self._want("channel_metadata"):
                put("channel_metadata", []).append(list(channel.metadata.items()))
        if self._include_row_id and self._want(ROW_ID_COLUMN):
            put(ROW_ID_COLUMN, []).append(row_id)

    def build(self) -> pa.Table:
        n = self.num_rows
        cols = self._columns
        arrays: Dict[str, pa.Array] = {}
        if "frame" in cols:
            # Same-shaped frames become one fixed-shape tensor column; a block
            # mixing resolutions falls back to the variable-shaped tensor type.
            arrays["frame"] = convert_to_pyarrow_array(cols["frame"], "frame")
        if "data" in cols:
            if self._decode_json:
                # JSON payloads decoded to Python values: let Ray's converter
                # infer a struct, as the legacy block builder does. Blocks whose
                # structs differ (a key missing from some messages) are unified
                # downstream with nulls.
                arrays["data"] = convert_to_pyarrow_array(cols["data"], "data")
            else:
                arrays["data"] = pa.array(cols["data"], type=pa.binary())
        for name, type_ in (
            ("topic", pa.string()),
            ("log_time", pa.int64()),
            ("publish_time", pa.int64()),
            ("sequence", pa.int64()),
            ("channel_id", pa.int64()),
            ("message_encoding", pa.string()),
            ("schema_name", pa.string()),
            ("schema_encoding", pa.string()),
        ):
            if name in cols:
                arrays[name] = pa.array(cols[name], type=type_)
        if "schema_data" in cols:
            arrays["schema_data"] = pa.array(
                cols["schema_data"], type=pa.binary()
            ).dictionary_encode()
        if "channel_metadata" in cols:
            arrays["channel_metadata"] = pa.array(
                cols["channel_metadata"], type=pa.map_(pa.string(), pa.string())
            )
        if ROW_ID_COLUMN in cols:
            arrays[ROW_ID_COLUMN] = pa.array(cols[ROW_ID_COLUMN], type=pa.string())
        if not arrays:
            # Every column was pruned (``count()`` projects to nothing): keep
            # the row count through a stub column, as ``FileReader`` does.
            return pa.table({_BATCH_SIZE_PRESERVING_STUB_COL_NAME: pa.nulls(n)})
        return pa.table(arrays)


def _take_pending(
    waiting: Dict[int, List[_Selected]], log_time: Optional[int]
) -> Optional[_Selected]:
    """Pop the oldest message still owed a frame at ``log_time``, if any."""
    if log_time is None:
        return None
    queue = waiting.get(log_time)
    if not queue:
        return None
    item = queue.pop(0)
    if not queue:
        del waiting[log_time]
    return item


@dataclass
class _PendingWindow:
    """An owned window being filled while its task's entries stream by."""

    start: int
    end: int
    # The non-video messages inside it, in log-time order.
    messages: List[_Entry] = dataclasses_field(default_factory=list)
    # Per decoded topic: the kept frames' log times and the frames themselves.
    frames: Dict[str, Tuple[List[int], List[Any]]] = dataclasses_field(
        default_factory=dict
    )
    decoded_bytes: int = 0


def _warn_cold_channel(topic: str, path: str, cap_ns: int, before: int) -> None:
    """Say once per topic that a channel starts cold, so skipped frames are not silent.

    Not said for a channel whose stream simply begins here (nothing precedes
    it within the cap): that is the recording's own start, not a split.
    """
    if log_once(f"mcap_cold_channel:{topic}"):
        logger.warning(
            "Video topic %r in %r has no keyframe within the %.3g s look-back cap "
            "before log_time %d: its frames are skipped until the next keyframe. "
            "Set RAY_DATA_MCAP_MAX_LEAD_IN_S to look further back.",
            topic,
            path,
            cap_ns / 1e9,
            before,
        )


@dataclass
class _WindowChannel:
    """One decoded video channel of a window task.

    Mirrors the message path's rules: a channel is cold until a keyframe is in
    reach, parameter sets are fed but never counted, and with ``frames``
    pruned (``decoder`` is ``None``) only the frame times a decoder would keep
    are produced, through the shared thinning.
    """

    topic: str
    codec: VideoCodec
    thinner: FrameThinner
    decoder: Optional[FrameDecoder] = None
    # Entries at or after the anchor still to feed; the stream is flushed when
    # it reaches zero.
    remaining: int = 0
    decodable: bool = False
    # Log time of the last fed picture, and of the last frame released.
    last_picture: Optional[int] = None
    released: Optional[int] = None
    done: bool = False
    # For the cold-channel warning: the file and the look-back cap.
    path: str = ""
    cap_ns: int = 0

    def prime(self, lead: List[_Entry], before: Optional[int] = None) -> None:
        """Feed the lead-in (the span before the anchor) so decoding and thinning
        start as a whole-file read's would; its frames are dropped."""
        keyframe_at = None
        for index in range(len(lead) - 1, -1, -1):
            if is_keyframe(lead[index][2].data, self.codec):
                keyframe_at = index
                break
        self.decodable = keyframe_at is not None or self.codec.every_frame_is_a_keyframe
        if not self.decodable and any(
            carries_picture(m.data, self.codec) for _, _, m in lead
        ):
            _warn_cold_channel(
                self.topic,
                self.path,
                self.cap_ns,
                lead[-1][2].log_time if before is None else before,
            )
        if self.decoder is not None and not self.codec.every_frame_is_a_keyframe:
            # From the keyframe on, plus parameter sets written on their own,
            # so the keyframe has them.
            for index, (_, _, message) in enumerate(lead):
                if (keyframe_at is not None and index >= keyframe_at) or (
                    not carries_picture(message.data, self.codec)
                ):
                    for _ in self.decoder.decode(message):
                        pass
        for _, _, message in lead:
            if carries_picture(message.data, self.codec):
                self.thinner.observe(message.log_time)
        if self.remaining == 0:
            self.finish()

    def feed(self, message: "Message") -> List[Tuple[int, Any]]:
        """Feed one entry at or after the anchor; the frames it releases."""
        self.remaining -= 1
        has_picture = carries_picture(message.data, self.codec)
        if has_picture and not self.decodable:
            if not is_keyframe(message.data, self.codec):
                # Cold: no reference to decode against until the next keyframe,
                # but a frame a whole-file read would keep still holds its
                # ``fps`` interval.
                self.thinner.observe(message.log_time)
                return []
            self.decodable = True
        if has_picture:
            self.last_picture = message.log_time
        released: List[Tuple[int, Any]] = []
        if self.decoder is not None:
            released.extend(self.decoder.decode(message))
        elif has_picture and self.thinner.keep(message.log_time):
            released.append((message.log_time, None))
        if released:
            self.released = released[-1][0]
        return released

    def finish(self) -> List[Tuple[int, Any]]:
        """Drain the decoder at the end of the stream."""
        released: List[Tuple[int, Any]] = []
        if self.decoder is not None:
            for log_time, frame in self.decoder.flush():
                # An unstamped frame belongs to the last picture fed.
                stamp = log_time if log_time is not None else self.last_picture
                if stamp is not None:
                    released.append((stamp, frame))
        self.done = True
        return released

    def past(self, end: int) -> bool:
        """Whether every frame before ``end`` is out.

        Frames leave a B-frame-free decoder in log-time order, so a released
        frame at or past ``end`` means none before it is still held; without a
        decoder (frame times only) the last fed picture decides.
        """
        if self.done:
            return True
        if self.decoder is None:
            return self.last_picture is not None and self.last_picture >= end
        return self.released is not None and self.released >= end


@dataclass(frozen=True)
class _Assignment:
    """What one task reads of one file: its chunks, and at topic granularity
    the topic. ``offsets`` is ``None`` for a whole-file listing row."""

    path: str
    offsets: Optional[Set[int]]
    topic: Optional[str] = None

    @property
    def unit(self) -> ReadUnit:
        if self.topic is not None:
            return ReadUnit(id=topic_unit_id(self.path, self.topic), source=self.path)
        return ReadUnit(id=self.path, source=self.path, count=1)


@dataclass
class _ChannelMessages:
    """One video channel's messages in a read range, for lead-in lookups."""

    channel: "Channel"
    schema: Optional["Schema"]
    times: List[int]
    entries: List[_Entry]
    codec: Optional[VideoCodec]
    # Cached ``is_keyframe`` per entry; windows overlap, so each is asked often.
    keyframe: List[Optional[bool]]

    def is_keyframe_at(self, index: int) -> bool:
        cached = self.keyframe[index]
        if cached is None:
            assert self.codec is not None
            cached = is_keyframe(self.entries[index][2].data, self.codec)
            self.keyframe[index] = cached
        return cached


@DeveloperAPI
class MCAPReader(Reader[FileManifest], SupportsMetadata):
    """Reads the chunks of MCAP files a manifest assigns to one task.

    Created by ``MCAPScanner.create_reader`` with every pushdown applied:
    the message selection, the projected columns and the per-task row limit.
    Also answers ``count()`` from the summaries (:meth:`read_metadata`) when
    the selection can be counted there.
    """

    # Files whose summaries one count task reads. A summary read is two small
    # ranged requests, so several per task amortize the task overhead.
    _COUNT_ROWS_BATCH_SIZE = 16

    def __init__(
        self,
        *,
        selection: MCAPSelection,
        granularity: str = MESSAGE_GRANULARITY,
        window: Optional[WindowSpec] = None,
        video: Optional[VideoOptions] = None,
        video_topics: Optional[VideoTopics] = None,
        include_metadata: bool = True,
        include_row_id: bool = False,
        log_time_order: bool = True,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        filesystem: Optional[FileSystem] = None,
        partitioning: Optional[Partitioning] = None,
        synthesized_columns: Sequence[SynthesizedColumn] = (),
        target_block_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        decode_json: bool = False,
        max_row_bytes: int = DEFAULT_MAX_ROW_BYTES,
        max_lead_in_ns: int = DEFAULT_MAX_LEAD_IN_NS,
    ):
        """Initialize the reader.

        Args:
            selection: Which messages to keep.
            granularity: What one row is: ``message``, ``window``, ``topic`` or
                ``file``.
            window: Window placement, required at ``window`` granularity.
            video: Decode the video topics in the task: at ``message``
                granularity one frame per row, at ``window`` granularity the
                window's frames per topic, thinned to ``fps`` and scaled to
                ``resize``.
            video_topics: Which topics planning found to carry video; a topic
                it never saw is sniffed from its first payload.
            include_metadata: Whether to emit the channel and schema columns.
            include_row_id: Whether to emit ``row_id``.
            log_time_order: Whether a task's message rows come out in ascending
                ``log_time`` order (its chunks merged by time) or in file order.
                Coarse rows are always in log-time order.
            columns: Columns to produce, in order; ``None`` for all of them.
            limit: Stop after this many rows per manifest.
            filesystem: Filesystem the paths resolve against; local when ``None``.
            partitioning: Path partitioning whose values become string columns.
            synthesized_columns: Columns appended to every table rather than
                read, e.g. ``PathColumn`` for ``include_paths``.
            target_block_size: Estimated bytes per yielded table; ``None`` yields
                one table per file.
            schema: Dataset schema, used to type partition columns.
            decode_json: Whether ``data`` holds decoded JSON values (planned so
                because every selected channel of the sample was JSON-encoded)
                rather than the payload bytes.
            max_row_bytes: Largest payload a topic or file row may carry before
                the read fails rather than build it.
            max_lead_in_ns: How far before a window a video topic's lead-in may
                reach (``RAY_DATA_MCAP_MAX_LEAD_IN_S``).
        """
        if granularity == WINDOW_GRANULARITY and window is None:
            raise ValueError("window granularity needs a WindowSpec")
        self._selection = selection
        self._decode_json = decode_json
        self._granularity = granularity
        self._window = window
        self._video = video
        self._video_topics = video_topics if video_topics is not None else VideoTopics()
        self._include_metadata = include_metadata
        self._include_row_id = include_row_id
        self._log_time_order = log_time_order
        self._columns = list(columns) if columns is not None else None
        self._limit = limit
        self._filesystem = filesystem
        self._partition_parser = (
            PathPartitionParser(partitioning) if partitioning is not None else None
        )
        self._synthesized_columns = tuple(synthesized_columns)
        self._target_block_size = target_block_size
        self._schema = schema
        self._max_row_bytes = max_row_bytes
        self._max_lead_in_ns = max_lead_in_ns

    def read(self, input_split: FileManifest) -> Iterator[pa.Table]:
        """Read the files and chunks named by ``input_split``.

        Rows of one file are read together however many manifest rows name it;
        files come out in manifest order.
        """
        from ray.data.context import DataContext

        if len(input_split) == 0:
            return
        retried_io_errors = DataContext.get_current().retried_io_errors
        remaining = self._limit
        for assignment in _assignments(input_split):
            tables = iterate_with_retry(
                partial(self._read_file, assignment),
                f"read MCAP file {assignment.path}",
                match=retried_io_errors,
            )
            for table in tables:
                if remaining is not None:
                    if remaining <= 0:
                        return
                    if table.num_rows > remaining:
                        table = table.slice(0, remaining)
                    remaining -= table.num_rows
                yield table

    # -- metadata ----------------------------------------------------------

    @override
    def read_metadata(self, file_manifest: FileManifest) -> Iterator[BlockMetadata]:
        """Yield one ``BlockMetadata`` per file with its selected message count.

        ``Statistics`` holds the count per channel, so a selection by topic or
        schema is summed from it without reading a payload. A file with no
        statistics is counted by scanning it, which is still exact.
        """
        from mcap.reader import SeekingReader
        from mcap.records import Attachment, Metadata

        filesystem = self._filesystem or LocalFileSystem()
        for path in dict.fromkeys(str(p) for p in file_manifest.paths):
            with filesystem.open_input_file(path) as f:
                summary = SeekingReader(f).get_summary()
                statistics = summary.statistics if summary is not None else None
                if self._granularity == ATTACHMENT_GRANULARITY:
                    if statistics is not None:
                        num_rows = statistics.attachment_count
                    elif summary is not None and summary.attachment_indexes:
                        num_rows = len(summary.attachment_indexes)
                    else:
                        num_rows = sum(1 for _ in iter_records(f, Attachment))
                elif self._granularity == METADATA_GRANULARITY:
                    if statistics is not None:
                        num_rows = statistics.metadata_count
                    elif summary is not None and summary.metadata_indexes:
                        num_rows = len(summary.metadata_indexes)
                    else:
                        num_rows = sum(1 for _ in iter_records(f, Metadata))
                elif (
                    summary is not None
                    and statistics is not None
                    and set(statistics.channel_message_counts) <= set(summary.channels)
                    and self._schemas_known_for_selection(summary)
                ):
                    selected = self._selection.selected_channel_ids(
                        summary.channels, summary.schemas
                    )
                    num_rows = sum(
                        statistics.channel_message_counts.get(cid, 0)
                        for cid in selected
                    )
                else:
                    # No statistics, a channel declared only inside a chunk (no
                    # topic or schema in the summary to apply the selection to),
                    # or a schema ``message_types`` needs that only a chunk
                    # declares: count by reading, which the message reader does
                    # too.
                    num_rows = sum(1 for _ in self._iter_unindexed(f, path))
            yield BlockMetadata(
                num_rows=num_rows,
                size_bytes=None,
                exec_stats=None,
                input_files=(path,),
            )

    def _schemas_known_for_selection(self, summary: "Summary") -> bool:
        """Whether the summary carries every schema ``message_types`` needs.

        A channel whose schema record lives only inside a chunk passes the
        listing's schema filter unchecked; the reader filters its messages once
        the chunk declares the schema, so a count summed from the statistics
        would include them. Without ``message_types`` the schemas do not matter.
        """
        if self._selection.message_types is None:
            return True
        return all(
            not channel.schema_id or channel.schema_id in summary.schemas
            for channel in summary.channels.values()
        )

    @override
    def available_metadata(self) -> Set[MetadataType]:
        # A time range cannot be counted from statistics; a coarse row is not a
        # message, so nothing counts it. Metadata records carry no time. A
        # decoded read emits frames, which ``fps`` thins and a decoder may drop.
        if self._granularity == METADATA_GRANULARITY:
            return {MetadataType.NUM_ROWS}
        if (
            self._granularity not in (MESSAGE_GRANULARITY, ATTACHMENT_GRANULARITY)
            or self._selection.time_range is not None
            or self._video is not None
        ):
            return set()
        return {MetadataType.NUM_ROWS}

    @override
    def get_target_metadata_batch_size(self) -> Optional[int]:
        return self._COUNT_ROWS_BATCH_SIZE

    # -- one file ----------------------------------------------------------

    def _read_file(self, assignment: _Assignment) -> Iterator[pa.Table]:
        """Yield the tables of one file, limited to the assigned chunks."""
        from mcap.reader import SeekingReader

        filesystem = self._filesystem or LocalFileSystem()
        with filesystem.open_input_file(assignment.path) as f:
            if self._granularity in (ATTACHMENT_GRANULARITY, METADATA_GRANULARITY):
                yield from self._record_tables(f, assignment)
                return
            summary = SeekingReader(f).get_summary()
            if summary is not None and not summary.chunk_indexes:
                summary = None
            if (
                summary is not None
                and not summary.channels
                and self._granularity != MESSAGE_GRANULARITY
            ):
                # The indexer listed this file whole: a summary that repeats
                # no channel record says nothing about which chunk holds what,
                # so a coarse row is built by scanning the file, as without an
                # index. Message rows keep the chunk walk, which filters the
                # channels as their records appear.
                summary = None
            if self._granularity == MESSAGE_GRANULARITY:
                if self._video is not None:
                    yield from self._decoded_tables(f, assignment, summary)
                    return
                if summary is None:
                    messages = self._iter_unindexed(f, assignment.path)
                else:
                    messages = self._iter_chunks(
                        f, assignment.path, summary, assignment.offsets
                    )
                yield from self._tables(messages, assignment)
            elif self._granularity == WINDOW_GRANULARITY:
                yield from self._window_tables(f, assignment, summary)
            elif self._granularity == TOPIC_GRANULARITY:
                yield from self._topic_tables(f, assignment, summary)
            else:
                yield from self._file_tables(f, assignment, summary)

    def _candidate_chunks(
        self, summary: "Summary", selected: Set[int]
    ) -> List["ChunkIndex"]:
        """The file's chunks that may hold a selected message, in file order.

        The indexer listed exactly these, so a window's owner is computed over
        the same chunks on both sides.
        """
        # A summary that repeats no channel record says nothing about which
        # chunk holds a selected message: every chunk in the time range is read
        # and the channels declared inside them are filtered as they appear.
        channels_known = bool(summary.channels)
        return sorted(
            (
                c
                for c in summary.chunk_indexes
                if (
                    self._selection.chunk_may_match(c, selected)
                    if channels_known
                    else self._selection.overlaps(
                        c.message_start_time, c.message_end_time
                    )
                )
            ),
            key=lambda c: c.chunk_start_offset,
        )

    def _iter_chunks(
        self,
        f: Any,
        path: str,
        summary: "Summary",
        offsets: Optional[Set[int]],
        *,
        chunk_indexes: Optional[List["ChunkIndex"]] = None,
        selected: Optional[Set[int]] = None,
        time_bounds: Optional[Tuple[Optional[int], Optional[int]]] = None,
        log_time_order: Optional[bool] = None,
        with_offsets: bool = False,
    ) -> Iterator[Any]:
        """Yield the selected messages of the owned chunks of an indexed file.

        With ``with_offsets`` every item is ``(chunk offset, message)``, for a
        consumer that must notice when a channel's stream jumps between chunks.

        With log-time order the chunks are merged through one heap holding
        chunk indexes (keyed by their first log time) and messages (keyed by
        theirs): a chunk is expanded when it reaches the top, so at most the
        chunks that overlap in time are decompressed at once, and a message is
        yielded only once every chunk that could precede it has been expanded.
        Without it the chunks are read in file order.

        ``time_bounds`` overrides the selection's time range; coarse rows use
        it to read a window's lead-in, which lies before the range.
        """
        if selected is None:
            selected = self._selection.selected_channel_ids(
                summary.channels, summary.schemas
            )
        if chunk_indexes is None:
            chunk_indexes = self._candidate_chunks(summary, selected)
        if offsets is not None:
            chunk_indexes = [
                c for c in chunk_indexes if c.chunk_start_offset in offsets
            ]
        if time_bounds is None:
            time_bounds = (self._selection.start_time, self._selection.end_time)
        if log_time_order is None:
            log_time_order = self._log_time_order
        declared = _Declared(dict(summary.channels), dict(summary.schemas), set())
        if not log_time_order:
            for chunk_index in chunk_indexes:
                for item in self._read_chunk(
                    f, path, summary, chunk_index, selected, declared, time_bounds
                ):
                    yield (
                        chunk_index.chunk_start_offset,
                        item,
                    ) if with_offsets else item
            return

        # Heap entries: (log time, kind, chunk offset, index in chunk, item).
        # ``kind`` 0 is a chunk index, 1 a message, so on a tied log time a
        # chunk is expanded before a message is yielded.
        heap: List[Tuple[int, int, int, int, Any]] = [
            (c.message_start_time, 0, c.chunk_start_offset, 0, c) for c in chunk_indexes
        ]
        heapq.heapify(heap)
        while heap:
            _, kind, offset, _, item = heapq.heappop(heap)
            if kind == 0:
                for index, selected_message in self._read_chunk(
                    f,
                    path,
                    summary,
                    item,
                    selected,
                    declared,
                    time_bounds,
                    with_index=True,
                ):
                    heapq.heappush(
                        heap,
                        (
                            selected_message[2].log_time,
                            1,
                            offset,
                            index,
                            selected_message,
                        ),
                    )
            else:
                yield (offset, item) if with_offsets else item

    def _read_chunk(
        self,
        f: Any,
        path: str,
        summary: "Summary",
        chunk_index: "ChunkIndex",
        selected: Set[int],
        declared: "_Declared",
        time_bounds: Tuple[Optional[int], Optional[int]],
        with_index: bool = False,
    ) -> Iterator[Any]:
        """Decompress one chunk and yield its selected messages in file order.

        A chunk may carry its own ``Schema`` and ``Channel`` records (a writer
        that did not repeat them in the summary); they are honoured over the
        summary's. ``row_id`` counts every message record of the chunk, so it
        does not depend on the selection.
        """
        from mcap.data_stream import ReadDataStream
        from mcap.records import Channel, Chunk, Message, Schema
        from mcap.stream_reader import breakup_chunk

        start_time, end_time = time_bounds
        # Skip the record's opcode (1 byte) and length (8 bytes).
        f.seek(chunk_index.chunk_start_offset + 1 + 8)
        chunk = Chunk.read(ReadDataStream(f))
        # Channel and schema records found in chunks accumulate across the
        # chunks of one file read (``declared``), so a channel a writer declared
        # only in an earlier chunk serves the later ones too.
        channels: Dict[int, "Channel"] = declared.channels
        schemas: Dict[int, "Schema"] = declared.schemas
        index = -1
        for record in breakup_chunk(chunk):
            if isinstance(record, Message):
                index += 1
                channel = channels.get(record.channel_id)
                if channel is None:
                    self._find_in_earlier_chunks(
                        f, summary, chunk_index, declared, channel_id=record.channel_id
                    )
                    channel = channels.get(record.channel_id)
                if channel is None:
                    raise ValueError(
                        f"MCAP file {path!r} has a message on channel "
                        f"{record.channel_id}, which neither the summary nor any "
                        "chunk up to this one declares."
                    )
                if channel.schema_id and channel.schema_id not in schemas:
                    # The channel is known but its schema record was written in
                    # an earlier chunk only (``repeat_schemas=False``): read back
                    # for it, so ``message_types`` and the schema columns hold.
                    self._find_in_earlier_chunks(
                        f, summary, chunk_index, declared, schema_id=channel.schema_id
                    )
                schema = schemas.get(channel.schema_id) if channel.schema_id else None
                if record.channel_id not in selected:
                    if channel.id in summary.channels:
                        continue
                    # A channel declared only inside this chunk was unknown to
                    # the listing; apply the filters to it now.
                    if not self._selection.accepts_channel(channel, schema):
                        continue
                elif (
                    channel.schema_id
                    and channel.schema_id not in summary.schemas
                    and not self._selection.accepts_channel(channel, schema)
                ):
                    # The summary named the channel but not its schema, so the
                    # listing could not apply ``message_types``; the schema
                    # record found inside the chunks settles it now.
                    continue
                if start_time is not None and record.log_time < start_time:
                    continue
                if end_time is not None and record.log_time >= end_time:
                    continue
                row_id = message_row_id(path, chunk_index.chunk_start_offset, index)
                item = (schema, channel, record, row_id)
                yield (index, item) if with_index else item
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Schema):
                schemas[record.id] = record

    def _find_in_earlier_chunks(
        self,
        f: Any,
        summary: "Summary",
        chunk_index: "ChunkIndex",
        declared: "_Declared",
        *,
        channel_id: Optional[int] = None,
        schema_id: Optional[int] = None,
    ) -> None:
        """Read back through the chunks before ``chunk_index`` for a declaration.

        A writer may declare a channel or a schema inside the first chunk that
        uses it and repeat it nowhere else; a task that owns only a later chunk
        then has to read back for the record. Every channel and schema record
        met on the way is kept in ``declared``, and each chunk is read back at
        most once per file, so the walk stops as soon as the wanted record is
        in ``declared`` (or every earlier chunk has been seen).
        """
        from mcap.data_stream import ReadDataStream
        from mcap.records import Channel, Chunk, Schema
        from mcap.stream_reader import breakup_chunk

        earlier = sorted(
            (
                c
                for c in summary.chunk_indexes
                if c.chunk_start_offset < chunk_index.chunk_start_offset
                and c.chunk_start_offset not in declared.scanned
            ),
            key=lambda c: c.chunk_start_offset,
        )
        for earlier_index in earlier:
            declared.scanned.add(earlier_index.chunk_start_offset)
            f.seek(earlier_index.chunk_start_offset + 1 + 8)
            for record in breakup_chunk(Chunk.read(ReadDataStream(f))):
                if isinstance(record, Channel):
                    declared.channels.setdefault(record.id, record)
                elif isinstance(record, Schema):
                    declared.schemas.setdefault(record.id, record)
            if (channel_id is None or channel_id in declared.channels) and (
                schema_id is None or schema_id in declared.schemas
            ):
                break
        # Back to the chunk being read.
        f.seek(chunk_index.chunk_start_offset + 1 + 8)

    def _iter_unindexed(
        self,
        f: Any,
        path: str,
        time_bounds: Optional[Tuple[Optional[int], Optional[int]]] = None,
        *,
        on_message: Optional[Callable[["Message"], None]] = None,
    ) -> Iterator[_Selected]:
        """Scan a file without a chunk index from the start, in file order.

        Ordering by log time would mean holding the whole file, which is what
        the legacy datasource did for every file; a file without an index is
        read as written and ``log_time_order`` is not honoured for its message
        rows. Coarse rows sort what they collect. ``on_message`` sees every
        message record before any filter, selected or not: it stands in for
        the file-wide statistics an indexed file carries.
        """
        from mcap.records import Channel, Message, Schema
        from mcap.stream_reader import StreamReader

        if time_bounds is None:
            time_bounds = (self._selection.start_time, self._selection.end_time)
        start_time, end_time = time_bounds
        f.seek(0)
        schemas: Dict[int, "Schema"] = {}
        channels: Dict[int, "Channel"] = {}
        ordinal = -1
        for record in StreamReader(f).records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Message):
                ordinal += 1
                if on_message is not None:
                    on_message(record)
                channel = channels.get(record.channel_id)
                if channel is None:
                    raise ValueError(
                        f"MCAP file {path!r} has a message on channel "
                        f"{record.channel_id} before that channel is declared."
                    )
                schema = schemas.get(channel.schema_id) if channel.schema_id else None
                if not self._selection.accepts_channel(channel, schema):
                    continue
                if start_time is not None and record.log_time < start_time:
                    continue
                if end_time is not None and record.log_time >= end_time:
                    continue
                yield schema, channel, record, unindexed_message_row_id(path, ordinal)

    # -- message rows ------------------------------------------------------

    def _tables(
        self, messages: Iterator[_Selected], assignment: _Assignment
    ) -> Iterator[pa.Table]:
        """Build tables of about ``target_block_size`` bytes from the messages."""
        wanted = set(self._columns) if self._columns is not None else None
        builder = _MessageTableBuilder(
            columns=wanted,
            include_metadata=self._include_metadata,
            include_row_id=self._include_row_id,
            decode_json=self._decode_json,
        )
        rows_before = 0
        for selected in messages:
            builder.add(selected)
            if (
                self._target_block_size is not None
                and builder.estimated_bytes >= self._target_block_size
            ):
                yield self._finish(builder.build(), assignment, rows_before)
                rows_before += builder.num_rows
                builder.reset()
        if builder.num_rows > 0:
            yield self._finish(builder.build(), assignment, rows_before)

    # -- decoded frame rows --------------------------------------------------

    def _decoded_tables(
        self, f: Any, assignment: _Assignment, summary: Optional["Summary"]
    ) -> Iterator[pa.Table]:
        """One row per decoded frame of the task's messages.

        The task's first message on a channel is rarely a keyframe, so when a
        channel's first owned message comes up its decoder is primed with that
        channel's frames back to the previous keyframe (the "lead-in": its
        messages before that one, within the look-back cap, read from whichever
        chunks hold them), whose output is discarded. Each channel has its own
        cutoff, since a task's channels start at different times, and the same
        happens again wherever a channel's owned chunks are not consecutive
        (a chunk between them belongs to another task, or was excluded by a
        checkpoint): the skipped messages are fed to the decoder before the
        next owned one. The lead-in also seeds the ``fps`` thinning, so the
        frames that survive do not depend on where the task starts. If a
        projection dropped ``frame``, nothing is decoded and a row is a
        message, thinned the same way: a message without picture data
        (parameter sets written on their own) is no row, and a channel that
        starts without a keyframe in reach yields no row until its next
        keyframe, exactly as the decoder would. Only a payload the codec
        rejects outright still counts as a row when ``frame`` is not read.

        Streams are taken to be free of B-frames, as the robotics recorders
        that write MCAP video require (each message decodes to one frame, in
        log-time order), so frames come out of the decoder in log-time order.
        """
        assert self._video is not None
        video = self._video
        path = assignment.path
        wanted = set(self._columns) if self._columns is not None else None
        decode = wanted is None or "frame" in wanted
        start_time, end_time = self._selection.start_time, self._selection.end_time
        lead_ns = self._max_lead_in_ns

        unindexed_lead: Dict[int, List[_Selected]] = {}
        if summary is None:
            # A file without an index is one task and is read front to back; the
            # lead-in is whatever precedes the time range, per channel.
            low = max(0, start_time - lead_ns) if start_time is not None else None
            items = sorted(
                self._iter_unindexed(f, path, time_bounds=(low, end_time)),
                key=lambda item: item[2].log_time,
            )
            if start_time is not None:
                for item in items:
                    if item[2].log_time < start_time:
                        unindexed_lead.setdefault(item[1].id, []).append(item)
                items = [m for m in items if m[2].log_time >= start_time]
            owned: Iterator[Tuple[Optional[int], _Selected]] = (
                (None, item) for item in items
            )
        else:
            owned = self._iter_chunks(
                f,
                path,
                summary,
                assignment.offsets,
                selected=self._selection.selected_channel_ids(
                    summary.channels, summary.schemas
                ),
                log_time_order=True,
                with_offsets=True,
            )

        def channel_messages(
            channel_id: int,
            low: int,
            high: int,
            skip_offsets: Optional[Set[int]] = None,
        ) -> List[_Selected]:
            """The channel's messages with ``low <= log_time < high`` of an indexed file.

            Read from every chunk that may hold the channel in that span except
            ``skip_offsets``: an owned chunk can hold messages before the time
            range too, and a channel's previous keyframe may sit in a chunk the
            listing never considered.
            """
            assert summary is not None
            if low >= high:
                return []
            chunks = sorted(
                (
                    c
                    for c in summary.chunk_indexes
                    if c.message_end_time >= low
                    and c.message_start_time < high
                    and (
                        skip_offsets is None or c.chunk_start_offset not in skip_offsets
                    )
                    and (
                        not c.message_index_offsets
                        or channel_id in c.message_index_offsets
                    )
                ),
                key=lambda c: c.chunk_start_offset,
            )
            if not chunks:
                return []
            return list(
                self._iter_chunks(
                    f,
                    path,
                    summary,
                    None,
                    chunk_indexes=chunks,
                    selected={channel_id},
                    time_bounds=(low, high),
                    log_time_order=True,
                )
            )

        def lead_in_for(channel_id: int, first_time: int) -> List[_Selected]:
            """The channel's messages in the lead-in span before its first owned one."""
            if summary is None:
                return unindexed_lead.get(channel_id, [])
            if not lead_ns:
                return []
            return channel_messages(
                channel_id, max(0, first_time - lead_ns), first_time
            )

        def gap_for(
            channel_id: int, last_time: int, next_time: int
        ) -> Tuple[List[_Selected], bool]:
            """The channel's messages another task owns between two of ours, and
            whether the gap was longer than the look-back cap and so clipped."""
            if (
                summary is None
                or assignment.offsets is None
                or next_time <= last_time + 1
            ):
                return [], False
            low = max(last_time + 1, next_time - lead_ns)
            # Clipped only if another task's chunk may hold the channel in the
            # part of the gap we do not read: a channel that merely pauses, or
            # a jump between consecutive chunks, keeps the decoder's state.
            skipped_before = low > last_time + 1 and any(
                c.chunk_start_offset not in assignment.offsets
                and c.message_end_time > last_time
                and c.message_start_time < low
                and (
                    not c.message_index_offsets or channel_id in c.message_index_offsets
                )
                for c in summary.chunk_indexes
            )
            # With the look-back cap at zero nothing is read back, but a skipped
            # span still leaves the channel cold.
            items = (
                channel_messages(
                    channel_id, low, next_time, skip_offsets=assignment.offsets
                )
                if low < next_time
                else []
            )
            return items, skipped_before

        decoders: Dict[int, FrameDecoder] = {}
        thinners: Dict[int, FrameThinner] = {}
        # Without ``frame``: each channel's codec, and whether a decoder would
        # have a keyframe to work from at this point of the stream.
        codecs: Dict[int, VideoCodec] = {}
        decodable: Dict[int, bool] = {}
        # Per channel: the task's messages whose frames have not come out yet,
        # by log time (a list, since two messages may share one), and the log
        # times of the primed lead-in, whose frames are dropped if the decoder
        # releases them late.
        pending: Dict[int, Dict[int, List[_Selected]]] = {}
        primed: Dict[int, Set[int]] = {}

        def decoder_for(item: _Selected) -> Optional[FrameDecoder]:
            channel_id = item[1].id
            decoder = decoders.get(channel_id)
            if decoder is None:
                codec = codec_for(item)
                if codec is None:
                    return None
                decoder = FrameDecoder(
                    codec, resize=video.resize, thinner=thinner_for(channel_id)
                )
                decoders[channel_id] = decoder
            return decoder

        def thinner_for(channel_id: int) -> FrameThinner:
            thinner = thinners.get(channel_id)
            if thinner is None:
                thinner = thinners[channel_id] = FrameThinner(video.fps_interval_ns)
            return thinner

        def codec_for(
            item: _Selected, context: Sequence[_Selected] = ()
        ) -> Optional[VideoCodec]:
            """The channel's codec, sniffed from its payloads.

            The task's first payload rarely tells for VP9 and AV1 (an inter
            frame's header is too short), so the lead-in is tried too, latest
            first, where the previous keyframe sits. A video topic whose codec
            none of them tells yet stays cold, its messages skipped, until a
            later payload (its next keyframe) identifies it. A topic nothing
            recognises as video fails the read, naming it.
            """
            schema, channel, message, _ = item
            codec = codecs.get(channel.id)
            if codec is not None:
                return codec
            for payload in itertools.chain(
                (message.data,), (m[2].data for m in reversed(context))
            ):
                codec = detect_codec(payload)
                if codec is not None:
                    codecs[channel.id] = codec
                    return codec
            if not self._video_topics.recognises(
                channel.topic, schema.name if schema else None
            ):
                raise ValueError(
                    f"Cannot decode topic {channel.topic!r} in {path!r}: its payload "
                    "is not a recognised video payload (JPEG, PNG, H.264/H.265 "
                    "Annex-B, VP9 or AV1). Pass topics=[...] to select only the "
                    "video topics."
                )
            if log_once(f"mcap_codec_pending:{channel.topic}"):
                logger.warning(
                    "The codec of video topic %r in %r cannot be told from its "
                    "first payloads; its frames are skipped until a keyframe "
                    "identifies it.",
                    channel.topic,
                    path,
                )
            return None

        def emit(
            channel_id: int, log_time: int, frame: Any, fallback: Optional[_Selected]
        ) -> None:
            """Attribute a decoded frame to the message that held it, if it is ours.

            A frame stamped with a pending message's log time is that message's;
            one stamped with a primed (lead-in or gap) time belongs to another
            task's row and is dropped; anything else goes to ``fallback``, the
            message being decoded, or is dropped while priming.
            """
            waiting = pending.setdefault(channel_id, {})
            source = _take_pending(waiting, log_time)
            if source is None:
                if log_time in primed.get(channel_id, ()) or fallback is None:
                    return
                source = fallback
            # Frames come out in log-time order, so a message older than this
            # frame will not be given one: stop holding it.
            for stale in list(itertools.takewhile(lambda t: t < log_time, waiting)):
                del waiting[stale]
            builder.add(source, frame)

        def prime_channel(
            item: _Selected, items: List[_Selected], gap: bool, clipped: bool = False
        ) -> None:
            """Feed a channel's decoder and thinning the messages before ``item``.

            For a lead-in, decoding starts at the last keyframe among them; a
            channel without one is cold, and stays cold until its next keyframe
            (its frames are neither decoded nor rows: without a reference the
            codec would either reject them or conceal, and a concealed frame
            is not the recording). For a gap between two owned chunks every
            skipped message is fed when no keyframe lies among them, so the
            decoder's references stay continuous; a gap longer than
            the look-back cap cannot be fed whole, so without a keyframe in its
            tail the channel goes cold instead. The fed messages' own frames
            are dropped, but a frame of ours the decoder was still holding back
            comes out first and is kept, so the frame just before a gap is not
            lost.
            """
            channel_id = item[1].id
            codec = codec_for(item, items)
            thinner = thinner_for(channel_id)
            if codec is None:
                # Nothing to decode with yet: cold until a payload tells.
                decodable[channel_id] = False
                for m in items:
                    thinner.observe(m[2].log_time)
                return
            keyframe_at = None
            for index in range(len(items) - 1, -1, -1):
                if is_keyframe(items[index][2].data, codec):
                    keyframe_at = index
                    break
            if keyframe_at is not None or codec.every_frame_is_a_keyframe:
                decodable[channel_id] = True
            elif not gap or clipped:
                decodable[channel_id] = False
                if clipped or any(carries_picture(m[2].data, codec) for m in items):
                    _warn_cold_channel(
                        item[1].topic, path, self._max_lead_in_ns, item[2].log_time
                    )
            if decode:
                decoder = decoder_for(item)
                assert decoder is not None
                if keyframe_at is not None:
                    start: Optional[int] = keyframe_at
                elif gap and not clipped:
                    start = 0
                else:
                    start = None
                # Parameter sets written on their own are fed whatever the
                # start, so the keyframe that follows has them.
                fed = [
                    m
                    for index, m in enumerate(items)
                    if (start is not None and index >= start)
                    or not carries_picture(m[2].data, codec)
                ]
                primed.setdefault(channel_id, set()).update(m[2].log_time for m in fed)
                for m in fed:
                    for log_time, frame in decoder.decode(m[2]):
                        emit(channel_id, log_time, frame, None)
            # After the feed: a held-back frame of ours must meet the thinning
            # state it was recorded under, not one the lead-in moved forward.
            # Only messages that are frames hold an interval.
            for m in items:
                if carries_picture(m[2].data, codec):
                    thinner.observe(m[2].log_time)

        builder = _MessageTableBuilder(
            columns=wanted,
            include_metadata=self._include_metadata,
            include_row_id=self._include_row_id,
            decode_json=False,
            decoded=True,
        )
        rows_before = 0

        def flush_rows() -> Iterator[pa.Table]:
            nonlocal rows_before
            if builder.num_rows > 0:
                yield self._finish(builder.build(), assignment, rows_before)
                rows_before += builder.num_rows
                builder.reset()

        # Per channel: the chunk and log time of the last owned message, to
        # notice a jump to a non-consecutive chunk.
        last_seen: Dict[int, Tuple[Optional[int], int]] = {}
        for offset, item in owned:
            channel_id, message = item[1].id, item[2]
            seen = last_seen.get(channel_id)
            if seen is None:
                prime_channel(
                    item, lead_in_for(channel_id, message.log_time), gap=False
                )
            elif offset is not None and offset != seen[0]:
                skipped, clipped = gap_for(channel_id, seen[1], message.log_time)
                if skipped or clipped:
                    prime_channel(item, skipped, gap=True, clipped=clipped)
            last_seen[channel_id] = (offset, message.log_time)
            # Parameter sets written on their own yield no picture: no row,
            # and no claim on the frame of a keyframe stamped alike. A cold
            # channel (no keyframe in reach) yields nothing until its next
            # keyframe; its parameter sets are still fed to the decoder.
            codec = codec_for(item)
            if codec is None:
                # A frame a whole-file read would have kept still holds its
                # ``fps`` interval, so the split read keeps the same frames.
                thinner_for(channel_id).observe(message.log_time)
                continue
            has_picture = carries_picture(message.data, codec)
            if has_picture and not decodable.get(channel_id, False):
                if not is_keyframe(message.data, codec):
                    thinner_for(channel_id).observe(message.log_time)
                    continue
                decodable[channel_id] = True
            if not decode:
                if has_picture and thinner_for(channel_id).keep(message.log_time):
                    builder.add(item)
            else:
                decoder = decoder_for(item)
                assert decoder is not None
                if has_picture:
                    pending.setdefault(channel_id, {}).setdefault(
                        message.log_time, []
                    ).append(item)
                for log_time, frame in decoder.decode(message):
                    emit(channel_id, log_time, frame, item if has_picture else None)
            if (
                self._target_block_size is not None
                and builder.estimated_bytes >= self._target_block_size
            ):
                yield from flush_rows()
        for channel_id, decoder in decoders.items():
            waiting = pending.get(channel_id, {})
            for log_time, frame in decoder.flush():
                source = _take_pending(waiting, log_time)
                if source is None:
                    if log_time in primed.get(channel_id, ()) or not waiting:
                        continue
                    # An unstamped frame belongs to the last message still owed one.
                    source = _take_pending(waiting, max(waiting))
                    assert source is not None
                builder.add(source, frame)
        yield from flush_rows()

    # -- coarse rows -------------------------------------------------------

    def _span(self, summary: "Summary") -> Tuple[int, int]:
        """The file's first and last log time, clipped to the time range.

        File-wide, not the selection's: the window grid of a recording must
        not move when a read selects different topics, so ``file_start``
        means the file's first message whatever is read from it.
        """
        statistics = summary.statistics
        if statistics is not None and statistics.message_count > 0:
            start, end = statistics.message_start_time, statistics.message_end_time
        else:
            start = min(c.message_start_time for c in summary.chunk_indexes)
            end = max(c.message_end_time for c in summary.chunk_indexes)
        return self._clip_span(start, end)

    def _clip_span(self, start: int, end: int) -> Tuple[int, int]:
        if self._selection.start_time is not None:
            start = max(start, self._selection.start_time)
        if self._selection.end_time is not None:
            end = min(end, self._selection.end_time - 1)
        return start, end

    def _entries(
        self, selected: Iterator[_Selected], what: Optional[str] = None
    ) -> List[_Entry]:
        """Collect entries; with ``what`` (a row's name), fail as soon as their
        payloads exceed the row limit rather than after holding them all."""
        entries: List[_Entry] = []
        payload_bytes = 0
        for schema, channel, message, _ in selected:
            if what is not None:
                payload_bytes += len(message.data)
                if payload_bytes > self._max_row_bytes:
                    self._raise_row_too_large(payload_bytes, what, partial=True)
            entries.append((schema, channel, message))
        return entries

    def _window_tables(
        self, f: Any, assignment: _Assignment, summary: Optional["Summary"]
    ) -> Iterator[pa.Table]:
        """Emit the window rows this task owns for one file."""
        assert self._window is not None
        path = assignment.path
        if summary is None:
            # No chunk index: the whole file is one task, so every window is
            # ours. Everything is read; a recorder that wrote no index gives
            # nothing to plan with.
            if log_once(f"mcap_window_unindexed:{path}"):
                logger.warning(
                    "MCAP file %r has no chunk index; reading it whole to place "
                    "windows. Rewrite the file with an index to split it across "
                    "tasks.",
                    path,
                )
            bounds: List[int] = []

            def observe(message: "Message") -> None:
                # The file-wide span, as an indexed file's statistics give it.
                if not bounds:
                    bounds.extend((message.log_time, message.log_time))
                else:
                    bounds[0] = min(bounds[0], message.log_time)
                    bounds[1] = max(bounds[1], message.log_time)

            entries = self._entries(
                self._iter_unindexed(f, path, (None, None), on_message=observe)
            )
            entries.sort(key=lambda e: e[2].log_time)
            if not entries or not any(
                self._selection.in_time_range(e[2].log_time) for e in entries
            ):
                return
            start, end = self._clip_span(bounds[0], bounds[1])
            if start > end:
                return
            windows = place_windows(self._window, start, end)
        else:
            selected = self._selection.selected_channel_ids(
                summary.channels, summary.schemas
            )
            candidates = self._candidate_chunks(summary, selected)
            if not candidates:
                return
            windows = place_windows(self._window, *self._span(summary))
            owners = owner_offsets(candidates, [start for start, _ in windows])
            windows = [
                window
                for window, owner in zip(windows, owners)
                if assignment.offsets is None or owner in assignment.offsets
            ]
            if not windows:
                return
            lead_in = self._lead_in_span_ns(summary, selected)
            low = windows[0][0] - lead_in
            high = max(end for _, end in windows)
            if self._selection.end_time is not None:
                high = min(high, self._selection.end_time)
            # Every chunk of the file that may hold a selected message in the
            # span, not only the candidates: a window's lead-in can lie before
            # the time range, in chunks the listing never considered.
            to_read = [
                c
                for c in summary.chunk_indexes
                if c.message_end_time >= low
                and c.message_start_time < high
                and (
                    not c.message_index_offsets
                    or not selected.isdisjoint(c.message_index_offsets)
                )
            ]
            to_read.sort(key=lambda c: c.chunk_start_offset)
            entries = self._entries(
                self._iter_chunks(
                    f,
                    path,
                    summary,
                    None,
                    chunk_indexes=to_read,
                    selected=selected,
                    time_bounds=(low, high),
                    log_time_order=True,
                )
            )
        if self._video is not None:
            yield from self._decoded_window_rows(assignment, entries, windows)
        else:
            yield from self._window_rows(assignment, entries, windows)

    def _lead_in_span_ns(self, summary: "Summary", selected: Set[int]) -> int:
        """How far before a window this task reads: zero without video topics.

        A topic planning never saw counts as possibly video: its first payload,
        not at hand yet, decides once the chunks are read.
        """
        for cid in selected:
            channel = summary.channels[cid]
            schema = (
                summary.schemas.get(channel.schema_id) if channel.schema_id else None
            )
            if (
                self._video_topics.recognises(
                    channel.topic, schema.name if schema else None
                )
                is not False
            ):
                return self._max_lead_in_ns
        return 0

    def _window_rows(
        self,
        assignment: _Assignment,
        entries: List[_Entry],
        windows: Sequence[Tuple[int, int]],
    ) -> Iterator[pa.Table]:
        """Cut ``entries`` (log-time ordered) into the given windows."""
        path = assignment.path
        digest = self._selection.digest()
        times = [e[2].log_time for e in entries]
        video_channels = self._video_channels(entries)
        range_start = self._selection.start_time
        batch = self._new_batch()
        for start, end in windows:
            first = bisect.bisect_left(times, start)
            last = bisect.bisect_left(times, end)
            in_window = [
                e
                for e in entries[first:last]
                if self._selection.in_time_range(e[2].log_time)
            ]
            if not in_window:
                continue
            # A window that opens before the time range holds frames from the
            # range on, so its lead-in must reach the keyframe before *those*.
            anchor = start if range_start is None else max(start, range_start)
            lead_in: List[_Entry] = []
            for channel_messages in video_channels.values():
                lead_in.extend(self._lead_in(channel_messages, anchor))
            lead_in.sort(key=lambda e: e[2].log_time)
            batch.add(
                CoarseRow(
                    path=path,
                    row_id=f"{path}#[{start},{end})@{digest}",
                    messages=lead_in + in_window,
                    window=(start, end),
                    num_lead_in=len(lead_in),
                )
            )
            if (
                self._target_block_size is not None
                and batch.payload_bytes >= self._target_block_size
            ):
                yield self._finish(batch.build(), assignment, 0)
                batch = self._new_batch()
        if len(batch) > 0:
            yield self._finish(batch.build(), assignment, 0)

    # How many payloads of a channel planning never probed are sniffed before
    # the channel is taken as not video: a VP9 or AV1 inter frame does not name
    # its codec, the next keyframe does.
    _RECOGNITION_PROBES = 32

    def _is_video(
        self, channel: "Channel", schema: Optional["Schema"], payloads: Iterable[bytes]
    ) -> bool:
        """Whether a channel is video: by planning's verdict, or by its bytes."""
        schema_name = schema.name if schema else None
        verdict = self._video_topics.recognises(channel.topic, schema_name)
        if verdict is not None:
            return verdict
        return any(
            detect_codec(payload) is not None
            for payload in itertools.islice(payloads, self._RECOGNITION_PROBES)
        )

    @staticmethod
    def _codec_of(payloads: Iterable[bytes]) -> Optional[VideoCodec]:
        """The first codec any of ``payloads`` names; an inter frame names none."""
        return next(
            (c for c in (detect_codec(p) for p in payloads) if c is not None), None
        )

    def _video_channels(self, entries: List[_Entry]) -> Dict[int, _ChannelMessages]:
        """Index the entries of every video channel for lead-in lookups."""
        grouped: Dict[int, List[_Entry]] = {}
        for entry in entries:
            grouped.setdefault(entry[1].id, []).append(entry)
        by_channel: Dict[int, _ChannelMessages] = {}
        for channel_id, channel_entries in grouped.items():
            schema, channel, _ = channel_entries[0]
            if not self._is_video(
                channel, schema, (e[2].data for e in channel_entries)
            ):
                continue
            # The codec may only be named further on (a VP9 or AV1 keyframe).
            codec = self._codec_of(e[2].data for e in channel_entries)
            if codec is None:
                self._warn_capped_lead_in(channel.topic)
            by_channel[channel_id] = _ChannelMessages(
                channel,
                schema,
                [e[2].log_time for e in channel_entries],
                channel_entries,
                codec,
                [None] * len(channel_entries),
            )
        return by_channel

    def _lead_in(self, messages: _ChannelMessages, anchor: int) -> List[_Entry]:
        """The frames a decoder needs before ``anchor`` on one video channel.

        ``anchor`` is the first log time whose frames the row carries. The
        lead-in is the frames from the last keyframe before ``anchor``, searched
        back at most the look-back cap; none if no keyframe is found there. A
        topic whose codec cannot be parsed carries the whole span: a decoder
        can start on it as long as the stream's keyframe interval fits the cap.
        """
        end = bisect.bisect_left(messages.times, anchor)
        start = bisect.bisect_left(messages.times, anchor - self._max_lead_in_ns)
        if messages.codec is None:
            return messages.entries[start:end]
        if end < len(messages.entries) and messages.is_keyframe_at(end):
            # The window opens on a keyframe: nothing before it is needed.
            return []
        for index in range(end - 1, start - 1, -1):
            if messages.is_keyframe_at(index):
                return messages.entries[index:end]
        return []

    # -- decoded window rows -----------------------------------------------

    def _decoded_topics(self) -> Tuple[str, ...]:
        """The video topics a decoded window row carries frame columns for.

        Settled at planning from the sample files, so the schema is fixed
        before any task runs; a video topic planning never saw stays encoded
        in the message lists (``topics=`` pins the set when files differ).
        """
        if self._video is None or self._granularity != WINDOW_GRANULARITY:
            return ()
        return tuple(sorted(self._video_topics.video))

    def _decoded_window_rows(
        self,
        assignment: _Assignment,
        entries: List[_Entry],
        windows: Sequence[Tuple[int, int]],
    ) -> Iterator[pa.Table]:
        """Cut ``entries`` (log-time ordered) into windows, decoding the video topics.

        Each planned video topic's stream is decoded once for the task, in
        log-time order: primed from the last keyframe before the first owned
        window (within the look-back cap) as a message-granularity task is
        primed, with the same cold-channel and ``fps`` rules, so a frame comes
        out the same whichever task's window holds it. Every kept frame goes to
        each owned window whose span holds its log time (overlapping windows
        duplicate frames, as they duplicate messages); frames before the first
        window, or outside the time range, are dropped. The other topics'
        messages fill the usual list columns. A window is emitted once every
        decoded channel has released a frame past its end or has nothing left,
        so a task holds the frames of a few windows at a time, and a window
        whose decoded frames would pass the row limit fails before it is built.
        """
        assert self._video is not None and self._window is not None
        if not windows:
            return
        video = self._video
        path = assignment.path
        digest = self._selection.digest()
        wanted = set(self._columns) if self._columns is not None else None
        planned = set(self._decoded_topics())
        length = self._window.length_ns
        starts = [start for start, _ in windows]
        range_start = self._selection.start_time
        anchor = starts[0] if range_start is None else max(starts[0], range_start)
        cap = self._max_lead_in_ns

        # Sort the channels: a planned video topic's messages form its stream;
        # any other channel's messages go to the list columns.
        streams: Dict[int, List[_Entry]] = {}
        plain: Set[int] = set()
        for entry in entries:
            schema, channel, message = entry
            if channel.id in streams:
                streams[channel.id].append(entry)
            elif channel.id in plain:
                continue
            elif channel.topic in planned:
                streams[channel.id] = [entry]
            else:
                if self._video_topics.recognises(
                    channel.topic, schema.name if schema else None, message.data
                ) and log_once(f"mcap_unplanned_video:{channel.topic}"):
                    logger.warning(
                        "Video topic %r in %r was not in the files planning "
                        "sampled, so its frames stay encoded in the message lists "
                        "of the window rows; pass topics=[...] naming the video "
                        "topics to decode it.",
                        channel.topic,
                        path,
                    )
                plain.add(channel.id)

        channels: Dict[int, _WindowChannel] = {}
        for channel_id, stream in streams.items():
            topic = stream[0][1].topic
            wants_frames = wanted is None or f"{FRAMES_PREFIX}{topic}" in wanted
            # With ``frames:`` pruned the channel runs in mirror mode (no
            # decoder, only the frame times a decoder would keep), and it runs
            # even when ``frame_times:`` is pruned too: a count, or a projection
            # to the window bounds, must still see every window that holds a
            # frame, as the message path keeps its rows without ``frame``.
            # A VP9 or AV1 inter frame does not name its codec: look on until
            # a payload (the next keyframe) does.
            codec = next(
                (c for c in (detect_codec(e[2].data) for e in stream) if c is not None),
                None,
            )
            if codec is None:
                if log_once(f"mcap_codec_pending:{topic}"):
                    logger.warning(
                        "The codec of video topic %r in %r cannot be told from any "
                        "of its payloads in this task; its frames are skipped.",
                        topic,
                        path,
                    )
                continue
            state = _WindowChannel(
                topic=topic,
                codec=codec,
                thinner=FrameThinner(video.fps_interval_ns),
                path=path,
                cap_ns=cap,
            )
            if wants_frames:
                state.decoder = FrameDecoder(
                    codec, resize=video.resize, thinner=state.thinner
                )
            times = [e[2].log_time for e in stream]
            first = bisect.bisect_left(times, anchor)
            state.remaining = len(stream) - first
            state.prime(
                stream[bisect.bisect_left(times, anchor - cap) : first], before=anchor
            )
            channels[channel_id] = state

        pending = [_PendingWindow(start, end) for start, end in windows]
        next_window = 0
        # One frame-shape memory per task, shared by its batches, so a window
        # without frames still gets a tensor of the topic's shape.
        frame_shape: Dict[str, Tuple[int, int]] = (
            {t: video.resize for t in self._decoded_topics()}
            if video.resize is not None
            else {}
        )
        batch = self._new_batch(frame_shape)

        def windows_holding(log_time: int) -> range:
            # Windows share one length, so those holding ``log_time`` are the
            # ones starting in ``(log_time - length, log_time]``.
            low = bisect.bisect_right(starts, log_time - length)
            high = bisect.bisect_right(starts, log_time)
            return range(max(low, next_window), high)

        def place_frame(log_time: int, frame: Any, topic: str) -> None:
            if log_time < anchor or not self._selection.in_time_range(log_time):
                return
            for index in windows_holding(log_time):
                window = pending[index]
                frame_times, frames = window.frames.setdefault(topic, ([], []))
                frame_times.append(log_time)
                if frame is None:
                    continue
                frames.append(frame)
                window.decoded_bytes += frame.nbytes
                if window.decoded_bytes > self._max_row_bytes:
                    raise ValueError(
                        f"The decoded frames of window [{window.start}, {window.end}) "
                        f"of {path!r} would exceed {self._max_row_bytes} bytes "
                        "(RAY_DATA_MCAP_MAX_ROW_BYTES): thin them with "
                        "VideoOptions(fps=...), shrink them with "
                        "resize=(height, width), or read at "
                        "read_granularity='message'."
                    )

        def take_frames(state: _WindowChannel, released: List[Tuple[int, Any]]):
            for log_time, frame in released:
                place_frame(log_time, frame, state.topic)

        def emit_ready(up_to: Optional[int]) -> Iterator[pa.Table]:
            """Emit the windows complete by now; all of them when ``up_to`` is None."""
            nonlocal next_window, batch
            while next_window < len(pending):
                window = pending[next_window]
                if up_to is not None and (
                    window.end > up_to
                    or not all(s.past(window.end) for s in channels.values())
                ):
                    break
                next_window += 1
                if not window.messages and not any(
                    frame_times for frame_times, _ in window.frames.values()
                ):
                    continue
                batch.add(
                    CoarseRow(
                        path=path,
                        row_id=f"{path}#[{window.start},{window.end})@{digest}",
                        messages=window.messages,
                        window=(window.start, window.end),
                        num_lead_in=0,
                        frames=window.frames,
                        decoded_bytes=window.decoded_bytes,
                    )
                )
                # The row owns those containers now; the window lets go of
                # them, so the task holds only its in-flight windows in memory.
                window.messages = []
                window.frames = {}
                window.decoded_bytes = 0
                if (
                    self._target_block_size is not None
                    and batch.payload_bytes >= self._target_block_size
                ):
                    yield self._finish(batch.build(), assignment, 0)
                    batch = self._new_batch(frame_shape)

        first_entry = bisect.bisect_left([e[2].log_time for e in entries], starts[0])
        for entry in entries[first_entry:]:
            schema, channel, message = entry
            log_time = message.log_time
            state = channels.get(channel.id)
            if state is not None:
                if log_time >= anchor and not state.done:
                    take_frames(state, state.feed(message))
                    if state.remaining == 0:
                        take_frames(state, state.finish())
            elif channel.id in plain:
                if self._selection.in_time_range(log_time):
                    for index in windows_holding(log_time):
                        pending[index].messages.append(entry)
            # A planned video channel with both columns pruned, or whose codec
            # nothing told: its messages are neither frames nor list entries.
            yield from emit_ready(log_time)
        for state in channels.values():
            if not state.done:
                take_frames(state, state.finish())
        yield from emit_ready(None)
        if len(batch) > 0:
            yield self._finish(batch.build(), assignment, 0)

    def _topic_tables(
        self, f: Any, assignment: _Assignment, summary: Optional["Summary"]
    ) -> Iterator[pa.Table]:
        """Emit one row per topic this task was assigned (one, when indexed)."""
        path = assignment.path
        if summary is None:
            by_topic: Dict[str, List[_Entry]] = {}
            sizes: Dict[str, int] = {}
            for schema, channel, message, _ in self._iter_unindexed(f, path):
                topic = channel.topic
                if assignment.topic is not None and topic != assignment.topic:
                    continue
                sizes[topic] = sizes.get(topic, 0) + len(message.data)
                if sizes[topic] > self._max_row_bytes:
                    self._raise_row_too_large(
                        sizes[topic], f"topic {topic!r} of {path!r}", partial=True
                    )
                by_topic.setdefault(topic, []).append((schema, channel, message))
            for topic_entries in by_topic.values():
                topic_entries.sort(key=lambda e: e[2].log_time)
            topics = (
                [assignment.topic] if assignment.topic is not None else sorted(by_topic)
            )
            groups = [(topic, by_topic.get(topic, [])) for topic in topics]
        else:
            assert assignment.topic is not None, "an indexed topic row names its topic"
            selected = self._selection.selected_channel_ids(
                summary.channels, summary.schemas
            )
            topic_ids = {
                cid
                for cid in selected
                if summary.channels[cid].topic == assignment.topic
            }
            entries = self._entries(
                self._iter_chunks(
                    f,
                    path,
                    summary,
                    assignment.offsets,
                    chunk_indexes=self._candidate_chunks(summary, topic_ids),
                    selected=topic_ids,
                    log_time_order=True,
                ),
                what=f"topic {assignment.topic!r} of {path!r}",
            )
            groups = [(assignment.topic, entries)]
        digest = self._selection.digest()
        batch = self._new_batch()
        for topic, topic_entries in groups:
            topic_entries, num_lead_in = self._decodable_topic(
                f, path, summary, topic, topic_entries
            )
            if not topic_entries:
                continue
            row = CoarseRow(
                path=path,
                row_id=f"{path}#{topic}@{digest}",
                messages=topic_entries,
                topic=topic,
                num_lead_in=num_lead_in,
            )
            self._check_row_size(row, f"topic {topic!r} of {path!r}")
            batch.add(row)
        if len(batch) > 0:
            yield self._finish(batch.build(), assignment, 0)

    def _decodable_topic(
        self,
        f: Any,
        path: str,
        summary: Optional["Summary"],
        topic: str,
        entries: List[_Entry],
    ) -> Tuple[List[_Entry], int]:
        """A video topic's messages from a keyframe on, and how many are lead-in.

        A topic row must decode on its own. When the selected messages do not
        open on a keyframe and a ``time_range`` cut the stream, the frames
        from the last keyframe before the range (within the look-back cap) are
        prepended as lead-in, as window rows do; without one, the frames before
        the first keyframe in range are dropped, and a topic with no keyframe
        anywhere is dropped with a warning rather than emitted undecodable.
        """
        if not entries:
            return entries, 0
        schema, channel, message = entries[0]
        if not self._is_video(channel, schema, (m.data for _, _, m in entries)):
            return entries, 0
        range_start = self._selection.start_time
        lead: List[_Entry] = []
        if range_start is not None:
            channel_ids = {c.id for _, c, _ in entries}
            lead = self._look_back(f, path, summary, channel_ids, range_start)
        # The codec may only be named further on, or in the look-back (a VP9
        # or AV1 keyframe); an inter frame says nothing.
        codec = self._codec_of(m.data for _, _, m in entries) or self._codec_of(
            m.data for _, _, m in lead
        )
        if codec is None:
            self._warn_capped_lead_in(channel.topic)
            return entries, 0
        if codec.every_frame_is_a_keyframe or is_keyframe(message.data, codec):
            return entries, 0
        if range_start is not None:
            for index in range(len(lead) - 1, -1, -1):
                if is_keyframe(lead[index][2].data, codec):
                    lead = lead[index:]
                    return lead + entries, len(lead)
        for index, (_, _, candidate) in enumerate(entries):
            if is_keyframe(candidate.data, codec):
                return entries[index:], 0
        if log_once(f"mcap_topic_no_keyframe:{topic}"):
            logger.warning(
                "Video topic %r of %r has no keyframe in the selected span nor in "
                "the %.2f s before it, so its topic row cannot be decoded and is "
                "dropped. Widen time_range, or raise RAY_DATA_MCAP_MAX_LEAD_IN_S.",
                topic,
                path,
                self._max_lead_in_ns / 1e9,
            )
        return [], 0

    def _look_back(
        self,
        f: Any,
        path: str,
        summary: Optional["Summary"],
        channel_ids: Set[int],
        before: int,
    ) -> List[_Entry]:
        """The channels' messages in the look-back span before ``before``."""
        low = max(0, before - self._max_lead_in_ns)
        if low >= before:
            return []
        if summary is None:
            # File order is not log-time order: sort, as every other unindexed
            # coarse path does, so the walk back for a keyframe is by time.
            return sorted(
                (
                    (schema, channel, message)
                    for schema, channel, message, _ in self._iter_unindexed(
                        f, path, time_bounds=(low, before)
                    )
                    if channel.id in channel_ids
                ),
                key=lambda e: e[2].log_time,
            )
        chunks = sorted(
            (
                c
                for c in summary.chunk_indexes
                if c.message_end_time >= low
                and c.message_start_time < before
                and (
                    not c.message_index_offsets
                    or not channel_ids.isdisjoint(c.message_index_offsets)
                )
            ),
            key=lambda c: c.chunk_start_offset,
        )
        if not chunks:
            return []
        return self._entries(
            self._iter_chunks(
                f,
                path,
                summary,
                None,
                chunk_indexes=chunks,
                selected=channel_ids,
                time_bounds=(low, before),
                log_time_order=True,
            )
        )

    def _file_tables(
        self, f: Any, assignment: _Assignment, summary: Optional["Summary"]
    ) -> Iterator[pa.Table]:
        """Emit the one row holding every selected message of the file."""
        path = assignment.path
        what = f"file {path!r}"
        if summary is None:
            entries = self._entries(self._iter_unindexed(f, path), what=what)
            entries.sort(key=lambda e: e[2].log_time)
        else:
            entries = self._entries(
                self._iter_chunks(f, path, summary, None, log_time_order=True),
                what=what,
            )
        if not entries:
            return
        row = CoarseRow(
            path=path,
            row_id=f"{path}@{self._selection.digest()}",
            messages=entries,
        )
        self._check_row_size(row, f"file {path!r}")
        batch = self._new_batch()
        batch.add(row)
        yield self._finish(batch.build(), assignment, 0)

    def _new_batch(
        self, frame_shape: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> CoarseRowBatch:
        batch = CoarseRowBatch(
            granularity=self._granularity,
            include_metadata=self._include_metadata,
            include_row_id=self._include_row_id,
            video_topics=self._decoded_topics(),
        )
        if frame_shape is not None:
            batch.frame_shape = frame_shape
        return batch

    def _warn_capped_lead_in(self, topic: str) -> None:
        if log_once(f"mcap_capped_lead_in:{topic}"):
            logger.warning(
                "Video topic %r uses a codec whose keyframes cannot be told from "
                "the bytes (JPEG, PNG, H.264, H.265, VP9 and AV1 are recognised). "
                "Its window rows carry the whole %.2f s look-back as lead-in rather "
                "than the frames since the last keyframe, and its topic rows start "
                "at the first message. RAY_DATA_MCAP_MAX_LEAD_IN_S sets the span.",
                topic,
                self._max_lead_in_ns / 1e9,
            )

    def _check_row_size(self, row: CoarseRow, what: str) -> None:
        if row.payload_bytes > self._max_row_bytes:
            self._raise_row_too_large(row.payload_bytes, what, partial=False)

    def _raise_row_too_large(self, payload_bytes: int, what: str, partial: bool):
        raise ValueError(
            f"The row for {what} would carry "
            f"{'at least ' if partial else ''}{payload_bytes} bytes of payload, "
            f"over the {self._max_row_bytes}-byte limit for one row "
            "(RAY_DATA_MCAP_MAX_ROW_BYTES). Read this data at "
            "read_granularity='window' or 'message' instead."
        )

    # -- attachment and metadata rows --------------------------------------

    def _record_tables(self, f: Any, assignment: _Assignment) -> Iterator[pa.Table]:
        """Emit the Attachment or Metadata rows this task was assigned.

        An indexed file's rows name the records by byte offset, so each is
        read with one seek. A whole-file row means the records are not indexed;
        the file is scanned for them, and ``time_range`` is applied to
        attachments either way.
        """
        from mcap.records import Attachment, Metadata

        path = assignment.path
        attachments = self._granularity == ATTACHMENT_GRANULARITY
        batch = RecordRowBatch(
            granularity=self._granularity, include_row_id=self._include_row_id
        )
        if assignment.offsets is not None:
            located = [
                (
                    read_record_at(f, offset),
                    (attachment_unit_id if attachments else metadata_unit_id)(
                        path, offset
                    ),
                    offset,
                )
                for offset in sorted(assignment.offsets)
            ]
        else:
            kind = "a" if attachments else "md"
            scanned = iter_records(f, Attachment if attachments else Metadata)
            located = [
                (record, unindexed_record_row_id(path, kind, ordinal), None)
                for ordinal, record in enumerate(scanned)
            ]
        for record, row_id, offset in located:
            if attachments:
                if not isinstance(record, Attachment):
                    raise ValueError(
                        f"MCAP file {path!r}: expected an Attachment record at "
                        f"offset {offset}, found {type(record).__name__}"
                    )
                if not self._selection.in_time_range(record.log_time):
                    continue
                batch.add_attachment(path, row_id, record)
            else:
                if not isinstance(record, Metadata):
                    raise ValueError(
                        f"MCAP file {path!r}: expected a Metadata record at offset "
                        f"{offset}, found {type(record).__name__}"
                    )
                batch.add_metadata(path, row_id, record)
            if (
                self._target_block_size is not None
                and batch.payload_bytes >= self._target_block_size
            ):
                yield self._finish(batch.build(), assignment, 0)
                batch = RecordRowBatch(
                    granularity=self._granularity, include_row_id=self._include_row_id
                )
        if len(batch) > 0:
            yield self._finish(batch.build(), assignment, 0)

    # -- finishing a table -------------------------------------------------

    def _finish(
        self, table: pa.Table, assignment: _Assignment, rows_before: int
    ) -> pa.Table:
        """Append partition and synthesized columns, then apply the projection."""
        wanted = set(self._columns) if self._columns is not None else None
        num_rows = table.num_rows
        path = assignment.path
        if self._partition_parser is not None:
            for name, value in self._partition_parser(path).items():
                if wanted is not None and name not in wanted:
                    continue
                if name in table.column_names:
                    table = table.drop([name])
                table = table.append_column(
                    name, self._partition_value_array(name, value, num_rows)
                )
        position = ReadUnitPosition(unit=assignment.unit, rows_before=rows_before)
        for column in self._synthesized_columns:
            if wanted is not None and column.name not in wanted:
                continue
            if column.name in table.column_names:
                table = table.drop([column.name])
            table = table.append_column(column.name, column.compute(position, num_rows))
        if self._columns is not None:
            produced = set(table.column_names)
            table = table.select([c for c in self._columns if c in produced])
            if table.num_columns == 0 and num_rows > 0:
                table = table.append_column(
                    _BATCH_SIZE_PRESERVING_STUB_COL_NAME, pa.nulls(num_rows)
                )
        # A JSON payload Arrow cannot type falls back to Ray's pickled-object
        # extension. Unpickling runs arbitrary code, so, like every other
        # datasource, refuse such a column unless the user opted in.
        raise_on_pickle_object_columns(table)
        return table

    def _partition_value_array(self, name: str, value: Any, num_rows: int) -> pa.Array:
        """Broadcast one path-derived partition value, typed by the schema if known."""
        as_str = None if value is None else str(value)
        array = pa.repeat(pa.scalar(as_str, type=pa.string()), num_rows)
        if self._schema is not None:
            idx = self._schema.get_field_index(name)
            if idx != -1 and self._schema.field(idx).type != pa.string():
                array = array.cast(self._schema.field(idx).type)
        return array


def _assignments(manifest: FileManifest) -> List[_Assignment]:
    """Group a manifest's rows by file (and topic): what this task reads of each.

    A row with chunk metadata contributes its ``unit_ids`` (chunk byte offsets);
    a row without it means the whole file, which wins over any offsets listed
    for the same path. A topic-granularity row also names its topic.
    """
    owned: Dict[Tuple[str, Optional[str]], Optional[Set[int]]] = {}
    for path, metadata in zip(manifest.paths, manifest.file_chunk_metadatas):
        path = str(path)
        topic = None
        if metadata is not None and metadata.get("topic") is not None:
            topic = str(metadata["topic"])
        key = (path, topic)
        if metadata is None or "unit_ids" not in metadata:
            owned[key] = None
            continue
        offsets = {int(i) for i in metadata["unit_ids"]}
        current = owned.get(key, offsets)
        if current is not None:
            current.update(offsets)
        owned[key] = current
    return [
        _Assignment(path=path, offsets=offsets, topic=topic)
        for (path, topic), offsets in owned.items()
    ]
