"""Window placement, window ownership and the coarse row layouts.

Window, topic and file rows pack many messages into one row of parallel lists:
entry *i* of each list column is the same message, in log-time order. The
functions here are pure, so the indexer, the reader and the tests reason about
the same windows and the same rows.
"""

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from ray.data._internal.datasource_v2.formats.mcap.mcap_options import (
    FILE_GRANULARITY,
    ROW_ID_COLUMN,
    TOPIC_GRANULARITY,
    WINDOW_GRANULARITY,
    WindowSpec,
)
from ray.data._internal.tensor_extensions.arrow import (
    ArrowVariableShapedTensorArray,
    ArrowVariableShapedTensorType,
)

if TYPE_CHECKING:
    from mcap.records import Channel, ChunkIndex, Message, Schema

# A window as ``[start, end)`` in nanoseconds.
Window = Tuple[int, int]


def place_windows(spec: WindowSpec, span_start: int, span_end: int) -> List[Window]:
    """The windows of ``spec`` that meet the closed span ``[span_start, span_end]``.

    ``span_start`` and ``span_end`` are the first and last selected log times of
    a file. With ``anchor="file_start"`` window 0 begins at ``span_start`` and
    nothing precedes it; with ``"epoch"`` or an absolute anchor, window starts
    are the anchor plus any whole number of strides, so windows line up across
    files. ``drop_partial`` drops the windows that end after the last message.
    """
    if span_end < span_start:
        return []
    length, stride = spec.length_ns, spec.stride_ns
    if spec.anchor == "file_start":
        base, first_k = span_start, 0
    else:
        base = 0 if spec.anchor == "epoch" else int(spec.anchor)
        # The first window that still reaches span_start: base + k*stride > span_start - length.
        first_k = -((span_start - length + 1 - base) // -stride)
    windows: List[Window] = []
    k = first_k
    while True:
        start = base + k * stride
        if start > span_end:
            break
        end = start + length
        if spec.drop_partial and end > span_end + 1:
            break
        windows.append((start, end))
        k += 1
    return windows


def owner_offsets(
    chunk_indexes: Sequence["ChunkIndex"], window_starts: Iterable[int]
) -> List[int]:
    """The byte offset of the chunk that owns each window.

    A window belongs to the last chunk, in log-time order of chunk starts,
    that starts at or before the window; a window before every chunk belongs
    to the first. The task that owns that chunk emits the window and reads
    past its own chunks to fill it. Every task derives this from the same
    summary, so each window is emitted exactly once.
    """
    ordered = sorted(
        chunk_indexes, key=lambda c: (c.message_start_time, c.chunk_start_offset)
    )
    starts = [c.message_start_time for c in ordered]
    owners = []
    for window_start in window_starts:
        index = max(bisect.bisect_right(starts, window_start) - 1, 0)
        owners.append(ordered[index].chunk_start_offset)
    return owners


# -- row layouts -------------------------------------------------------------

CHANNEL_STRUCT = pa.struct(
    [
        pa.field("channel_id", pa.int32()),
        pa.field("topic", pa.string()),
        pa.field("message_encoding", pa.string()),
        pa.field("schema_name", pa.string()),
        pa.field("schema_encoding", pa.string()),
        pa.field("schema_data", pa.binary()),
        pa.field("metadata", pa.map_(pa.string(), pa.string())),
    ]
)

_MESSAGE_LISTS = [
    pa.field("channel_id", pa.list_(pa.int32())),
    pa.field("log_time", pa.list_(pa.int64())),
    pa.field("publish_time", pa.list_(pa.int64())),
    pa.field("sequence", pa.list_(pa.uint32())),
    # A row is one list element in one contiguous Arrow array. With int32
    # offsets a row past 2 GiB fails in Ray's chunk combiner or Arrow's builder;
    # int64 offsets cost 4 bytes per message and keep the schema independent of
    # RAY_DATA_MCAP_MAX_ROW_BYTES, which stays a memory guardrail.
    pa.field("data", pa.large_list(pa.large_binary())),
]

# Decoded window rows (``window`` granularity with ``video``): per video topic
# the window's frames and their log times, in columns named after the topic.
FRAMES_PREFIX = "frames:"
FRAME_TIMES_PREFIX = "frame_times:"
FRAMES_TYPE = ArrowVariableShapedTensorType(pa.uint8(), 4)


def decoded_window_fields(video_topics: Sequence[str]) -> List[pa.Field]:
    """The two columns a decoded window row carries per video topic.

    ``frames:<topic>`` holds one ``uint8`` tensor per row of shape
    ``(n, height, width, 3)``, ragged across rows since windows hold different
    numbers of frames; ``frame_times:<topic>`` the ``log_time`` of each frame,
    ascending.

    Args:
        video_topics: The decoded topics, in column order.

    Returns:
        The fields, two per topic.
    """
    fields = []
    for topic in video_topics:
        fields.append(pa.field(f"{FRAMES_PREFIX}{topic}", FRAMES_TYPE))
        fields.append(pa.field(f"{FRAME_TIMES_PREFIX}{topic}", pa.list_(pa.int64())))
    return fields


def coarse_row_schema(
    granularity: str,
    *,
    include_metadata: bool,
    include_row_id: bool,
    video_topics: Sequence[str] = (),
) -> pa.Schema:
    """The schema of window, topic or file rows.

    Payloads stay encoded (``large_list<large_binary>``, so one row is not
    capped at 2 GiB by its offsets): a row's decoding needs are met by
    its ``channels`` column, one struct per channel present in the row, which
    ``include_metadata=False`` drops. ``path`` is always present, because a
    coarse row is meaningless without the recording it came from. A decoded
    window row (``video`` given) adds ``frames:<topic>`` and
    ``frame_times:<topic>`` per video topic, whose messages then leave the
    lists.

    Args:
        granularity: ``window``, ``topic`` or ``file``.
        include_metadata: Whether the ``channels`` column is present.
        include_row_id: Whether ``row_id`` is present.
        video_topics: The topics decoded into frame columns; window rows only.

    Returns:
        The schema, columns in output order.
    """
    fields = [pa.field("path", pa.string())]
    if include_row_id:
        fields.append(pa.field(ROW_ID_COLUMN, pa.string()))
    if granularity == WINDOW_GRANULARITY:
        fields += [
            pa.field("window_start", pa.int64()),
            pa.field("window_end", pa.int64()),
            pa.field("num_messages", pa.int64()),
            pa.field("num_lead_in", pa.int32()),
            pa.field("topic", pa.list_(pa.string())),
        ]
    elif granularity == TOPIC_GRANULARITY:
        fields += [
            pa.field("topic", pa.string()),
            pa.field("start_time", pa.int64()),
            pa.field("end_time", pa.int64()),
            pa.field("num_messages", pa.int64()),
            pa.field("num_lead_in", pa.int32()),
        ]
    elif granularity == FILE_GRANULARITY:
        fields += [
            pa.field("start_time", pa.int64()),
            pa.field("end_time", pa.int64()),
            pa.field("num_messages", pa.int64()),
            pa.field("topic", pa.list_(pa.string())),
        ]
    else:
        raise ValueError(f"not a coarse granularity: {granularity!r}")
    fields += _MESSAGE_LISTS
    if include_metadata:
        fields.append(pa.field("channels", pa.list_(CHANNEL_STRUCT)))
    if video_topics:
        assert granularity == WINDOW_GRANULARITY, "only window rows are decoded"
        fields += decoded_window_fields(video_topics)
    return pa.schema(fields)


@dataclass
class CoarseRow:
    """One window, topic or file row before it is turned into Arrow."""

    path: str
    row_id: str
    # Messages in log-time order, with their channel and schema.
    messages: List[Tuple[Optional["Schema"], "Channel", "Message"]]
    # Window rows only.
    window: Optional[Window] = None
    # Window and topic rows: how many leading messages precede the row's own
    # span (frames back to a keyframe, so the row decodes on its own).
    num_lead_in: int = 0
    # Topic rows only.
    topic: Optional[str] = None
    # Decoded window rows only: per video topic, the kept frames' log times
    # and the frames themselves (``uint8`` arrays), in log-time order.
    frames: Optional[Dict[str, Tuple[List[int], List[Any]]]] = None
    decoded_bytes: int = 0

    @property
    def payload_bytes(self) -> int:
        return sum(len(m.data) for _, _, m in self.messages)


def _stack_frames(
    frames: List[Any],
    topic: str,
    row: CoarseRow,
    empty_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """One ``(n, height, width, 3)`` array per row.

    A row with no frame gets a ``(0, height, width, 3)`` array when the frame
    size is known (``empty_shape``: from ``resize`` or a frame seen earlier in
    the task), so the column keeps one spatial shape; ``(0, 0, 0, 3)`` only
    when no frame of the topic was ever decoded.
    """
    if not frames:
        height, width = empty_shape if empty_shape is not None else (0, 0)
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) > 1:
        start, end = row.window if row.window is not None else (0, 0)
        raise ValueError(
            f"Topic {topic!r} changes frame size inside window [{start}, {end}) "
            f"of {row.path!r} ({sorted(shapes)}); pass VideoOptions(resize="
            "(height, width)) to decode it to one size."
        )
    return np.stack(frames)


@dataclass
class CoarseRowBatch:
    """Accumulates coarse rows and builds one Arrow table from them."""

    granularity: str
    include_metadata: bool
    include_row_id: bool
    # Decoded window rows: the topics with frame columns, in column order.
    video_topics: Sequence[str] = ()
    # Per decoded topic, the (height, width) of its frames: seeded from
    # ``resize`` or learnt from the first frame built, and shared across the
    # batches of one task, so a window without frames still gets a tensor of
    # the topic's shape.
    frame_shape: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    rows: List[CoarseRow] = field(default_factory=list)
    # Encoded payload plus decoded frame bytes: what the table will weigh.
    payload_bytes: int = 0

    def add(self, row: CoarseRow) -> None:
        self.rows.append(row)
        self.payload_bytes += row.payload_bytes + row.decoded_bytes

    def __len__(self) -> int:
        return len(self.rows)

    def build(self) -> pa.Table:
        columns: Dict[str, Any] = {
            "path": pa.array([r.path for r in self.rows], pa.string())
        }
        if self.include_row_id:
            columns[ROW_ID_COLUMN] = pa.array(
                [r.row_id for r in self.rows], pa.string()
            )
        times = [[m.log_time for _, _, m in r.messages] for r in self.rows]
        if self.granularity == WINDOW_GRANULARITY:
            windows = []
            for row in self.rows:
                assert row.window is not None, "a window row names its window"
                windows.append(row.window)
            columns["window_start"] = pa.array([w[0] for w in windows], pa.int64())
            columns["window_end"] = pa.array([w[1] for w in windows], pa.int64())
            columns["num_messages"] = pa.array(
                [len(r.messages) for r in self.rows], pa.int64()
            )
            columns["num_lead_in"] = pa.array(
                [r.num_lead_in for r in self.rows], pa.int32()
            )
            columns["topic"] = pa.array(
                [[c.topic for _, c, _ in r.messages] for r in self.rows],
                pa.list_(pa.string()),
            )
        else:
            if self.granularity == TOPIC_GRANULARITY:
                columns["topic"] = pa.array([r.topic for r in self.rows], pa.string())
            # The span is that of the row's own messages; a lead-in precedes it.
            columns["start_time"] = pa.array(
                [t[r.num_lead_in] for t, r in zip(times, self.rows)], pa.int64()
            )
            columns["end_time"] = pa.array([t[-1] for t in times], pa.int64())
            columns["num_messages"] = pa.array(
                [len(r.messages) for r in self.rows], pa.int64()
            )
            if self.granularity == TOPIC_GRANULARITY:
                columns["num_lead_in"] = pa.array(
                    [r.num_lead_in for r in self.rows], pa.int32()
                )
            if self.granularity == FILE_GRANULARITY:
                columns["topic"] = pa.array(
                    [[c.topic for _, c, _ in r.messages] for r in self.rows],
                    pa.list_(pa.string()),
                )
        columns["channel_id"] = pa.array(
            [[m.channel_id for _, _, m in r.messages] for r in self.rows],
            pa.list_(pa.int32()),
        )
        columns["log_time"] = pa.array(times, pa.list_(pa.int64()))
        columns["publish_time"] = pa.array(
            [[m.publish_time for _, _, m in r.messages] for r in self.rows],
            pa.list_(pa.int64()),
        )
        columns["sequence"] = pa.array(
            [[m.sequence for _, _, m in r.messages] for r in self.rows],
            pa.list_(pa.uint32()),
        )
        columns["data"] = pa.array(
            [[m.data for _, _, m in r.messages] for r in self.rows],
            pa.large_list(pa.large_binary()),
        )
        if self.include_metadata:
            columns["channels"] = pa.array(
                [_channel_structs(r.messages) for r in self.rows],
                pa.list_(CHANNEL_STRUCT),
            )
        for topic in self.video_topics:
            known = self.frame_shape.get(topic)
            if known is None:
                for row in self.rows:
                    frames = (row.frames or {}).get(topic, ([], []))[1]
                    if frames:
                        known = (int(frames[0].shape[0]), int(frames[0].shape[1]))
                        break
            stacks = []
            frame_times = []
            for row in self.rows:
                times, frames = (row.frames or {}).get(topic, ([], []))
                frame_times.append(list(times))
                stack = _stack_frames(frames, topic, row, known)
                if len(stack):
                    known = (int(stack.shape[1]), int(stack.shape[2]))
                stacks.append(stack)
            if known is not None:
                self.frame_shape[topic] = known
            # Always the variable-shaped type: windows hold different numbers of
            # frames, and the planned schema says so even when a block's rows
            # happen to agree.
            columns[
                f"{FRAMES_PREFIX}{topic}"
            ] = ArrowVariableShapedTensorArray.from_numpy(stacks)
            columns[f"{FRAME_TIMES_PREFIX}{topic}"] = pa.array(
                frame_times, pa.list_(pa.int64())
            )
        return pa.table(columns)


def _channel_structs(
    messages: Iterable[Tuple[Optional["Schema"], "Channel", "Message"]]
) -> List[Dict[str, Any]]:
    """One struct per distinct channel among ``messages``, in first-seen order."""
    seen: Dict[int, Dict[str, Any]] = {}
    for schema, channel, _ in messages:
        if channel.id in seen:
            continue
        seen[channel.id] = {
            "channel_id": channel.id,
            "topic": channel.topic,
            "message_encoding": channel.message_encoding,
            "schema_name": schema.name if schema else None,
            "schema_encoding": schema.encoding if schema else None,
            "schema_data": schema.data if schema else None,
            "metadata": list(channel.metadata.items()),
        }
    return list(seen.values())
