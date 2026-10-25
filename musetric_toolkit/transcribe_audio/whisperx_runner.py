import collections
import itertools
import logging
import re
import shutil
import sys
import typing
import zlib
from contextlib import contextmanager, suppress
from importlib.util import find_spec
from pathlib import Path

import omegaconf.base
import omegaconf.dictconfig
import omegaconf.listconfig
import omegaconf.nodes
import pyannote.audio.core.model
import torch
import whisperx
from lightning.pytorch import __version__ as pl_version
from lightning.pytorch.utilities.migration import migrate_checkpoint, pl_legacy_patch
from packaging.version import Version

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio.system_info import (
    print_acceleration_info,
    setup_torch_optimization,
)
from musetric_toolkit.transcribe_audio.download_progress import (
    intercept_hf_downloads,
)
from musetric_toolkit.transcribe_audio.hallucination_filter import (
    _is_hallucination,
    filter_hallucinated_segments,
    strip_credit_segments,
)
from musetric_toolkit.transcribe_audio.language_detector import (
    detect_language_loudest,
)
from musetric_toolkit.transcribe_audio.silence_filter import (
    filter_silent_segments,
)
from musetric_toolkit.transcribe_audio.spectral_chunker import (
    SAMPLE_RATE,
    build_compaction,
    compute_packed_chunks,
    map_time,
)

# WhisperX integration for transcription and alignment.
# WhisperX is BSD-2-Clause; see thirdPartyNotices.md.


def configure_torch_serialization() -> None:
    # Allowlist safe types used by OmegaConf/typing in PyTorch weights-only loads.
    torch.serialization.add_safe_globals(
        [
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
            omegaconf.nodes.AnyNode,
            typing.Any,
            int,
            list,
            dict,
            tuple,
            set,
            collections.defaultdict,
            collections.OrderedDict,
            torch.torch_version.TorchVersion,
            pyannote.audio.core.model.Introspection,
        ]
    )


def patch_speechbrain_lazy_imports() -> None:
    try:
        from speechbrain.utils import importutils  # noqa: PLC0415
    except Exception as error:
        logging.debug("SpeechBrain lazy import patch skipped: %s", error)
        return

    if getattr(importutils.LazyModule.ensure_module, "_musetric_patched", False):
        return

    original_ensure_module = importutils.LazyModule.ensure_module

    def ensure_module(self, stacklevel: int):
        try:
            importer_frame = importutils.inspect.getframeinfo(
                importutils.sys._getframe(stacklevel + 1)
            )
        except AttributeError:
            return original_ensure_module(self, stacklevel)

        if Path(importer_frame.filename).name == "inspect.py":
            raise AttributeError()

        return original_ensure_module(self, stacklevel)

    ensure_module._musetric_patched = True
    importutils.LazyModule.ensure_module = ensure_module


@contextmanager
def allow_unsafe_torch_load():
    original_torch_load = torch.load
    original_serialization_load = torch.serialization.load

    def torch_load_unrestricted(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_unrestricted
    torch.serialization.load = torch_load_unrestricted
    try:
        yield
    finally:
        torch.load = original_torch_load
        torch.serialization.load = original_serialization_load


def maybe_upgrade_whisperx_checkpoint() -> None:
    try:
        spec = find_spec("whisperx")
        if not spec or not spec.origin:
            return
        checkpoint_path = (
            Path(spec.origin).resolve().parent / "assets" / "pytorch_model.bin"
        )
        if not checkpoint_path.is_file():
            return

        with pl_legacy_patch():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )

        ckpt_version = checkpoint.get("pytorch-lightning_version")
        if not ckpt_version:
            return
        if Version(ckpt_version) >= Version(pl_version):
            return

        backup_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.bak")
        if not backup_path.exists():
            shutil.copyfile(checkpoint_path, backup_path)

        migrate_checkpoint(checkpoint)
        torch.save(checkpoint, checkpoint_path)
    except Exception as error:
        logging.debug("WhisperX checkpoint upgrade skipped: %s", error)


_PROGRESS_PATTERN = re.compile(r"^Progress:\s+([0-9]+(?:\.[0-9]+)?)%")


class ProgressTracker:
    def __init__(self, min_delta: float = 0.01) -> None:
        self._min_delta = min_delta
        self._last = -1.0

    def report_fraction(self, progress: float) -> None:
        progress = max(0.0, min(progress, 1.0))
        if progress <= self._last:
            return
        if (
            self._last >= 0.0
            and progress < self._last + self._min_delta
            and progress < 1.0
        ):
            return
        self._last = progress
        send_message({"type": "progress", "progress": progress})

    def report_percent(self, percent: float) -> None:
        self.report_fraction(percent / 100.0)

    def ensure_minimum(self, progress: float) -> None:
        if progress > self._last:
            self.report_fraction(progress)

    def finalize(self) -> None:
        self.report_fraction(1.0)


class ProgressLineInterceptor:
    def __init__(self, stream, tracker: ProgressTracker) -> None:
        self._stream = stream
        self._tracker = tracker
        self._buffer = ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _maybe_report_progress(self, line: str) -> bool:
        match = _PROGRESS_PATTERN.match(line)
        if not match:
            return False
        try:
            percent = float(match.group(1))
        except ValueError:
            return False
        self._tracker.report_percent(percent)
        return True

    def write(self, message: str | bytes) -> int:
        if not message:
            return 0
        if isinstance(message, bytes):
            message = message.decode(self.encoding, errors=self.errors)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if self._maybe_report_progress(line):
                continue
            self._stream.write(line + "\n")
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line and not self._maybe_report_progress(line):
                self._stream.write(line)
            self._buffer = ""
        with suppress(Exception):
            self._stream.flush()

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        try:
            return self._stream.fileno()
        except Exception:
            return -1

    def writable(self) -> bool:
        return True


@contextmanager
def intercept_progress_lines(tracker: ProgressTracker):
    original_stdout = sys.stdout
    wrapper = ProgressLineInterceptor(original_stdout, tracker)
    sys.stdout = wrapper
    try:
        yield
    finally:
        sys.stdout = original_stdout
        wrapper.flush()


@contextmanager
def use_precomputed_chunks(chunks):
    """Force WhisperX to decode exactly ``chunks`` for one transcribe call.

    WhisperX groups pyannote VAD speech into ``<= chunk_size`` second bins via
    ``Pyannote.merge_chunks`` and decodes ``audio[start:end]`` for each. We pass
    it a silence-free *compacted* audio (voiced spans concatenated) plus the
    matching span boundaries in compacted time, so each inference call carries a
    full budget of real singing with the long silences removed; timestamps are
    mapped back to the real timeline afterwards (see ``transcribe_with_whisperx``).
    The pyannote VAD still runs on the compacted audio but its segmentation is
    discarded here.
    """
    from whisperx.vads.pyannote import Pyannote  # noqa: PLC0415

    original_merge_chunks = Pyannote.merge_chunks
    Pyannote.merge_chunks = staticmethod(lambda *a, **k: list(chunks))
    try:
        yield
    finally:
        Pyannote.merge_chunks = original_merge_chunks


def remap_segments_to_original(segments, mapping):
    """Map compacted-time segment and word timestamps back to the real timeline."""
    for segment in segments:
        if segment.get("start") is not None:
            segment["start"] = round(map_time(float(segment["start"]), mapping), 3)
        if segment.get("end") is not None:
            segment["end"] = round(map_time(float(segment["end"]), mapping), 3)
        for word in segment.get("words", []):
            if word.get("start") is not None:
                word["start"] = round(map_time(float(word["start"]), mapping), 3)
            if word.get("end") is not None:
                word["end"] = round(map_time(float(word["end"]), mapping), 3)
    return segments


# --- Collapse repair -------------------------------------------------------
#
# Whisper's language-model prior dislikes emitting a line verbatim twice. On a
# repeated lyric (a chorus, a chanted refrain) it can therefore *collapse*: it
# decodes the first pass and then stops, dropping the repeats and leaving a
# stretch of audible singing with no transcribed words. We detect those windows
# and rebuild them.
#
# Detection runs on the aligned timeline: a window is a collapse suspect when it
# is long and phonetically busy (many syllable onsets) yet word alignment covers
# almost none of it -- vocal energy with no words. Caption hallucinations
# (subtitle boilerplate the filter would delete, also leaving a hole) are flagged
# the same way.
#
# Repair re-decodes a flagged window with a *runway*: a slice of the nearest
# earlier stable audio is prepended so the decoder enters the problem region with
# momentum and does not collapse on the first repeat. The window is rebuilt from
# two such decodes -- one per half -- so each half gets a full runway and the
# combined slice stays within the 30s decode budget. A short silence pad sits
# between runway and payload (a hard concat makes Whisper bleed/EOT at the seam).
# The runway text is trimmed off afterwards by localized word-level alignment;
# the rebuilt halves replace the original only when they are real, diverse,
# non-looped, non-hallucinated content with more words than the collapse -- so a
# repair can never make a window worse than the plain decode.

_MIN_WINDOW_SECONDS = 12.0  # only long windows can hide a collapse
_MIN_ONSET_RATE = 2.0  # syllable onsets/s below this is a held note, not singing
_MAX_COVERED_FRACTION = 0.30  # aligned-word coverage below this fraction -> suspect
_MIN_UNCOVERED_SECONDS = 8.0  # ... with at least this many voiced-but-wordless seconds
_MIN_REPAIR_WORD_GAIN = 3  # a repair must add at least this many words
_MIN_UNIQUE_RATIO = 0.35  # below this the rebuild is a loop / vowel chant, not lyrics
# (true loops sit near 0.01-0.15; a repetitive but real chorus is ~0.45+)

_WINDOW_CAP_SECONDS = 30.0  # hard clamp on any decode slice (the model's budget)
_RUNWAY_SECONDS = 15.0  # stable lead-in taken from the nearest clean window
_RUNWAY_PAD_SECONDS = 1.5  # silence spliced between runway and payload
_RUNWAY_SNAP_SECONDS = 2.5  # snap the half-split point to a nearby silence
_MIN_HALF_SECONDS = 6.0  # each rebuilt half must span at least this
_MIN_CLEAN_PAYLOAD_SECONDS = 6.0  # a runway-source window needs this much singing
_MIN_RUNWAY_BUDGET_SECONDS = 4.0  # skip a repair if less runway than this fits

# Loop signatures used by :func:`_looks_looped`.
_LOOP_MIN_TOKENS = 4
_LOOP_RUN_LENGTH = 6  # this many identical tokens in a row is a decoder loop
_LOOP_DOMINANCE_MIN_TOKENS = 8
_LOOP_DOMINANCE_RATIO = 0.6  # one token covering this fraction of a line is a loop

# Spectral-flux onset detection (:func:`_syllable_onsets`).
_ONSET_BAND_HZ = (100, 4000)  # vocal band the flux is measured in
_ONSET_MIN_FRAMES = 5  # too few frames to judge an onset rate

# --- Loop guard -------------------------------------------------------------
#
# A chunk whose text compresses far better than natural language is a decoder
# loop ("nein, nein, nein, ..."), not a transcription. WhisperX's batched path
# never runs faster-whisper's temperature fallback, so a loop is final unless we
# re-decode it ourselves. Detection is whisper's own compression-ratio test;
# the retries add the repetition constraints CTranslate2 accepts through
# `generate`, which is what the browser runtime's guard does.
#
# The threshold is *not* whisper's own 2.4: that is calibrated on speech, and
# sung lyrics repeat by design. Measured per chunk on the leads corpus, a
# legitimate chorus reaches 2.9 while a real decoder loop sits at 4.3+. Firing
# in between re-decodes real choruses and measurably loses words.
_LOOP_COMPRESSION_RATIO = 3.5
_LOOP_MIN_BYTES = 48  # shorter texts compress unpredictably
_LOOP_GUARD_LADDER = (
    {"no_repeat_ngram_size": 3},
    {"no_repeat_ngram_size": 3, "repetition_penalty": 1.15},
)

# --- Halves -----------------------------------------------------------------
#
# Whisper can stop early in a window, or jump from its start to its end and
# leave the sung middle out. A long window whose aligned words stop early or
# leave a long gap is decoded again as two halves, and the halves replace it
# when they give more words. The browser runtime does the same with its own
# word times (packages/ai/src/runtime/whisper/whisperRuntime.ts).
_HALVES_MIN_SECONDS = 12.0  # only long windows are split
_HALVES_COVERAGE = 0.6  # words ending before this share of the window -> split
_HALVES_MAX_GAP_SECONDS = 6.0  # a gap this long between words -> split
_HALVES_MIN_GAIN = 3  # the halves must add at least this many words

# The lead with the backing at half level is what is transcribed, as in the
# app's host (packages/server/src/analysis/transcribe.rs): the separation often
# moves a sung line, a double or a duet into the backing.
BACKING_WEIGHT = 0.5


class Aligner(typing.NamedTuple):
    """The WhisperX alignment model bundled with the data its calls need."""

    model: object
    metadata: object
    device: str


class Compaction(typing.NamedTuple):
    """Outputs of the spectral compaction, passed together to the repair pass."""

    audio: object
    chunks: list
    packed_chunks: list
    mapping: list


def _word_count(text: str) -> int:
    return len((text or "").split())


def _unique_ratio(text: str) -> float:
    """Lexical diversity of the normalised text (lower-cased, punctuation gone).

    A recovered chorus is lexically diverse (~0.6+); a sustained-vowel chant or a
    one-phrase loop ("oh oh oh", "nein nein nein") collapses to a tiny ratio, so
    this cleanly rejects the rebuild's pathologies.
    """
    words = re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
    return len(set(words)) / len(words) if words else 1.0


def _looks_looped(text: str) -> bool:
    """True when the payload is dominated by a repeated token/phrase loop.

    ``_unique_ratio`` is a *global* diversity check, so a real prefix can mask a
    long tail loop: "the world in your hand ha ha ha ... ha" scores 6/16=0.38 and
    passes, yet the 12x "ha" is a decoder loop the rebuild injected. This catches
    such payloads two ways: a long run of one identical token, or one token that
    dominates a non-trivial payload -- both signatures of a decoder loop, neither
    fired by a genuinely repeated *lyric* (a word sung twice stays short and its
    token never dominates the line).
    """
    tokens = re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
    if len(tokens) < _LOOP_MIN_TOKENS:
        return False
    longest_run = run = 1
    for prev, cur in itertools.pairwise(tokens):
        run = run + 1 if cur == prev else 1
        longest_run = max(longest_run, run)
    if longest_run >= _LOOP_RUN_LENGTH:
        return True
    counts = collections.Counter(tokens)
    top = counts.most_common(1)[0][1]
    return (
        len(tokens) >= _LOOP_DOMINANCE_MIN_TOKENS
        and top / len(tokens) >= _LOOP_DOMINANCE_RATIO
    )


def _syllable_onsets(window_audio) -> int:
    """Count syllable/note onsets in a 16 kHz window via spectral flux.

    A collapse drops lyrics over audio that is *phonetically busy* -- many
    syllable onsets (rapid spectral change). A sustained vowel ("oh", "nein") or
    a quiet/instrumental stretch has an almost static spectrum, so very few
    onsets. Counting onsets therefore tells "there really are words here" apart
    from "this is genuinely a held note / near-silence", which word coverage
    alone cannot. Returns the number of detected onsets in the window.
    """
    import numpy as np  # noqa: PLC0415

    win, hop = 1024, 256
    if len(window_audio) < win * 2:
        return 0
    frame_count = (len(window_audio) - win) // hop + 1
    offsets = np.arange(win)[None, :] + hop * np.arange(frame_count)[:, None]
    window = np.hanning(win).astype(np.float32)
    frames = window_audio[offsets].astype(np.float32) * window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / SAMPLE_RATE)
    band = (freqs >= _ONSET_BAND_HZ[0]) & (freqs <= _ONSET_BAND_HZ[1])
    mag = mag[:, band]
    flux = np.maximum(0.0, mag[1:] - mag[:-1]).sum(axis=1)
    if flux.size < _ONSET_MIN_FRAMES or flux.max() <= 0:
        return 0
    flux = flux / flux.max()
    # Onset = local max above an adaptive floor, with a refractory gap so one
    # syllable is not counted twice (>= ~110 ms apart -> max ~9 onsets/s).
    floor = max(0.12, float(flux.mean() + 0.5 * flux.std()))
    gap = round(0.11 * SAMPLE_RATE / hop)
    onsets = 0
    last = -gap
    for i in range(1, len(flux) - 1):
        if (
            flux[i] >= floor
            and flux[i] >= flux[i - 1]
            and flux[i] > flux[i + 1]
            and i - last >= gap
        ):
            onsets += 1
            last = i
    return onsets


def _word_coverage(words, start, end) -> float:
    """Seconds of ``[start, end]`` overlapped by any aligned word interval."""
    covered = 0.0
    for word_start, word_end in words:
        lo, hi = max(start, word_start), min(end, word_end)
        if hi > lo:
            covered += hi - lo
    return covered


def _concat_pad(head, tail):
    """Splice ``head`` and ``tail`` with a short silence pad between them."""
    import numpy as np  # noqa: PLC0415

    pad = np.zeros(int(_RUNWAY_PAD_SECONDS * SAMPLE_RATE), dtype=head.dtype)
    return np.concatenate([head, pad, tail])


def _decode_span(model, window_audio, language, option_overrides=None) -> str:
    """Decode one audio window without timestamps; return its plain text.

    The window is decoded as a single chunk; the model's options are restored
    afterwards so this never leaks state into the main transcribe pass.
    ``option_overrides`` patches the decode options for this call only -- the
    loop guard uses it to add repetition constraints.
    """
    from dataclasses import replace  # noqa: PLC0415

    duration = len(window_audio) / SAMPLE_RATE
    one_chunk = [{"start": 0.0, "end": duration, "segments": [(0.0, duration)]}]
    original_options = model.options
    model.options = replace(
        original_options, without_timestamps=True, **(option_overrides or {})
    )
    try:
        with use_precomputed_chunks(one_chunk):
            result = model.transcribe(
                window_audio,
                batch_size=1,
                language=language,
                chunk_size=30,
                print_progress=False,
            )
    finally:
        model.options = original_options
    return " ".join(
        (segment.get("text") or "").strip() for segment in result.get("segments", [])
    ).strip()


def _payload_after_seam(aligner, audio_slice, text, seam):
    """Align ``text`` over ``audio_slice`` and return the words past ``seam``.

    A rebuilt slice begins with ``seam`` seconds of runway that earlier audio
    already transcribed. Localized alignment gives word-level times inside the
    slice, so the runway is dropped by a hard timestamp cut at ``seam`` and only
    the genuinely new payload text is kept. Returns ``(payload_text,
    payload_word_count)``; on any alignment failure returns ``("", 0)`` so the
    caller falls back to the safe original decode.
    """
    if not text.strip():
        return "", 0
    duration = len(audio_slice) / SAMPLE_RATE
    try:
        aligned = whisperx.align(
            [{"start": 0.0, "end": duration, "text": text}],
            aligner.model,
            aligner.metadata,
            audio_slice,
            aligner.device,
            return_char_alignments=False,
            print_progress=False,
        )
    except Exception as error:
        logging.warning("Seam alignment failed: %s", error)
        return "", 0
    words = [
        word
        for segment in aligned.get("segments", [])
        for word in segment.get("words", [])
        if float(word.get("start", 0.0)) >= seam
    ]
    payload_text = " ".join((w.get("word") or "").strip() for w in words).strip()
    return payload_text, _word_count(payload_text)


def plan_collapse_repairs(model, compaction, segments, aligned_segments, language):
    """Flag collapsed windows and decode each one's runway-fed halves.

    Detection uses the aligned timeline: a long, phonetically busy window whose
    aligned words cover too little of it is a collapse suspect, as is a caption
    hallucination. For each flagged window the split point is snapped to a silence
    near its centre and two slices are built -- ``runway ++ pad ++ head`` and
    ``runway ++ pad ++ tail`` -- where the runway is the tail of the nearest
    earlier *clean* window. Each slice is decoded here (ASR only); trimming and
    acceptance are deferred to :func:`apply_collapse_repairs` so the heavy
    alignment runs once, at the end. Returns one spec per repairable window.
    """
    audio, chunks = compaction.audio, compaction.chunks
    cut_points = sorted(
        {round(float(m[0]), 3) for m in compaction.mapping}
        | {round(float(m[1]), 3) for m in compaction.mapping}
    )

    def snap(target: float) -> float:
        near = [c for c in cut_points if abs(c - target) <= _RUNWAY_SNAP_SECONDS]
        return min(near, key=lambda c: abs(c - target)) if near else target

    def window_words(s: float, e: float) -> str:
        return " ".join(
            (seg.get("text") or "").strip()
            for seg in segments
            if s - 1e-3 <= float(seg.get("start", 0.0)) < e - 1e-3
        )

    aligned_words = [
        (float(w["start"]), float(w["end"]))
        for seg in aligned_segments
        for w in seg.get("words", [])
        if w.get("start") is not None and w.get("end") is not None
    ]
    payloads = [
        sum(end - start for start, end in pc) for pc in compaction.packed_chunks
    ]

    flagged: list[bool] = []
    window_text: list[str] = []
    for window, payload in zip(chunks, payloads, strict=False):
        s, e = float(window["start"]), float(window["end"])
        text = window_words(s, e)
        window_text.append(text)
        onsets = _syllable_onsets(audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)])
        cover = _word_coverage(aligned_words, s, e)
        collapse = (
            payload >= _MIN_WINDOW_SECONDS
            and onsets / payload >= _MIN_ONSET_RATE
            and cover / payload < _MAX_COVERED_FRACTION
            and payload - cover >= _MIN_UNCOVERED_SECONDS
        )
        flagged.append(collapse or _is_hallucination(text))

    cap = round(_WINDOW_CAP_SECONDS * SAMPLE_RATE)
    candidates: list[dict] = []
    for index, window in enumerate(chunks):
        if not flagged[index]:
            continue
        s, e = float(window["start"]), float(window["end"])
        # Runway = tail of the nearest earlier *non-flagged* window (clean audio);
        # without such a window the repair has no safe momentum, so skip it.
        prior = next(
            (
                j
                for j in range(index - 1, -1, -1)
                if not flagged[j] and payloads[j] >= _MIN_CLEAN_PAYLOAD_SECONDS
            ),
            None,
        )
        runway_anchor = float(chunks[prior]["end"]) if prior is not None else s
        if runway_anchor < _RUNWAY_SECONDS + 0.5:
            continue

        m = min(max(snap((s + e) / 2.0), s + _MIN_HALF_SECONDS), e - _MIN_HALF_SECONDS)
        budget = min(
            _RUNWAY_SECONDS,
            _WINDOW_CAP_SECONDS - _RUNWAY_PAD_SECONDS - (m - s),
            _WINDOW_CAP_SECONDS - _RUNWAY_PAD_SECONDS - (e - m),
        )
        if budget < _MIN_RUNWAY_BUDGET_SECONDS:
            continue
        r0 = max(0.0, runway_anchor - budget)
        runway = audio[int(r0 * SAMPLE_RATE) : int(runway_anchor * SAMPLE_RATE)]
        seam = (runway_anchor - r0) + _RUNWAY_PAD_SECONDS

        head = audio[int(s * SAMPLE_RATE) : int(m * SAMPLE_RATE)]
        tail = audio[int(m * SAMPLE_RATE) : int(e * SAMPLE_RATE)]
        slice1 = _concat_pad(runway, head)[:cap]
        slice2 = _concat_pad(runway, tail)[:cap]
        candidates.append(
            {
                "start": s,
                "end": e,
                "orig_words": _word_count(window_text[index]),
                "span1": (s, m),
                "slice1": slice1,
                "blob1": _decode_span(model, slice1, language),
                "seam1": seam,
                "span2": (m, e),
                "slice2": slice2,
                "blob2": _decode_span(model, slice2, language),
                "seam2": seam,
            }
        )
    return candidates


def apply_collapse_repairs(candidates, segments, aligner):
    """Trim each candidate's runway, keep sane halves, splice them in.

    A window is replaced only when the rebuilt payload is real, diverse,
    non-looped, non-hallucinated content with more words than the collapsed
    original, so the repair can never make a window worse than the plain decode.
    """
    for cand in candidates:
        halves = []
        total_words = 0
        for span, slice_audio, blob, seam in (
            (cand["span1"], cand["slice1"], cand["blob1"], cand["seam1"]),
            (cand["span2"], cand["slice2"], cand["blob2"], cand["seam2"]),
        ):
            text, words = _payload_after_seam(aligner, slice_audio, blob, seam)
            sane = (
                words >= _MIN_REPAIR_WORD_GAIN
                and _unique_ratio(text) >= _MIN_UNIQUE_RATIO
                and not _looks_looped(text)
                and not _is_hallucination(text)
            )
            if sane:
                halves.append({"start": span[0], "end": span[1], "text": text})
                total_words += words
        if not halves or total_words < cand["orig_words"] + _MIN_REPAIR_WORD_GAIN:
            continue
        start, end = cand["start"], cand["end"]
        segments = [
            seg
            for seg in segments
            if not (start - 1e-3 <= float(seg.get("start", 0.0)) < end - 1e-3)
        ]
        segments.extend(halves)
        logging.info(
            "Repaired collapsed window at %.1fs: %d -> %d words",
            cand["start"],
            cand["orig_words"],
            total_words,
        )
    segments.sort(key=lambda seg: float(seg.get("start", 0.0)))
    return segments


def _compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if len(data) < _LOOP_MIN_BYTES:
        return 0.0
    return len(data) / len(zlib.compress(data))


def _is_looped(text: str) -> bool:
    return _compression_ratio(text) > _LOOP_COMPRESSION_RATIO


def repair_looped_segments(model, audio, segments, language):
    """Re-decode every segment whose text is a decoder loop.

    Each retry walks :data:`_LOOP_GUARD_LADDER` until the text stops looping;
    a retry that still loops (or empties the window) is discarded, so the guard
    can never make a segment worse than the plain decode.
    """
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not _is_looped(text):
            continue
        start, end = float(segment["start"]), float(segment["end"])
        window = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        for guard in _LOOP_GUARD_LADDER:
            retry = _decode_span(model, window, language, guard)
            if not retry or _is_looped(retry):
                continue
            logging.info(
                "Repaired looped window at %.1fs: %d -> %d words",
                start,
                _word_count(text),
                _word_count(retry),
            )
            segment["text"] = retry
            break
    return segments


def _needs_halves(words, start: float, end: float) -> bool:
    """True when a window's aligned words stop early or leave a long gap."""
    duration = end - start
    if duration < _HALVES_MIN_SECONDS:
        return False
    inside = sorted(
        (word_start - start, word_end - start)
        for word_start, word_end in words
        if start - 1e-3 <= word_start < end - 1e-3
    )
    last_end = inside[-1][1] if inside else 0.0
    widest_gap = max(
        (later[0] - earlier[1] for earlier, later in itertools.pairwise(inside)),
        default=0.0,
    )
    return (
        last_end < _HALVES_COVERAGE * duration or widest_gap >= _HALVES_MAX_GAP_SECONDS
    )


def split_windows_in_halves(model, compaction, segments, aligned_segments, language):
    """Decode again in halves every window that stopped early or skipped a stretch.

    A half whose text loops or is a caption is dropped; the halves replace the
    window only when they give at least ``_HALVES_MIN_GAIN`` more words. Returns
    the segments and whether any window was replaced.
    """
    audio = compaction.audio
    aligned_words = [
        (float(w["start"]), float(w["end"]))
        for seg in aligned_segments
        for w in seg.get("words", [])
        if w.get("start") is not None and w.get("end") is not None
    ]
    replaced = False
    for window in compaction.chunks:
        start, end = float(window["start"]), float(window["end"])
        if not _needs_halves(aligned_words, start, end):
            continue
        inside = [
            seg
            for seg in segments
            if start - 1e-3 <= float(seg.get("start", 0.0)) < end - 1e-3
        ]
        before = _word_count(" ".join(seg.get("text") or "" for seg in inside))
        middle = start + (end - start) / 2.0
        halves = []
        for span_start, span_end in ((start, middle), (middle, end)):
            clip = audio[int(span_start * SAMPLE_RATE) : int(span_end * SAMPLE_RATE)]
            text = _decode_span(model, clip, language)
            if text and not _is_looped(text) and not _is_hallucination(text):
                halves.append({"start": span_start, "end": span_end, "text": text})
        after = _word_count(" ".join(half["text"] for half in halves))
        if after < before + _HALVES_MIN_GAIN:
            continue
        inside_ids = {id(seg) for seg in inside}
        segments = [seg for seg in segments if id(seg) not in inside_ids] + halves
        replaced = True
        logging.info(
            "Decoded the window at %.1fs in halves: %d -> %d words",
            start,
            before,
            after,
        )
    segments.sort(key=lambda seg: float(seg.get("start", 0.0)))
    return segments, replaced


def load_voice(audio_path: str, backing_path: str | None):
    """The lead, with the backing added at ``BACKING_WEIGHT`` when given."""
    audio = whisperx.load_audio(audio_path)
    if backing_path:
        backing = whisperx.load_audio(backing_path)
        length = min(len(audio), len(backing))
        audio[:length] += BACKING_WEIGHT * backing[:length]
    return audio


def transcribe_with_whisperx(
    audio_path: str,
    backing_path: str | None = None,
    log_level: str = "info",
):
    patch_speechbrain_lazy_imports()
    configure_torch_serialization()
    maybe_upgrade_whisperx_checkpoint()
    print_acceleration_info()
    setup_torch_optimization()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    with allow_unsafe_torch_load(), intercept_hf_downloads("WhisperX model"):
        model = whisperx.load_model(
            "large-v3-turbo",
            device,
            compute_type=compute_type,
            vad_method="pyannote",
            vad_options={
                # Chunk boundaries come from the spectral chunker (see
                # use_precomputed_chunks) and the language from the loudest
                # windows; WhisperX still runs this VAD, its output unused.
                "vad_onset": 0.45,
                "vad_offset": 0.35,
            },
        )
    audio = load_voice(audio_path, backing_path)

    # Pack the spectral voiced spans into full-budget chunks and concatenate them
    # into a silence-free buffer. Transcription and alignment both run on this
    # compacted audio; ``mapping`` carries every timestamp back to real time.
    packed_chunks = compute_packed_chunks(audio, chunk_size=30)
    compacted_audio, chunks, mapping = build_compaction(audio, packed_chunks)
    compaction = Compaction(compacted_audio, chunks, packed_chunks, mapping)
    detected_language = detect_language_loudest(
        model, compacted_audio, chunks, SAMPLE_RATE
    )

    progress_tracker = ProgressTracker()
    progress_tracker.report_fraction(0.0)

    # Pass 1: decode the whole compacted track in one sweep.
    with intercept_progress_lines(progress_tracker), use_precomputed_chunks(chunks):
        result = model.transcribe(
            compacted_audio,
            batch_size=1,
            language=detected_language,
            chunk_size=30,
            print_progress=True,
            combined_progress=True,
        )
        progress_tracker.ensure_minimum(0.5)
    segments = result.get("segments", [])
    detected_language = result.get("language", detected_language)
    segments = repair_looped_segments(
        model, compacted_audio, segments, detected_language
    )

    with (
        allow_unsafe_torch_load(),
        intercept_hf_downloads("WhisperX alignment model"),
    ):
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device
        )
    aligner = Aligner(align_model, metadata, device)

    def align_for_detection(current):
        return whisperx.align(
            [dict(seg) for seg in current],
            aligner.model,
            aligner.metadata,
            compacted_audio,
            aligner.device,
            return_char_alignments=False,
            print_progress=False,
        ).get("segments", [])

    # Pass 2: align the first decode to get word timestamps, decode again in
    # halves the windows that stopped early or skipped a stretch, flag windows
    # where the prior collapsed a repeated line (vocal energy, no words), and
    # rebuild those windows from runway-fed re-decodes.
    with intercept_progress_lines(progress_tracker):
        detect_aligned = align_for_detection(segments)
        segments, replaced = split_windows_in_halves(
            model, compaction, segments, detect_aligned, detected_language
        )
        if replaced:
            detect_aligned = align_for_detection(segments)
        candidates = plan_collapse_repairs(
            model, compaction, segments, detect_aligned, detected_language
        )
        if candidates:
            segments = apply_collapse_repairs(candidates, segments, aligner)

    segments = strip_credit_segments(segments)
    segments = filter_silent_segments(segments, compacted_audio)
    segments = filter_hallucinated_segments(segments)

    # Final alignment on the repaired transcript.
    try:
        with intercept_progress_lines(progress_tracker):
            aligned = whisperx.align(
                segments,
                aligner.model,
                aligner.metadata,
                compacted_audio,
                aligner.device,
                return_char_alignments=False,
                print_progress=True,
                combined_progress=True,
            )
        segments = aligned.get("segments", segments)
        detected_language = aligned.get("language", detected_language)
    except Exception as align_error:
        logging.warning("Alignment skipped: %s", align_error)
    finally:
        progress_tracker.finalize()

    segments = remap_segments_to_original(segments, mapping)
    return segments, detected_language
