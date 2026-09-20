"""Compare explicit local WAV-to-text benchmark receipts, including quality.

This is a single-request, greedy ASR comparison, not full WhisperX acceptance.
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def words(text):
    return re.findall(r"\w+", text.casefold())


def edits(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        next_row = [i]
        for j, actual in enumerate(hypothesis, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1,
                                row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def group(receipt, native):
    groups = {}
    for run in receipt["runs"]:
        name = Path(run["file"]).stem if native else run["id"]
        groups.setdefault(name, []).append(run)
    return groups


def p95(values):
    ordered = sorted(values)
    return ordered[max(0, __import__("math").ceil(len(ordered) * 0.95) - 1)]


p = argparse.ArgumentParser()
p.add_argument("manifest", type=Path)
p.add_argument("native", type=Path)
p.add_argument("reference", type=Path)
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
native = json.loads(a.native.read_text())
reference = json.loads(a.reference.read_text())
assert native.get("suppression") == reference.get("suppression"), "token-suppression policy mismatch"
ng, rg = group(native, True), group(reference, False)
assert set(ng) == set(rg) == {i["id"] for i in manifest["items"]}, "input set mismatch"
rows = []
native_edits = reference_edits = total_words = 0
scored_clips = 0
scored_seconds = 0
for item in manifest["items"]:
    name = item["id"]
    nr, rr = ng[name], rg[name]
    assert len(nr) > 1 and len(rr) > 1, "need cold plus warm repetitions"
    nw = [x["total_ms"] for x in nr if x["index"] > 0]
    rw = [x["total_ms"] for x in rr if x["index"] > 0]
    nt, rt = nr[-1]["result"]["text"], rr[-1]["text"]
    # Null means unavailable ground truth, not silence. Keep these inputs in
    # timing/parity/stability results, but exclude them from word-error scores.
    # An explicit empty string remains known silence and counts insertions.
    expected = words(item["reference"]) if item.get("reference") is not None else None
    nwords, rwords = words(nt), words(rt)
    ne = edits(expected, nwords) if expected is not None else None
    re_ = edits(expected, rwords) if expected is not None else None
    if expected is not None:
        native_edits += ne
        reference_edits += re_
        total_words += len(expected)
        scored_clips += 1
        scored_seconds += item["seconds"]
    rows.append(dict(id=name, seconds=item["seconds"], reference=item["reference"],
        native_text=nt, comparison_text=rt, normalized_text_match=nwords == rwords,
        token_match=(nr[-1]["result"]["tokens"] == rr[-1]["tokens"]) if rr[-1].get("tokens") is not None else None,
        native_all_ended=all(x["result"]["ended"] for x in nr),
        native_stable=all(x["result"]["tokens"] == nr[-1]["result"]["tokens"] for x in nr),
        reference_stable=all(x["text"] == rt and x.get("tokens") == rr[-1].get("tokens") for x in rr),
        reference_scorable=expected is not None,
        reference_words=len(expected) if expected is not None else None,
        native_word_edits=ne, reference_word_edits=re_,
        native_warm_median_ms=statistics.median(nw),
        reference_warm_median_ms=statistics.median(rw),
        native_warm_p95_ms=p95(nw), reference_warm_p95_ms=p95(rw)))
n_total = sum(x["native_warm_median_ms"] for x in rows)
r_total = sum(x["reference_warm_median_ms"] for x in rows)
print(json.dumps(dict(
    scope="WAV-to-text ASR only; greedy English, batch one; no VAD/alignment/diarization",
    timing_note="Resident latency sums per-clip medians, excluding each clip's first repetition. Fresh-process first result includes imports/model initialization; OS file cache is not cleared.",
    quality_note="Word edit rates ignore case and punctuation; explicit silence insertions are included. Null reference text is unavailable and excluded only from error scores, not parity or timing.",
    clips=len(rows), audio_seconds=sum(x["seconds"] for x in rows),
    native_backend=native["backend"], reference_backend=reference["backend"],
    native_precision=native["precision"], reference_precision=reference["compute_type"],
    native_first_result_ms=native["first_result_ms"], reference_first_result_ms=reference["first_result_ms"],
    native_load_ms=native["load_ms"], reference_load_ms=reference["load_ms"],
    native_warm_corpus_ms=n_total, reference_warm_corpus_ms=r_total, warm_speedup=r_total/n_total,
    normalized_text_matches=sum(x["normalized_text_match"] for x in rows),
    exact_token_matches=sum(x["token_match"] for x in rows) if all(x["token_match"] is not None for x in rows) else None,
    reference_tokens_available=all(x["token_match"] is not None for x in rows),
    suppression=native.get("suppression"),
    reference_scored_clips=scored_clips, reference_scored_audio_seconds=scored_seconds,
    reference_words=total_words,
    native_word_edit_rate=native_edits / total_words if total_words else None,
    reference_word_edit_rate=reference_edits / total_words if total_words else None,
    all_native_requests_ended=all(x["native_all_ended"] for x in rows),
    all_native_outputs_stable=all(x["native_stable"] for x in rows),
    all_reference_outputs_stable=all(x["reference_stable"] for x in rows),
    full_goal_acceptance=False, rows=rows), indent=2))
