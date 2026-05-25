"""
calculate_cer_wer.py
────────────────────
Compute Character Error Rate (CER) and Word Error Rate (WER) between
OCR predictions and ground-truth strings.

CER is the normalised Levenshtein edit distance at character level [14]:
    CER = (S + D + I) / N
where S = substitutions, D = deletions, I = insertions, N = reference length.

WER is the same metric applied at word (whitespace-delimited token) level.

Both are bounded [0, ∞) — values > 1.0 are possible when insertions dominate.

References
----------
[14] V. I. Levenshtein, "Binary codes capable of correcting deletions,
     insertions and reversals," Soviet Physics Doklady, vol. 10, no. 8,
     pp. 707–710, 1966.
"""

from __future__ import annotations

import re
import editdistance


# ── normalisation ─────────────────────────────────────────────────────────────

def normalise(text: str, *, lowercase: bool = True, strip_spaces: bool = False) -> str:
    """
    Light normalisation for German blood-bag label fields.
    - Strip leading/trailing whitespace.
    - Collapse runs of internal whitespace to a single space.
    - Optionally lowercase (useful for non-structured text; turn off for IDs).
    - Optionally remove all spaces (useful for fixed-format identifiers).
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lowercase:
        text = text.lower()
    if strip_spaces:
        text = text.replace(" ", "")
    return text


# ── CER ───────────────────────────────────────────────────────────────────────

def cer(hypothesis: str, reference: str, *, norm: bool = True) -> float:
    """
    Character Error Rate.

    Parameters
    ----------
    hypothesis : OCR prediction
    reference  : ground truth
    norm       : apply normalise() before comparison

    Returns
    -------
    CER as a float in [0, ∞). Returns 0.0 if both strings are empty.
    Returns 1.0 if reference is empty but hypothesis is not.
    """
    if norm:
        hypothesis = normalise(hypothesis)
        reference = normalise(reference)

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    dist = editdistance.eval(hypothesis, reference)
    return dist / len(reference)


# ── WER ───────────────────────────────────────────────────────────────────────

def wer(hypothesis: str, reference: str, *, norm: bool = True) -> float:
    """
    Word Error Rate.

    Parameters
    ----------
    hypothesis : OCR prediction
    reference  : ground truth

    Returns
    -------
    WER as a float in [0, ∞).
    """
    if norm:
        hypothesis = normalise(hypothesis)
        reference = normalise(reference)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = editdistance.eval(hyp_words, ref_words)
    return dist / len(ref_words)


# ── Field-level evaluation ────────────────────────────────────────────────────

def evaluate_field(
    predictions: list[str],
    references: list[str],
    *,
    field_name: str = "field",
    norm: bool = True,
) -> dict:
    """
    Compute aggregate CER and WER for a list of (prediction, reference) pairs.

    Returns
    -------
    dict with keys:
        field, n_samples, mean_cer, mean_wer,
        exact_match_rate, n_empty_ref, n_failed_decode
    """
    assert len(predictions) == len(references), "Length mismatch"

    cer_scores, wer_scores, exact_matches = [], [], []
    n_empty_ref = 0
    n_failed_decode = 0

    for hyp, ref in zip(predictions, references):
        if ref is None or ref == "":
            n_empty_ref += 1
            continue
        if hyp is None:
            n_failed_decode += 1
            hyp = ""

        c = cer(hyp, ref, norm=norm)
        w = wer(hyp, ref, norm=norm)
        cer_scores.append(c)
        wer_scores.append(w)

        ref_n = normalise(ref) if norm else ref
        hyp_n = normalise(hyp) if norm else hyp
        exact_matches.append(ref_n == hyp_n)

    n = len(cer_scores)
    return {
        "field": field_name,
        "n_samples": n,
        "mean_cer": sum(cer_scores) / n if n else None,
        "mean_wer": sum(wer_scores) / n if n else None,
        "exact_match_rate": sum(exact_matches) / n if n else None,
        "n_empty_ref": n_empty_ref,
        "n_failed_decode": n_failed_decode,
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("W1234567890", "W1234567890", "perfect match"),
        ("W1234567B90", "W1234567890", "1 substitution (8→B)"),
        ("W123456789",  "W1234567890", "1 deletion"),
        ("W12345678900","W1234567890", "1 insertion"),
        ("31.12.2025",  "31.12.2025",  "date perfect"),
        ("3l.12.2025",  "31.12.2025",  "date: l vs 1"),
        ("A Rh pos",    "A Rh pos",    "blood group perfect"),
        ("A Rh p0s",    "A Rh pos",    "blood group: 0 vs o"),
    ]

    print(f"{'Description':<35} {'CER':>6}  {'WER':>6}")
    print("-" * 52)
    for hyp, ref, desc in cases:
        print(f"{desc:<35} {cer(hyp, ref):>6.3f}  {wer(hyp, ref):>6.3f}")
