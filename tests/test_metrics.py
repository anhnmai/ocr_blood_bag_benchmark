"""
tests/test_metrics.py
─────────────────────
Unit tests for CER/WER calculation and vocabulary constraint correction.
Run with:  pytest tests/
"""

import pytest
from src.metrics.calculate_cer_wer import cer, wer, evaluate_field
from src.ocr.vocab_constraint import constrain, is_valid, CHARSETS


# ── CER tests ─────────────────────────────────────────────────────────────────

class TestCER:
    def test_perfect_match(self):
        assert cer("W1234567890", "W1234567890") == 0.0

    def test_one_substitution(self):
        # "W123456789B" vs "W1234567890": 1 sub in 11 chars → CER ≈ 0.0909
        c = cer("W123456789B", "W1234567890", norm=False)
        assert abs(c - 1 / 11) < 1e-6

    def test_one_deletion(self):
        c = cer("W123456789", "W1234567890", norm=False)
        assert abs(c - 1 / 11) < 1e-6

    def test_one_insertion(self):
        c = cer("W12345678901", "W1234567890", norm=False)
        assert abs(c - 1 / 11) < 1e-6

    def test_empty_ref_nonempty_hyp(self):
        assert cer("hello", "") == 1.0

    def test_both_empty(self):
        assert cer("", "") == 0.0

    def test_date_common_confusion(self):
        # l vs 1
        c = cer("3l.12.2025", "31.12.2025")
        assert c > 0.0
        assert c < 0.2


# ── WER tests ─────────────────────────────────────────────────────────────────

class TestWER:
    def test_perfect_match(self):
        assert wer("A Rh pos", "A Rh pos") == 0.0

    def test_one_word_wrong(self):
        w = wer("A Rh p0s", "A Rh pos", norm=False)
        assert abs(w - 1 / 3) < 1e-6

    def test_single_token_field(self):
        # donation ID is typically a single token
        assert wer("W1234567890", "W1234567890") == 0.0


# ── evaluate_field tests ──────────────────────────────────────────────────────

class TestEvaluateField:
    def test_basic(self):
        preds = ["W1234567890", "A Rh pos", "31.12.2025"]
        refs  = ["W1234567890", "A Rh pos", "31.12.2025"]
        result = evaluate_field(preds, refs, field_name="donation_id")
        assert result["mean_cer"] == 0.0
        assert result["exact_match_rate"] == 1.0

    def test_partial_mismatch(self):
        preds = ["W1234567890", "W9999999999"]
        refs  = ["W1234567890", "W1234567890"]
        result = evaluate_field(preds, refs, field_name="donation_id")
        assert result["mean_cer"] > 0.0
        assert result["exact_match_rate"] == 0.5

    def test_none_prediction(self):
        result = evaluate_field([None], ["W1234567890"], field_name="x")
        assert result["n_failed_decode"] == 1
        assert result["mean_cer"] > 0.0

    def test_empty_ref_skipped(self):
        result = evaluate_field(["anything"], [None], field_name="x")
        assert result["n_empty_ref"] == 1
        assert result["n_samples"] == 0


# ── vocab constraint tests ────────────────────────────────────────────────────

class TestVocabConstraint:

    # blood_group
    def test_blood_group_perfect(self):
        assert constrain("A Rh pos", "blood_group") == "A Rh pos"

    def test_blood_group_zero_for_o(self):
        # "A Rh p0s" — digit zero instead of letter o
        result = constrain("A Rh p0s", "blood_group")
        assert result == "A Rh pos"

    def test_blood_group_typo(self):
        result = constrain("AB Rh neq", "blood_group")
        assert result == "AB Rh neg"

    def test_blood_group_too_far(self):
        # Completely garbled — should return original
        result = constrain("XXXXXXXXX", "blood_group", max_edit_distance=3)
        assert result == "XXXXXXXXX"

    # expiry_date
    def test_date_perfect(self):
        assert constrain("31.12.2025", "expiry_date") == "31.12.2025"

    def test_date_removes_non_digits(self):
        result = constrain("3l.12.2O25", "expiry_date")
        # digits extracted: 3, 1, 1, 2, 2, 0, 2, 5 → 31122025 → 31.12.2025
        assert result == "31.12.2025"

    def test_date_no_separator(self):
        assert constrain("31122025", "expiry_date") == "31.12.2025"

    # donation_id
    def test_donation_id_perfect(self):
        assert constrain("W1234567890", "donation_id") == "W1234567890"

    def test_donation_id_O_to_0(self):
        result = constrain("W123456789O", "donation_id")
        assert result == "W1234567890"

    # is_valid
    def test_is_valid_blood_group(self):
        assert is_valid("A Rh pos", "blood_group")
        assert not is_valid("A Rh p0s", "blood_group")

    def test_is_valid_date(self):
        assert is_valid("31.12.2025", "expiry_date")
        assert not is_valid("31122025", "expiry_date")

    def test_is_valid_donation_id(self):
        assert is_valid("W1234567890", "donation_id")
        assert not is_valid("1234567890A", "donation_id")  # digit first


# ── run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
