"""
tests/test_validator.py — Unit tests for the LLM Output Validator.
Run with: pytest
"""

import pytest
from llm_validator import LLMOutputValidator


@pytest.fixture
def v():
    return LLMOutputValidator()


# ─── PII Tests ────────────────────────────────────────────────────────────────

class TestPII:
    def test_clean_output_passes(self, v):
        result = v.validate("Neural networks learn from data.")
        assert result.passed

    def test_email_detected(self, v):
        result = v.validate("Contact us at user@example.com for help.")
        rules = [i.rule for i in result.issues]
        assert "pii_detected" in rules

    def test_ssn_detected(self, v):
        result = v.validate("Your SSN is 123-45-6789.")
        assert not result.passed

    def test_credit_card_detected(self, v):
        result = v.validate("Card: 4111111111111111")
        assert not result.passed

    def test_aws_key_detected(self, v):
        result = v.validate("Key: AKIAIOSFODNN7EXAMPLE")
        assert not result.passed


# ─── Toxicity Tests ───────────────────────────────────────────────────────────

class TestToxicity:
    def test_safe_output(self, v):
        result = v.validate("Python is a great programming language for beginners.")
        tox_issues = [i for i in result.issues if i.rule == "toxicity"]
        assert len(tox_issues) == 0

    def test_high_risk_term(self, v):
        result = v.validate("You should attack the server directly.")
        assert not result.passed

    def test_medium_risk_term(self, v):
        result = v.validate("The developer who wrote this is an idiot.")
        tox_issues = [i for i in result.issues if i.rule == "toxicity"]
        assert len(tox_issues) > 0


# ─── Topic Drift Tests ────────────────────────────────────────────────────────

class TestTopicDrift:
    def test_on_topic(self, v):
        result = v.validate(
            output="Python lists can be sorted using the sorted() function or list.sort().",
            prompt="How do I sort a Python list?"
        )
        drift_issues = [i for i in result.issues if i.rule == "topic_drift"]
        assert len(drift_issues) == 0

    def test_off_topic(self, v):
        result = v.validate(
            output="The French Revolution began in 1789 with the storming of the Bastille.",
            prompt="How do I sort a Python list?"
        )
        drift_issues = [i for i in result.issues if i.rule == "topic_drift"]
        assert len(drift_issues) > 0

    def test_no_prompt_skips_drift(self, v):
        result = v.validate(output="Something unrelated.", prompt="")
        drift_issues = [i for i in result.issues if i.rule == "topic_drift"]
        assert len(drift_issues) == 0


# ─── Structure Tests ──────────────────────────────────────────────────────────

class TestStructure:
    def test_very_short_output(self, v):
        result = v.validate("ok")
        struct_issues = [i for i in result.issues if i.rule == "output_length"]
        assert len(struct_issues) > 0

    def test_repetitive_output(self, v):
        text = " ".join(["banana"] * 50)
        result = v.validate(text)
        rep_issues = [i for i in result.issues if i.rule == "repetition"]
        assert len(rep_issues) > 0


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_pii_check_disabled(self):
        v = LLMOutputValidator(check_pii=False)
        result = v.validate("Email: user@example.com")
        pii_issues = [i for i in result.issues if i.rule == "pii_detected"]
        assert len(pii_issues) == 0

    def test_fail_on_high_only(self):
        v = LLMOutputValidator(fail_on=["high"])
        result = v.validate("The developer who wrote this is an idiot.")  # medium toxicity
        assert result.passed  # medium should not cause failure

    def test_score_increases_with_issues(self, v):
        clean = v.validate("This is a clean and safe response about Python.")
        risky = v.validate("SSN: 123-45-6789, email: user@example.com. You are an idiot.")
        assert risky.score > clean.score
