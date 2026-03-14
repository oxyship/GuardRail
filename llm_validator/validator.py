"""
LLM Output Validator
Core validation module - checks AI-generated outputs for PII, toxicity, and topic drift.
"""

from dataclasses import dataclass, field
from typing import Optional
import re


# ─── Result Types ────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    rule: str
    severity: str          # "high", "medium", "low"
    message: str
    detail: Optional[str] = None


@dataclass
class ValidationResult:
    passed: bool
    score: float           # 0.0 (clean) → 1.0 (very risky)
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed:
            return f"✅ PASSED (risk score: {self.score:.2f})"
        lines = [f"❌ FAILED (risk score: {self.score:.2f})"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.rule}: {issue.message}")
        return "\n".join(lines)


# ─── PII Detector ─────────────────────────────────────────────────────────────

class PIIDetector:
    """
    Detects common PII patterns using regex.
    For production, swap or augment with Microsoft Presidio.
    """

    PATTERNS = {
        "email": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "high"
        ),
        "ssn": (
            r"\b\d{3}-\d{2}-\d{4}\b",
            "high"
        ),
        "credit_card": (
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
            "high"
        ),
        "phone_us": (
            r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "medium"
        ),
        "ip_address": (
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "medium"
        ),
        "aws_key": (
            r"\bAKIA[0-9A-Z]{16}\b",
            "high"
        ),
    }

    def check(self, text: str) -> list[ValidationIssue]:
        issues = []
        for name, (pattern, severity) in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                # Redact for safe logging
                sample = matches[0]
                if name in ("ssn", "credit_card"):
                    sample = "***REDACTED***"
                issues.append(ValidationIssue(
                    rule="pii_detected",
                    severity=severity,
                    message=f"Found {name.replace('_', ' ')} in output",
                    detail=f"First match (redacted): {sample[:6]}..."
                ))
        return issues


# ─── Toxicity Checker ─────────────────────────────────────────────────────────

class ToxicityChecker:
    """
    Lightweight keyword-based toxicity checker.
    For production, replace with Detoxify: pip install detoxify
    and call: Detoxify('original').predict(text)
    """

    # Tiered word lists — extend these in production
    HIGH_RISK = [
        "kill", "murder", "attack", "bomb", "exploit",
        "hack into", "ddos", "ransomware",
    ]
    MEDIUM_RISK = [
        "idiot", "moron", "stupid", "dumb", "hate",
    ]

    def check(self, text: str) -> list[ValidationIssue]:
        issues = []
        lower = text.lower()

        for word in self.HIGH_RISK:
            if word in lower:
                issues.append(ValidationIssue(
                    rule="toxicity",
                    severity="high",
                    message=f"High-risk term detected: '{word}'",
                ))
                break  # One high-risk hit is enough to flag

        for word in self.MEDIUM_RISK:
            if word in lower:
                issues.append(ValidationIssue(
                    rule="toxicity",
                    severity="medium",
                    message=f"Medium-risk term detected: '{word}'",
                ))
                break

        return issues


# ─── Topic Drift Checker ──────────────────────────────────────────────────────

class TopicDriftChecker:
    """
    Checks whether the LLM output stays on topic relative to the original prompt.
    Uses keyword overlap as a lightweight proxy.
    For production, use sentence-transformers:
      from sentence_transformers import SentenceTransformer, util
      model = SentenceTransformer('all-MiniLM-L6-v2')
      similarity = util.cos_sim(model.encode(prompt), model.encode(output))
    """

    STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "and", "or",
        "to", "of", "for", "with", "this", "that", "be", "are",
        "was", "were", "i", "you", "he", "she", "we", "they",
    }

    def _keywords(self, text: str) -> set:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        return {w for w in words if w not in self.STOPWORDS}

    def check(self, prompt: str, output: str, threshold: float = 0.15) -> list[ValidationIssue]:
        if not prompt:
            return []

        prompt_kw = self._keywords(prompt)
        output_kw = self._keywords(output)

        if not prompt_kw or not output_kw:
            return []

        overlap = len(prompt_kw & output_kw)
        similarity = overlap / len(prompt_kw | output_kw)

        if similarity < threshold:
            return [ValidationIssue(
                rule="topic_drift",
                severity="medium",
                message="Output may have drifted from the original prompt topic",
                detail=f"Keyword overlap score: {similarity:.2f} (threshold: {threshold})"
            )]
        return []


# ─── Length / Coherence Checks ────────────────────────────────────────────────

class StructureChecker:
    """Basic checks for output length and coherence."""

    def check(self, output: str, min_length: int = 10, max_length: int = 8000) -> list[ValidationIssue]:
        issues = []
        length = len(output.strip())

        if length < min_length:
            issues.append(ValidationIssue(
                rule="output_length",
                severity="low",
                message=f"Output is suspiciously short ({length} chars)",
            ))

        if length > max_length:
            issues.append(ValidationIssue(
                rule="output_length",
                severity="low",
                message=f"Output exceeds max length ({length} chars)",
            ))

        # Check for runaway repetition
        words = output.lower().split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:
                issues.append(ValidationIssue(
                    rule="repetition",
                    severity="medium",
                    message="Output contains excessive repetition",
                    detail=f"Unique word ratio: {unique_ratio:.2f}"
                ))

        return issues


# ─── Main Validator ────────────────────────────────────────────────────────────

class LLMOutputValidator:
    """
    Main validator. Compose rules and call validate().

    Usage:
        validator = LLMOutputValidator()
        result = validator.validate(output="...", prompt="...")
        print(result.summary())
    """

    SEVERITY_SCORE = {"high": 0.4, "medium": 0.2, "low": 0.05}

    def __init__(
        self,
        check_pii: bool = True,
        check_toxicity: bool = True,
        check_drift: bool = True,
        check_structure: bool = True,
        drift_threshold: float = 0.10,
        fail_on: list[str] = None,   # severities that cause passed=False
    ):
        self.check_pii = check_pii
        self.check_toxicity = check_toxicity
        self.check_drift = check_drift
        self.check_structure = check_structure
        self.drift_threshold = drift_threshold
        self.fail_on = fail_on or ["high", "medium"]

        self._pii = PIIDetector()
        self._tox = ToxicityChecker()
        self._drift = TopicDriftChecker()
        self._struct = StructureChecker()

    def validate(self, output: str, prompt: str = "") -> ValidationResult:
        """Run all enabled checks and return a ValidationResult."""
        all_issues: list[ValidationIssue] = []

        if self.check_pii:
            all_issues.extend(self._pii.check(output))

        if self.check_toxicity:
            all_issues.extend(self._tox.check(output))

        if self.check_drift:
            all_issues.extend(self._drift.check(prompt, output, self.drift_threshold))

        if self.check_structure:
            all_issues.extend(self._struct.check(output))

        # Calculate risk score (capped at 1.0)
        score = min(1.0, sum(self.SEVERITY_SCORE.get(i.severity, 0) for i in all_issues))

        # Determine pass/fail
        failing_severities = {i.severity for i in all_issues}
        passed = not any(s in failing_severities for s in self.fail_on)

        return ValidationResult(
            passed=passed,
            score=round(score, 2),
            issues=all_issues,
            metadata={
                "output_length": len(output),
                "checks_run": [
                    c for c, enabled in [
                        ("pii", self.check_pii),
                        ("toxicity", self.check_toxicity),
                        ("topic_drift", self.check_drift),
                        ("structure", self.check_structure),
                    ] if enabled
                ]
            }
        )
