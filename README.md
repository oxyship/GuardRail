# LLM Output Validator

A Python library that validates AI-generated outputs for **PII leakage**, **toxic content**, and **topic drift** before they reach users.

Built as a reusable, composable validation layer you can drop into any LLM pipeline.

```
✅ PASSED  (risk score: 0.00)
❌ FAILED  (risk score: 0.60)
  [HIGH] pii_detected: Found email in output
  [HIGH] pii_detected: Found SSN in output
  [MEDIUM] toxicity: Medium-risk term detected: 'idiot'
```

---

## Why this exists

Most LLM tutorials focus on getting outputs *out* of a model — very few address what happens *after* the model responds. In production systems, unvalidated LLM outputs are a real risk:

- Models can regurgitate PII from training data or context windows
- Jailbroken or confused models can produce harmful content
- RAG pipelines can cause models to answer questions outside their intended scope

This project implements a **guardrail layer** — a validation step that sits between the model and the user.

**Threat model:** An attacker (or a confused model) attempts to make an LLM output sensitive data or harmful content. This validator detects and blocks that output before it surfaces.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-output-validator
cd llm-output-validator
pip install -r requirements.txt
python demo.py
```

**Run tests:**
```bash
pytest
```

---

## Usage

```python
from llm_validator import LLMOutputValidator

validator = LLMOutputValidator()

result = validator.validate(
    output="Here is your SSN: 123-45-6789 and email user@corp.com",
    prompt="Tell me about data privacy."
)

print(result.summary())
# ❌ FAILED (risk score: 0.80)
#   [HIGH] pii_detected: Found SSN in output
#   [HIGH] pii_detected: Found email in output
#   [MEDIUM] topic_drift: Output may have drifted from the original prompt topic

print(result.passed)   # False
print(result.score)    # 0.8
print(result.issues)   # list of ValidationIssue objects
```

### Configuration

```python
validator = LLMOutputValidator(
    check_pii=True,            # Detect emails, SSNs, credit cards, API keys
    check_toxicity=True,       # Flag harmful or abusive language
    check_drift=True,          # Catch off-topic responses
    check_structure=True,      # Detect empty, truncated, or looping outputs
    drift_threshold=0.15,      # Lower = stricter topic enforcement
    fail_on=["high", "medium"] # Which severities mark a result as failed
)
```

### Disable specific checks

```python
# Only check PII — skip everything else
validator = LLMOutputValidator(
    check_toxicity=False,
    check_drift=False,
    check_structure=False
)
```

---

## What it detects

| Check | What it catches | Severity |
|---|---|---|
| **PII — email** | `user@example.com` patterns | High |
| **PII — SSN** | `123-45-6789` patterns | High |
| **PII — credit card** | Visa/MC/Amex numbers | High |
| **PII — phone** | US phone numbers | Medium |
| **PII — IP address** | IPv4 addresses | Medium |
| **PII — AWS key** | `AKIA...` IAM key patterns | High |
| **Toxicity — high** | Violence, security attack terms | High |
| **Toxicity — medium** | Insults, hate language | Medium |
| **Topic drift** | Off-topic responses vs. the original prompt | Medium |
| **Repetition** | Model loops / stuck outputs | Medium |
| **Length** | Suspiciously short or truncated outputs | Low |

---

## Architecture

```
LLMOutputValidator
├── PIIDetector          ← regex patterns, easily swappable with Presidio
├── ToxicityChecker      ← keyword tiers, swappable with Detoxify (ML)
├── TopicDriftChecker    ← keyword overlap, swappable with sentence-transformers
└── StructureChecker     ← length + repetition heuristics
```

Each checker is independent. You can replace any layer with a more powerful implementation without touching the others.

---

## Upgrading to production-grade components

The baseline uses regex and keyword matching — no ML dependencies. When you're ready to go further:

### PII: Microsoft Presidio
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

Replace `PIIDetector.check()` with:
```python
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
results = analyzer.analyze(text=output, language="en")
```

### Toxicity: Detoxify
```bash
pip install detoxify
```

Replace `ToxicityChecker.check()` with:
```python
from detoxify import Detoxify
scores = Detoxify('original').predict(output)
# scores = {'toxicity': 0.92, 'severe_toxicity': 0.45, ...}
```

### Topic drift: Sentence Transformers
```bash
pip install sentence-transformers
```

Replace `TopicDriftChecker.check()` with:
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
similarity = float(util.cos_sim(model.encode(prompt), model.encode(output)))
```

---

## Wrap it as an API (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from llm_validator import LLMOutputValidator

app = FastAPI()
validator = LLMOutputValidator()

class ValidateRequest(BaseModel):
    output: str
    prompt: str = ""

@app.post("/validate")
def validate(req: ValidateRequest):
    result = validator.validate(output=req.output, prompt=req.prompt)
    return {
        "passed": result.passed,
        "score": result.score,
        "issues": [
            {"rule": i.rule, "severity": i.severity, "message": i.message}
            for i in result.issues
        ]
    }
```

```bash
pip install fastapi uvicorn
uvicorn api:app --reload
# POST http://localhost:8000/validate
```

---

## Project structure

```
llm-output-validator/
├── llm_validator/
│   ├── __init__.py
│   └── validator.py      ← core logic: all checkers live here
├── tests/
│   └── test_validator.py ← pytest unit tests
├── demo.py               ← run this to see it in action
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Presidio integration for production PII detection
- [ ] Detoxify integration for ML-based toxicity scoring
- [ ] Sentence-transformer semantic drift detection
- [ ] FastAPI wrapper for use as a microservice
- [ ] Configurable allow-lists (e.g. allow internal IP ranges)
- [ ] JSON output mode for logging pipelines
- [ ] Pre-commit hook integration

---

## Related work / references

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM06 covers sensitive information disclosure
- [Microsoft Presidio](https://github.com/microsoft/presidio) — production PII detection
- [Detoxify](https://github.com/unitaryai/detoxify) — ML toxicity classification
- [Guardrails AI](https://github.com/guardrails-ai/guardrails) — broader LLM guardrails framework

---

## License

MIT
