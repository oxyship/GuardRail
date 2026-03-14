"""
demo.py — Run this to see the LLM Output Validator in action.
No API key needed. Uses simulated LLM responses.
"""

from llm_validator import LLMOutputValidator

validator = LLMOutputValidator()

# ─── Test Cases ───────────────────────────────────────────────────────────────

test_cases = [
    {
        "name": "Clean response",
        "prompt": "Explain what a neural network is.",
        "output": (
            "A neural network is a machine learning model inspired by the human brain. "
            "It consists of layers of interconnected nodes (neurons) that process input data, "
            "learn patterns, and produce predictions. Each connection has a weight that is "
            "adjusted during training to minimize prediction errors."
        ),
    },
    {
        "name": "PII leakage — email",
        "prompt": "Summarize the customer complaint.",
        "output": (
            "The customer john.smith@example.com called to report an issue with their order. "
            "They were frustrated and asked for a refund to be processed immediately."
        ),
    },
    {
        "name": "PII leakage — SSN + credit card",
        "prompt": "Fill out the form for the user.",
        "output": (
            "User details: SSN 123-45-6789, card number 4111111111111111, "
            "phone (555) 867-5309."
        ),
    },
    {
        "name": "Toxic content",
        "prompt": "How do I improve website performance?",
        "output": (
            "You should kill all unnecessary background processes. "
            "Also, any developer who wrote this code is an idiot."
        ),
    },
    {
        "name": "Topic drift",
        "prompt": "Write a Python function to sort a list.",
        "output": (
            "The French Revolution began in 1789 with the storming of the Bastille. "
            "Napoleon later rose to power and conquered much of Europe before his exile."
        ),
    },
    {
        "name": "Repetitive output (model loop)",
        "prompt": "Tell me something interesting.",
        "output": "The cat sat sat sat sat sat sat sat sat sat sat sat sat sat sat sat sat sat.",
    },
    {
        "name": "AWS key leakage",
        "prompt": "Show me how to configure AWS.",
        "output": (
            "Set your access key: AKIAIOSFODNN7EXAMPLE. "
            "Then run aws configure and paste it in."
        ),
    },
]

# ─── Run and Print ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  LLM Output Validator — Demo")
print("=" * 60)

for case in test_cases:
    print(f"\n📋 Test: {case['name']}")
    print(f"   Prompt: \"{case['prompt'][:60]}\"")

    result = validator.validate(output=case["output"], prompt=case["prompt"])
    print(f"   {result.summary()}")

print("\n" + "=" * 60)
print("Demo complete. Check README.md for next steps.")
