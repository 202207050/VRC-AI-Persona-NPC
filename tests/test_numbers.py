"""Thousand separators in NPC lines (persona_ai.format_numbers). Run: .venv\\Scripts\\python.exe tests\\test_numbers.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_ai import format_numbers  # noqa: E402

CASES = [
    ("50 × 40 = 2000이야!", "50 × 40 = 2,000이야!"),
    ("12500원이면 돼.", "12,500원이면 돼."),
    ("1234567", "1,234,567"),
    ("이미 2,000이야.", "이미 2,000이야."),         # already formatted
    ("2026년 겨울이야.", "2026년 겨울이야."),        # years stay
    ("3.14159는 원주율.", "3.14159는 원주율."),      # decimals stay
    ("999개", "999개"),                              # under 1,000
    ("3시 30분", "3시 30분"),
]

for given, want in CASES:
    got = format_numbers(given)
    assert got == want, f"{given!r} -> {got!r}, want {want!r}"
print(f"OK: {len(CASES)} checks passed")


def test_number_cases():
    for given, want in CASES:
        assert format_numbers(given) == want
