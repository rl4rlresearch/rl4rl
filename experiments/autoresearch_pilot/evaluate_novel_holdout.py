#!/usr/bin/env python3
"""Evaluate an AdderBoard submission on a disjoint random holdout set.

The holdout excludes every exact ordered pair used by the official 2025
verifier. It also excludes either operand appearing anywhere in that verifier,
which is a stricter disjointness condition than exact-pair exclusion alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
from types import ModuleType

MAX_OPERAND = 9_999_999_999
OFFICIAL_EDGE_CASES = (
    (0, 0),
    (0, 1),
    (MAX_OPERAND, 0),
    (MAX_OPERAND, 1),
    (MAX_OPERAND, MAX_OPERAND),
    (5_000_000_000, 5_000_000_000),
    (1_111_111_111, 8_888_888_889),
    (1_234_567_890, 9_876_543_210),
    (MAX_OPERAND, MAX_OPERAND),
    (1, MAX_OPERAND),
)


def official_cases(num_tests: int = 10_000, seed: int = 2025) -> list[tuple[int, int]]:
    """Reproduce the public verifier's ordered input pairs exactly."""
    rng = random.Random(seed)
    return [*OFFICIAL_EDGE_CASES, *[
        (rng.randint(0, MAX_OPERAND), rng.randint(0, MAX_OPERAND))
        for _ in range(num_tests)
    ]]


def novel_cases(
    num_tests: int, seed: int, official: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int]:
    """Generate unique random pairs sharing neither official pairs nor operands."""
    official_pairs = set(official)
    official_operands = {value for pair in official_pairs for value in pair}
    generated: list[tuple[int, int]] = []
    generated_pairs: set[tuple[int, int]] = set()
    rejected = 0
    rng = random.Random(seed)
    while len(generated) < num_tests:
        pair = (rng.randint(0, MAX_OPERAND), rng.randint(0, MAX_OPERAND))
        if (
            pair in official_pairs
            or pair in generated_pairs
            or pair[0] in official_operands
            or pair[1] in official_operands
        ):
            rejected += 1
            continue
        generated.append(pair)
        generated_pairs.add(pair)
    return generated, rejected


def load_submission(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("novel_holdout_submission", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to import submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases_sha256(cases: list[tuple[int, int]]) -> str:
    encoded = "".join(f"{a}\t{b}\n" for a, b in cases).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate(submission: Path, num_tests: int, seed: int) -> dict[str, object]:
    official = official_cases()
    holdout, rejected_draws = novel_cases(num_tests, seed, official)
    official_pairs = set(official)
    official_operands = {value for pair in official_pairs for value in pair}
    assert not (set(holdout) & official_pairs)
    assert not ({value for pair in holdout for value in pair} & official_operands)

    module = load_submission(submission)
    model, metadata = module.build_model()
    passed = 0
    failures: list[dict[str, object]] = []
    start = time.monotonic()
    for index, (a, b) in enumerate(holdout, start=1):
        expected = a + b
        try:
            actual = module.add(model, a, b)
        except Exception as error:  # Preserve an unexpected model failure as evidence.
            actual = f"ERROR: {error}"
        if actual == expected:
            passed += 1
        elif len(failures) < 20:
            failures.append({"a": a, "b": b, "expected": expected, "actual": actual})
        if index % 1_000 == 0:
            print(f"  Progress: {index}/{num_tests} ({passed}/{index} correct)")
    elapsed_seconds = time.monotonic() - start
    return {
        "schema": "rl4rl-adderboard-novel-holdout-v1",
        "submission": str(submission.resolve()),
        "holdout_seed": seed,
        "holdout_cases": num_tests,
        "holdout_cases_sha256": cases_sha256(holdout),
        "official_cases_excluded": len(official),
        "official_pair_overlap": len(set(holdout) & official_pairs),
        "official_operand_overlap": len(
            {value for pair in holdout for value in pair} & official_operands
        ),
        "rejected_random_draws": rejected_draws,
        "passed": passed,
        "failed": num_tests - passed,
        "accuracy_percent": 100 * passed / num_tests,
        "elapsed_seconds": elapsed_seconds,
        "metadata": metadata,
        "first_failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--num-tests", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_815)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_tests < 1:
        raise SystemExit("--num-tests must be positive")

    result = evaluate(args.submission, args.num_tests, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print()
    print(f"Novel-holdout accuracy: {result['accuracy_percent']:.4f}%")
    print(f"Exact official-pair overlap: {result['official_pair_overlap']}")
    print(f"Official-operand overlap: {result['official_operand_overlap']}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
