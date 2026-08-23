#!/usr/bin/env python3
"""Trusted task wrappers that emit the shared Layer A JSON contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ACCURACY = re.compile(r"Results: (\d+)/(\d+) correct \(([0-9.]+)%\)")
PARAMETERS = re.compile(r"Parameters \(unique\):\s*(\d+)")
TRAINING_STEP = re.compile(r"\bstep(?:\s*=\s*|\s+)(\d+)\b")
FORBIDDEN_MODEL_PATTERNS = {
    "carry-specific model logic": re.compile(r"\bcarry\b", re.IGNORECASE),
    "finite-state arithmetic": re.compile(
        r"finite[-_ ]state|transducer", re.IGNORECASE
    ),
    "zero-length parameter anchor": re.compile(r"parameter_anchor", re.IGNORECASE),
    "fixed digit scatter": re.compile(r"scatter_\s*\("),
    "modulo-ten answer rule": re.compile(r"remainder\s*\(\s*10\s*\)|%\s*10\b"),
    "operand-index answer rule": re.compile(
        r"\b(previous_result|a_digit|b_digit|next_digit|ANSWER_DIGITS|INPUT_LENGTH)\b"
    ),
}


def _run(command: list[str], *, cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(completed.stdout, end="")
    return completed.returncode, completed.stdout


def _source_contract_error(workspace: Path) -> str | None:
    model_source = (workspace / "src/model.py").read_text(encoding="utf-8")
    for description, pattern in FORBIDDEN_MODEL_PATTERNS.items():
        if pattern.search(model_source):
            return f"src/model.py contains {description}"

    data_source = (workspace / "src/data.py").read_text(encoding="utf-8")
    try:
        tree = ast.parse(data_source)
    except SyntaxError as error:
        return f"src/data.py is not valid Python: {error}"
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in {
            "preprocess",
            "postprocess",
            "encode",
            "decode",
        }:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "make_target"
            ):
                return f"src/data.py {node.name} calls the target generator"
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Add):
                names = {
                    name.id for name in ast.walk(child) if isinstance(name, ast.Name)
                }
                if {"a", "b"}.issubset(names):
                    return f"src/data.py {node.name} directly adds the operands"
    return None


def preflight_candidate_source(workspace: Path) -> str | None:
    """Reject deterministic source defects before a training process starts."""

    for relative in ("src/model.py", "src/train.py"):
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            return f"{relative} is missing or unsafe"
        try:
            compile(path.read_text(encoding="utf-8"), relative, "exec")
        except (OSError, SyntaxError, UnicodeError) as error:
            return f"{relative} does not compile: {error}"
    return _source_contract_error(workspace)


def _source_hashes(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((workspace / "src").glob("*.py"))
    }


def _load_submission(workspace: Path):
    module_name = "ten_digit_addition_submission"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name, workspace / "submission.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the protected submission interface")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _token_logits(output):
    """Normalize common decoder outputs to the token-logit tensor."""

    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return output


def _trained_model_contract_error(
    workspace: Path, *, require_last_checkpoint: bool = False
) -> str | None:
    import torch

    checkpoint = workspace / "checkpoints/best.pt"
    if not checkpoint.is_file():
        return "training did not create checkpoints/best.pt"
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - convert candidate failure to evidence
        return f"the saved model could not be loaded: {error}"
    step = payload.get("step") if isinstance(payload, dict) else None
    state = payload.get("model_state") if isinstance(payload, dict) else None
    if require_last_checkpoint:
        final_checkpoint = workspace / "checkpoints/last.pt"
        try:
            final_payload = torch.load(
                final_checkpoint, map_location="cpu", weights_only=False
            )
        except Exception:  # noqa: BLE001 - absence/corruption is a violation
            return "training did not create a valid checkpoints/last.pt"
        final_step = (
            final_payload.get("step") if isinstance(final_payload, dict) else None
        )
        final_state = (
            final_payload.get("model_state")
            if isinstance(final_payload, dict)
            else None
        )
        if (
            isinstance(final_step, bool)
            or not isinstance(final_step, int)
            or final_step < 1
            or not isinstance(final_state, dict)
            or not final_state
        ):
            return "checkpoints/last.pt does not record a positive training step"
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        # A legitimately trained but completely unsuccessful candidate can
        # leave best.pt at the step-0 validation checkpoint.  Require a
        # distinct positive-step last.pt as durable evidence that training did
        # occur, then let the normal verifier classify the candidate as a
        # nonqualification.  This still rejects an untrained or fabricated
        # step-0-only submission.
        final_checkpoint = workspace / "checkpoints/last.pt"
        try:
            final_payload = torch.load(
                final_checkpoint, map_location="cpu", weights_only=False
            )
        except Exception:  # noqa: BLE001 - absence/corruption is a violation
            return "the saved model does not record a positive training step"
        final_step = (
            final_payload.get("step") if isinstance(final_payload, dict) else None
        )
        final_state = (
            final_payload.get("model_state")
            if isinstance(final_payload, dict)
            else None
        )
        if (
            isinstance(final_step, bool)
            or not isinstance(final_step, int)
            or final_step < 1
            or not isinstance(state, dict)
            or not isinstance(final_state, dict)
        ):
            return "the saved model does not record a positive training step"
        changed = any(
            isinstance(value, torch.Tensor)
            and isinstance(final_state.get(name), torch.Tensor)
            and value.shape == final_state[name].shape
            and not torch.equal(value, final_state[name])
            for name, value in state.items()
        )
        if not changed:
            return "the positive-step final checkpoint has no learned changes"
    if not isinstance(state, dict) or not state:
        return "the saved model has no learned state"
    learned_scalars = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    if learned_scalars <= 0:
        return "the saved model has zero learned scalars"

    try:
        submission = _load_submission(workspace)
        model, _metadata = submission.build_model()
        attention_modules = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.MultiheadAttention)
            or "attention" in module.__class__.__name__.lower()
        ]
        fired: set[int] = set()
        handles = [
            module.register_forward_hook(
                lambda hooked, _inputs, _output: fired.add(id(hooked))
            )
            for module in attention_modules
        ]
        try:
            input_ids = [submission.BOS_ID] + submission.encode(
                submission.preprocess(1234567890, 9081726354)
            )
            tokens = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                logits = _token_logits(model(tokens))
        finally:
            for handle in handles:
                handle.remove()
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            return "the model does not produce token logits"
        if not fired:
            return "no learned self-attention module participates in the forward pass"

        attention_parameters = []
        seen_parameters: set[int] = set()
        for module in attention_modules:
            for parameter in module.parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                attention_parameters.append(parameter)
        saved_parameters = [
            parameter.detach().clone() for parameter in attention_parameters
        ]
        with torch.no_grad():
            try:
                for parameter in attention_parameters:
                    parameter.zero_()
                ablated_logits = _token_logits(model(tokens))
            finally:
                for parameter, saved in zip(
                    attention_parameters, saved_parameters, strict=True
                ):
                    parameter.copy_(saved)
        if torch.allclose(logits, ablated_logits, rtol=1e-5, atol=1e-6):
            return "token logits do not materially depend on learned self-attention"

        rng = random.Random(9_417_203)
        probes = [
            (rng.randrange(10**10), rng.randrange(10**10)) for _ in range(64)
        ]
        normal_correct = sum(
            submission.add(model, a, b) == a + b for a, b in probes
        )
        with torch.no_grad():
            try:
                for parameter in attention_parameters:
                    parameter.zero_()
                ablated_correct = sum(
                    submission.add(model, a, b) == a + b for a, b in probes
                )
            finally:
                for parameter, saved in zip(
                    attention_parameters, saved_parameters, strict=True
                ):
                    parameter.copy_(saved)
        if normal_correct >= 60 and (
            ablated_correct > 16 or normal_correct - ablated_correct < 32
        ):
            return (
                "exact additions remain too accurate when learned self-attention "
                "is ablated"
            )
    except Exception as error:  # noqa: BLE001 - convert candidate failure to evidence
        return f"the trained transformer contract check failed: {error}"
    return None


def _report_contract_violation(message: str) -> int:
    print(f"MODEL_CONTRACT_VIOLATION: {message}")
    return 3


def evaluate_adderboard(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    train_output = ""
    if not args.verify_existing_checkpoint:
        train_exit, train_output = _run(
            [args.python_bin, "src/train.py"], cwd=workspace
        )
        if train_exit:
            return train_exit
    verify_exit, verify_output = _run(
        [
            args.python_bin,
            str(args.repo_root / "architecture_discovery/vendor/AdderBoard/verify.py"),
            str(workspace / "submission.py"),
            "--num-tests",
            str(args.num_tests),
            "--seed",
            str(args.seed),
        ],
        cwd=workspace,
    )
    accuracy_match = ACCURACY.search(verify_output)
    parameters_match = PARAMETERS.search(verify_output)
    if verify_exit or accuracy_match is None or parameters_match is None:
        return verify_exit or 2
    steps = [int(value) for value in TRAINING_STEP.findall(train_output)]
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "metrics": {
            "accuracy": float(accuracy_match.group(3)) / 100.0,
            "parameters": int(parameters_match.group(1)),
            "correct": int(accuracy_match.group(1)),
            "cases": int(accuracy_match.group(2)),
            "training_steps": max(steps, default=0),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def evaluate_ten_digit_transformer(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    source_error = _source_contract_error(workspace)
    if source_error is not None:
        return _report_contract_violation(source_error)

    train_output = ""
    source_hashes = _source_hashes(workspace)
    if not args.verify_existing_checkpoint:
        train_exit, train_output = _run(
            [args.python_bin, "src/train.py"], cwd=workspace
        )
        if train_exit:
            return train_exit
        if _source_hashes(workspace) != source_hashes:
            return _report_contract_violation(
                "training modified source files during evaluation"
            )
        source_error = _source_contract_error(workspace)
        if source_error is not None:
            return _report_contract_violation(source_error)

    trained_error = _trained_model_contract_error(workspace)
    if trained_error is not None:
        return _report_contract_violation(trained_error)

    verify_exit, verify_output = _run(
        [
            args.python_bin,
            str(args.repo_root / "architecture_discovery/vendor/AdderBoard/verify.py"),
            str(workspace / "submission.py"),
            "--num-tests",
            str(args.num_tests),
            "--seed",
            str(args.seed),
        ],
        cwd=workspace,
    )
    accuracy_match = ACCURACY.search(verify_output)
    parameters_match = PARAMETERS.search(verify_output)
    if verify_exit or accuracy_match is None or parameters_match is None:
        return verify_exit or 2
    steps = [int(value) for value in TRAINING_STEP.findall(train_output)]
    if args.verify_existing_checkpoint:
        import torch

        payload = torch.load(
            workspace / "checkpoints/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        steps.append(int(payload["step"]))
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "metrics": {
            "accuracy": float(accuracy_match.group(3)) / 100.0,
            "parameters": int(parameters_match.group(1)),
            "correct": int(accuracy_match.group(1)),
            "cases": int(accuracy_match.group(2)),
            "training_steps": max(steps, default=0),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def evaluate_pair_token_ten_digit_transformer(args: argparse.Namespace) -> int:
    """Evaluate the frozen pair-token parent under the shared safeguards."""

    workspace = args.workspace.resolve()
    source_error = _source_contract_error(workspace)
    if source_error is not None:
        return _report_contract_violation(source_error)

    train_output = ""
    source_hashes = _source_hashes(workspace)
    if not args.verify_existing_checkpoint:
        train_command = [args.python_bin, "-m", "src.train"]
        train_device = getattr(args, "train_device", None)
        if train_device:
            train_command.extend(("--device", train_device))
        train_exit, train_output = _run(train_command, cwd=workspace)
        if train_exit:
            return train_exit
        if _source_hashes(workspace) != source_hashes:
            return _report_contract_violation(
                "training modified source files during evaluation"
            )
        source_error = _source_contract_error(workspace)
        if source_error is not None:
            return _report_contract_violation(source_error)

    trained_error = _trained_model_contract_error(
        workspace,
        require_last_checkpoint=bool(
            getattr(args, "require_last_checkpoint", False)
        ),
    )
    if trained_error is not None:
        return _report_contract_violation(trained_error)

    verify_exit, verify_output = _run(
        [
            args.python_bin,
            str(args.repo_root / "architecture_discovery/vendor/AdderBoard/verify.py"),
            str(workspace / "submission.py"),
            "--num-tests",
            str(args.num_tests),
            "--seed",
            str(args.seed),
        ],
        cwd=workspace,
    )
    accuracy_match = ACCURACY.search(verify_output)
    parameters_match = PARAMETERS.search(verify_output)
    if verify_exit or accuracy_match is None or parameters_match is None:
        return verify_exit or 2
    steps = [int(value) for value in TRAINING_STEP.findall(train_output)]
    if args.verify_existing_checkpoint:
        import torch

        payload = torch.load(
            workspace / "checkpoints/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        steps.append(int(payload["step"]))
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "metrics": {
            "accuracy": float(accuracy_match.group(3)) / 100.0,
            "parameters": int(parameters_match.group(1)),
            "correct": int(accuracy_match.group(1)),
            "cases": int(accuracy_match.group(2)),
            "training_steps": max(steps, default=0),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, layer, default_seed in (
        ("adderboard", "A", 2025),
        ("adderboard-holdout", "C", 8_724_319),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--workspace", type=Path, required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
        subparser.add_argument("--python-bin", default=sys.executable)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--num-tests", type=int, default=10_000)
        subparser.add_argument("--seed", type=int, default=default_seed)
        subparser.add_argument(
            "--verify-existing-checkpoint",
            action="store_true",
            help="Verify the workspace checkpoint without first training a candidate.",
        )
        subparser.set_defaults(handler=evaluate_adderboard, layer=layer)
    for name, layer, default_seed in (
        ("ten-digit-addition", "A", 2025),
        ("ten-digit-addition-holdout", "C", 8_724_319),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--workspace", type=Path, required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
        subparser.add_argument("--python-bin", default=sys.executable)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--num-tests", type=int, default=10_000)
        subparser.add_argument("--seed", type=int, default=default_seed)
        subparser.add_argument("--verify-existing-checkpoint", action="store_true")
        subparser.set_defaults(handler=evaluate_ten_digit_transformer, layer=layer)
    for name, layer, default_seed in (
        ("pair-token-ten-digit-addition", "A", 2025),
        ("pair-token-ten-digit-addition-holdout", "C", 8_724_319),
        ("pair-token-ten-digit-addition-v2", "A", 2025),
        ("pair-token-ten-digit-addition-v2-holdout", "C", 8_724_319),
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--workspace", type=Path, required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
        subparser.add_argument("--python-bin", default=sys.executable)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument("--num-tests", type=int, default=10_000)
        subparser.add_argument("--seed", type=int, default=default_seed)
        subparser.add_argument("--verify-existing-checkpoint", action="store_true")
        subparser.add_argument(
            "--train-device", choices=("cpu", "mps", "cuda"), default=None
        )
        subparser.set_defaults(
            handler=evaluate_pair_token_ten_digit_transformer,
            layer=layer,
            require_last_checkpoint="-v2" in name,
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
