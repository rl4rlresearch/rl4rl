# Improve a recurrent keyword spotter

You are an autonomous ML engineer improving a learned causal recurrent model
that classifies one-second speech recordings into eight spoken commands.

## Goal

Produce a model with at least 85% accuracy on the fixed speaker-disjoint public
validation split, then minimize exact dense inference MACs. Among equal-MAC
models, fewer executed recurrent steps wins; among exact MAC-and-step ties,
fewer learned parameters wins. Every verification starts from fresh random
initialization and presents exactly 50,000 training clips drawn from a protected
training-speaker split.

The protected frontend supplies batches shaped `[batch, 32, 20]`: 32 causal
time frames with 20 normalized log-mel bands. `train.py` owns the model,
optimizer, loss, temporal augmentation, batch size, gradient handling, and
schedule. Keep its five top-level function interfaces intact.

The model interface is deliberately recurrent and evaluator-driven:

- `initial_state(batch_size, device, dtype)` returns batch-first tensor state,
  or a tuple/list of batch-first tensor states;
- `recurrent_step(frame, state)` updates that state from one `[batch, 20]`
  frame;
- `classify(state)` returns `[batch, 8]` logits;
- optional `recurrent_sequence(frames, state)` may run a standard causal
  sequence module efficiently, but must be numerically equivalent to repeated
  `recurrent_step` calls;
- optional `frame_schedule(available_frames)` returns 2–64 unique increasing
  input-frame indices, allowing causal striding;
- optional `exit_mask(state, logits, step, total_steps)` returns one boolean per
  active example after the mandatory first two recurrent steps.

All learned matrix operations must use `nn.Linear`, the standard
`nn.RNN`/`nn.GRU`/`nn.LSTM` modules, or their corresponding cell modules. Their
exact executed MACs are counted with protected runtime hooks over the complete
validation set. Bidirectional recurrence is rejected. Direct matmul, functional linear,
convolutions, and manually created Parameters are rejected because they could
bypass that counter. Dense matrices receive no credit for zero weights; only
structural reductions reduce cost. Elementwise gates, nonlinearities,
normalization, and recurrence logic remain flexible.

The verifier requires a state updated across at least two causal steps, material
dependence of the next state on the prior state, logits that materially depend
on recurrent output, learned recurrent-path weight changes, no complete-input
classifier bypass, and complete accounting of every executed recurrent step.
Layer C uses recordings from speakers absent from both search training and
public validation.

Public feedback includes accuracy, cross-entropy, the exact lexicographic
`inference_cost`, total and recurrent MACs, recurrent-step summaries, parameters,
peak hidden elements, training exposure, and training time.

## Work boundaries

Minimize inference_cost. Required result: validation_accuracy >= 0.85.
Editable source files: train.py.
Results reported after each verification: validation_accuracy, validation_cross_entropy, inference_cost, total_inference_macs, recurrent_macs, recurrent_steps, mean_recurrent_steps, median_recurrent_steps, p95_recurrent_steps, maximum_recurrent_steps, parameters, peak_hidden_elements, examples_processed, optimizer_steps, training_seconds, batch_size.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, external datasets, pretrained weights, or any
surrounding repository. Do not run training or validation yourself and do not
generate hidden alternatives. Return one patch for one implementation;
verification happens after you finish.

## Available designs

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 66.25597575004213, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4378691690831097}
prior_hypothesis: Processing 28 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed recurrent steps by 12.5% versus the verified 32-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: Using 24 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by exactly 25%.
change: Replace the full 32-frame schedule with a 24-frame schedule spanning the complete recording, including both endpoints.
mechanism: Uniform 24-step causal frame subsampling
evidence_used: The 128-unit GRU reaches 87.24% accuracy at 32 steps, providing a 2.24-point margin above the target; reducing only temporal redundancy preserves model capacity while targeting the dominant 1,482,178,560 recurrent MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 98.55075808311813, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.47674989290763997}

RECENT RESULT
hypothesis: Processing 28 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed recurrent steps by 12.5% versus the verified 32-step model.
change: Replace the full-frame schedule with 28 unique, increasing indices spanning the complete recording, including both endpoints.
mechanism: Uniform 28-step causal frame subsampling
evidence_used: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 84.42%; testing the midpoint of this bracket is the most informative next reduction with greater accuracy margin than the failed 24-step design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 66.25597575004213, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4378691690831097}

RECENT RESULT
hypothesis: Processing 27 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-step model.
change: Reduce the frame schedule from 28 to 27 unique, increasing indices spanning the complete recording.
mechanism: Uniform 27-step causal frame subsampling
evidence_used: The 28-step model achieved 85.276% accuracy, while 24 steps achieved 84.417%; a one-step reduction is the smallest structural cost reduction and the most conservative probe within this observed boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6527611327453078597, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1250588160, "recurrent_steps": 22005, "total_inference_macs": 1251422720, "training_seconds": 105.01874833297916, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.47251288291135446}

RECENT RESULT
hypothesis: A 125-unit GRU processing the verified 28-frame schedule will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 4.3%, below even the failed 27-step 128-unit design.
change: Reduce the GRU hidden state and classifier input width from 128 to 125 without changing training or the successful 28-frame schedule.
mechanism: Preserve temporal coverage while trimming recurrent state width
evidence_used: The 128-unit model passed at 28 steps with 85.276% accuracy but failed at 27 steps with 84.294%, indicating temporal evidence is sensitive; retaining all 28 observations while making a modest width reduction targets quadratic recurrent cost instead.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6476648373368131493, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 56173, "peak_hidden_elements": 128512, "recurrent_macs": 1240837500, "recurrent_steps": 22820, "total_inference_macs": 1241652500, "training_seconds": 151.87313704192638, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.44364757888887557}

RECENT RESULT
hypothesis: A 126-unit GRU on the verified 28-frame schedule will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 2.9% versus the passing 128-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 126 while preserving the successful training procedure and 28-frame schedule.
mechanism: One-channel-width boundary probe
evidence_used: The 128-unit model passed at 85.276%, while 125 units narrowly missed at 84.908%; 126 units is the lowest untested width between these results and therefore the most informative next cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6573455880231520300, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 57000, "peak_hidden_elements": 129536, "recurrent_macs": 1259390160, "recurrent_steps": 22820, "total_inference_macs": 1260211680, "training_seconds": 129.2636459171772, "validation_accuracy": 0.838036809815951, "validation_cross_entropy": 0.4534977637916986}

RECENT RESULT
hypothesis: A 127-unit GRU using the verified 28-frame schedule will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.45% versus the passing 128-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 127 while preserving the successful schedule and training procedure.
mechanism: One-channel recurrent-width boundary trim
evidence_used: The 128-unit model passed at 85.276%, while 125 units narrowly missed at 84.908%; despite the noisy 126-unit result, 127 is the most conservative untested structural reduction and isolates the remaining width boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6670977582648793233, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1278079740, "recurrent_steps": 22820, "total_inference_macs": 1278907780, "training_seconds": 112.90339691587724, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4357971753079467}

RECENT RESULT
hypothesis: A 125-unit GRU processing 29 uniformly distributed frames will reach at least 85% validation accuracy while reducing exact dense inference MACs by approximately 0.9% versus the passing 128-unit, 28-step model.
change: Reduce the recurrent and classifier width from 128 to 125, while increasing the causal frame schedule from 28 to 29 frames spanning the full recording.
mechanism: Width-for-temporal-coverage tradeoff
evidence_used: The 125-unit, 28-step model narrowly missed the target at 84.908%, only 0.092 percentage points short; adding one temporal observation targets that deficit while the narrower recurrence still yields fewer MACs than the verified 128-unit, 28-step design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6707805416559447933, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 56173, "peak_hidden_elements": 128512, "recurrent_macs": 1285153125, "recurrent_steps": 23635, "total_inference_macs": 1285968125, "training_seconds": 105.90041587501764, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4578903479078796}



Use the available technical evidence to choose the most informative next
change. Treat unsuccessful or malformed work as evidence when a useful
subject-level reason is provided. Do not invent missing evidence.

## Response

Return these short metadata lines followed by one or more exact
`SEARCH`/`REPLACE` blocks that together produce one implementation:

`MECHANISM: <a concise free-form name for the computational idea>`

`HYPOTHESIS: <a falsifiable claim grounded in the evidence above>`

`INTENDED_EDIT: <what this patch changes>`

`EVIDENCE: <the most relevant prior result and why it motivates this patch>`

Start each block with `<<<<<<< SEARCH`, put the exact existing lines next, use a
line containing `=======` as the divider, put the replacement lines after it,
and finish the block with `>>>>>>> REPLACE`.

Every `SEARCH` section must be nonempty and match exactly once after earlier
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.


# Editable source

```text
===== FILE: train.py =====
"""Editable recurrent keyword-spotting research program.

The protected evaluator owns the audio, speaker-disjoint splits, log-mel
frontend, training exposure, recurrent execution loop, and exact MAC counter.
This file owns the recurrent model and trainable procedure.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]


def build_model() -> nn.Module:
    return KeywordGRU()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)


def prepare_training_batch(
    frames: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    if torch.rand(()) < 0.8:
        frames = frames + 0.025 * torch.randn_like(frames)
    return frames, labels


def training_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)


def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    multiplier = 0.05 + 0.95 * 0.5 * (
        1.0 + math.cos(math.pi * step / max(total_steps, 1))
    )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
===== END FILE =====

```


