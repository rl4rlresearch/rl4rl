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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3187565289042172533, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 610462710, "recurrent_steps": 17930, "total_inference_macs": 611095150, "training_seconds": 129.0912150000222, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.42572902375203703}
prior_hypothesis: A 97-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference cost to approximately 611.1 million MACs.

REFERENCE DESIGN 1
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3234982772260267057, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 619514100, "recurrent_steps": 16300, "total_inference_macs": 620185660, "training_seconds": 161.66547937504947, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.48220550911558185}
prior_hypothesis: Training the 103-unit GRU at batch size 16 on frames 4–23 will retain at least 85% validation accuracy while reducing standard inference from 21 to 20 recurrent steps and total MACs to approximately 620.2 million.

REFERENCE DESIGN 2
verified_results: {"batch_size": 32, "examples_processed": 50000, "inference_cost": 3558130753571765697, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 1564, "p95_recurrent_steps": 22, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 681465510, "recurrent_steps": 17930, "total_inference_macs": 682137070, "training_seconds": 79.19273712509312, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.46322588774324197}
prior_hypothesis: Training the 22-step frames 4–25 model with batch size 32 will recover its 0.21-point accuracy shortfall and reach at least 85% validation accuracy while retaining approximately 682.1 million total inference MACs.

REFERENCE DESIGN 3
verified_results: {"batch_size": 32, "examples_processed": 50000, "inference_cost": 3396556762916016377, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 1564, "p95_recurrent_steps": 21, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 650489805, "recurrent_steps": 17115, "total_inference_macs": 651161365, "training_seconds": 101.10433149989694, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.47942851247963}
prior_hypothesis: A 103-unit GRU trained at batch size 32 on frames 4–24 will retain at least 85% validation accuracy while reducing inference from 22 to 21 recurrent steps and total MACs to approximately 651.2 million.

## Recent verification evidence

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–29 will retain at least 85% validation accuracy while reducing execution from the qualified 27-step design to 26 steps and total inference MACs to approximately 806 million.
change: Use the qualified batch-size-64 training procedure and remove frame 30 from its contiguous schedule, processing 26 frames per standard recording.
mechanism: Higher-update optimization with one-step trailing trim
evidence_used: The batch-size-64 model on frames 4–30 achieved 87.24% accuracy at 27 steps, providing 2.24 percentage points of margin; trimming one adjacent trailing frame is the smallest untested reduction below that qualified cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4204426716194762977, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 786, "p95_recurrent_steps": 26, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 805368330, "recurrent_steps": 21190, "total_inference_macs": 806039890, "training_seconds": 74.47286183293909, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.41319922698787387}

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–28 will retain at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and total inference MACs to approximately 775,064,185.
change: Halve the training batch size and schedule 25 contiguous frames for standard 32-frame recordings by omitting frames 0–3 and 29–31.
mechanism: Higher-update optimization with incremental trailing-frame trim
evidence_used: The batch-size-64, frames 4–29 design achieved 87.36% accuracy at 26 steps, leaving 2.36 percentage points of margin; removing the adjacent trailing frame is the smallest untested reduction below that qualified frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4042852725539013657, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 786, "p95_recurrent_steps": 25, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 774392625, "recurrent_steps": 20375, "total_inference_macs": 775064185, "training_seconds": 62.33617270900868, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.43089763430729966}

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 64 on frames 4–27 will retain at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and total inference MACs to approximately 744,088,480.
change: Remove frame 28 from the current contiguous schedule, processing 24 frames for standard 32-frame recordings.
mechanism: Incremental trailing-frame trim
evidence_used: The batch-size-64 frames 4–28 design achieved 86.26% accuracy at 25 steps; removing the adjacent trailing frame is the smallest untested reduction below the current qualified frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3881278734883264337, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 786, "p95_recurrent_steps": 24, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 743416920, "recurrent_steps": 19560, "total_inference_macs": 744088480, "training_seconds": 64.33718712511472, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4478612420017734}

RECENT RESULT
hypothesis: The 103-unit GRU trained at batch size 64 on frames 4–26 will retain at least 85% validation accuracy while reducing inference to 23 recurrent steps and approximately 713.1 million total MACs.
change: Schedule 23 contiguous frames for standard recordings by omitting frames 27–31, preserving the architecture and training procedure.
mechanism: Incremental trailing-frame trim at the qualified frontier
evidence_used: The qualified frames 4–27 design achieved 86.26% accuracy at 24 steps, leaving 1.26 percentage points of margin; removing the adjacent trailing frame is the smallest untested structural reduction below that frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3719704744227515017, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 712441215, "recurrent_steps": 18745, "total_inference_macs": 713112775, "training_seconds": 63.0624906248413, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.47033460184108994}

RECENT RESULT
hypothesis: The 103-unit GRU trained at batch size 64 on frames 4–25 will retain at least 85% validation accuracy while reducing standard inference from 23 to 22 recurrent steps and total MACs to approximately 682.1 million.
change: Remove frame 26 from the qualified contiguous schedule, processing 22 frames for standard 32-frame recordings while preserving safe schedules for shorter inputs.
mechanism: Incremental trailing-frame trim at the 23-step frontier
evidence_used: The batch-size-64 model on frames 4–26 achieved 85.52% accuracy at 23 steps; removing the adjacent trailing frame is the smallest untested structural reduction below the qualified cost frontier.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3558130753571765697, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 786, "p95_recurrent_steps": 22, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 681465510, "recurrent_steps": 17930, "total_inference_macs": 682137070, "training_seconds": 53.65904766693711, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.49172538336069305}

RECENT RESULT
hypothesis: Training the 22-step frames 4–25 model with batch size 32 will recover its 0.21-point accuracy shortfall and reach at least 85% validation accuracy while retaining approximately 682.1 million total inference MACs.
change: Remove frame 26 from the qualified schedule and halve the training batch size from 64 to 32, doubling the approximate number of optimizer updates without changing inference architecture or cost.
mechanism: Higher-update optimization at the 22-step frontier
evidence_used: At 27 steps, reducing batch size from 128 to 64 raised accuracy from 84.66% to 87.24%; the batch-64 22-step model already reached 84.79%, so applying the same higher-update strategy targets a much smaller deficit.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 32, "examples_processed": 50000, "inference_cost": 3558130753571765697, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 1564, "p95_recurrent_steps": 22, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 681465510, "recurrent_steps": 17930, "total_inference_macs": 682137070, "training_seconds": 79.19273712509312, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.46322588774324197}

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 32 on frames 4–24 will retain at least 85% validation accuracy while reducing inference from 22 to 21 recurrent steps and total MACs to approximately 651.2 million.
change: Halve the current training batch size and use a 21-frame contiguous schedule for standard 32-frame recordings, preserving the architecture and optimization procedure.
mechanism: Higher-update optimization with one-step trailing trim
evidence_used: The batch-size-32 model on frames 4–25 achieved 85.89% accuracy at 22 steps, while batch size 32 recovered the corresponding batch-size-64 model from 84.79%; its 0.89-point margin supports testing the next incremental trailing-frame trim with the qualified higher-update procedure.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 32, "examples_processed": 50000, "inference_cost": 3396556762916016377, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 1564, "p95_recurrent_steps": 21, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 650489805, "recurrent_steps": 17115, "total_inference_macs": 651161365, "training_seconds": 101.10433149989694, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.47942851247963}

RECENT RESULT
hypothesis: Training the 103-unit GRU at batch size 16 on frames 4–23 will retain at least 85% validation accuracy while reducing standard inference from 21 to 20 recurrent steps and total MACs to approximately 620.2 million.
change: Halve the qualified 21-step design’s training batch size and remove its final scheduled frame, while preserving the architecture and optimization procedure.
mechanism: Higher-update optimization with one-step trailing trim
evidence_used: Batch size 32 recovered the 22-step model from 84.79% to 85.89%, and the resulting 21-step model still qualified at 85.15%; another batch-size halving directly tests whether additional optimizer updates can support the next incremental trim without adding inference cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3234982772260267057, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 619514100, "recurrent_steps": 16300, "total_inference_macs": 620185660, "training_seconds": 161.66547937504947, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.48220550911558185}

RECENT RESULT
hypothesis: Training the 103-unit GRU at batch size 8 on frames 4–22 will retain at least 85% validation accuracy while reducing standard inference from 20 to 19 recurrent steps and total MACs to approximately 589.2 million.
change: Halve the training batch size and remove the final scheduled frame, while preserving the qualified architecture, optimizer, augmentation, and loss.
mechanism: Higher-update optimization with one-step trailing trim
evidence_used: Successive batch-size halvings enabled the 22-step and 20-step frontiers to qualify; most recently, batch size 16 produced 85.03% accuracy at 20 steps, motivating the same optimization-for-step exchange at 19 steps.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 16 on 19 frames distributed across the qualified frames 4–24 window will retain at least 85% validation accuracy while reducing total inference MACs to approximately 589.2 million.
change: Use batch size 16 and cap the schedule at 19 evenly distributed frames, preserving both endpoints of the qualified 21-frame temporal window.
mechanism: Full-span interior frame thinning
evidence_used: The contiguous 20-frame batch-size-16 design qualified at 85.03%, while the 21-frame frames 4–24 design qualified at 85.15%; thinning two redundant interior frames tests a lower-cost frontier without discarding the later frames omitted by the prior trailing-trim strategy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3073408781604517737, "maximum_recurrent_steps": 19, "mean_recurrent_steps": 19.0, "median_recurrent_steps": 19, "optimizer_steps": 3127, "p95_recurrent_steps": 19, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 588538395, "recurrent_steps": 15485, "total_inference_macs": 589209955, "training_seconds": 130.31128349993378, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.48320044769099885}

RECENT RESULT
hypothesis: Allowing predictions above 90% confidence to exit after 16 recurrent steps will preserve at least 85% validation accuracy while lowering mean execution below the qualified fixed 20-step model and therefore reducing total inference MACs below 620,185,660.
change: Add a parameter-free early-exit rule to the qualified 22-step, batch-size-32 model; uncertain examples retain all 22 steps while confident examples stop after step 16 or later.
mechanism: Confidence-gated adaptive recurrent depth
evidence_used: The current 22-step model reaches 85.89% accuracy, providing more margin than the barely qualifying 20-step model at 85.03%; the failed fixed 19-step thinning result shows that removing context uniformly is risky, motivating conditional computation that preserves full context for difficult clips.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 97-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference cost to approximately 611.1 million MACs.
change: Replace the 103-unit, 20-step frontier with a narrower 97-unit GRU processing 22 contiguous frames, using batch size 16 for additional optimizer updates.
mechanism: Width-for-temporal-context exchange
evidence_used: The 103-unit 22-step design reached 85.89% accuracy, providing substantially more margin than the 20-step design’s 85.03%; reducing width to 97 lowers estimated recurrent MACs despite restoring two informative frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3187565289042172533, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 610462710, "recurrent_steps": 17930, "total_inference_macs": 611095150, "training_seconds": 129.0912150000222, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.42572902375203703}



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

BATCH_SIZE = 16
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
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
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))


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

# Reference source

REFERENCE DESIGN 1
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

BATCH_SIZE = 16
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
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
        if available_frames <= 13:
            return list(range(available_frames))
        return list(range(4, available_frames - 8))


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

REFERENCE DESIGN 2
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

BATCH_SIZE = 32
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
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
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))


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

REFERENCE DESIGN 3
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

BATCH_SIZE = 32
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
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
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))


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
