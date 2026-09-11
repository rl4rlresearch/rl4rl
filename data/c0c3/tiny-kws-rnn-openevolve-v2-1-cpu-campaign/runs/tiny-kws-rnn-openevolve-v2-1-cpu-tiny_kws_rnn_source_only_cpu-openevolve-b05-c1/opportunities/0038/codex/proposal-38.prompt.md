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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1825628377341731875, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 348371750, "recurrent_steps": 20375, "total_inference_macs": 349995230, "training_seconds": 85.62386012496427, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.43377429985561256}
prior_hypothesis: Removing one additional densely spaced early frame from the 83-unit single-gate model will preserve at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and total inference MACs by approximately 3.8%.

## Recent verification evidence

RECENT RESULT
hypothesis: An 84-unit GRU with the successful 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.1%, from 569.02M to 556.99M.
change: Reduce the GRU and all summary-state widths from 85 to 84 and resize the concatenated classifier input from 255 to 252, preserving the verified schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 85-unit/26-step max-readout model achieved 86.26%, while an 84-unit model without max pooling passed at 85.64% over 28 steps; this supports testing one-unit compression while retaining the stronger readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2905347515319650126, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 555347520, "recurrent_steps": 21190, "total_inference_macs": 556990560, "training_seconds": 38.97008291701786, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4119058901546923}

RECENT RESULT
hypothesis: An 83-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.2% versus the successful 84-unit model.
change: Reduce the GRU and all recurrent summary widths from 84 to 83, and resize the classifier input from 252 to 249 while preserving the frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 84-unit/26-step model achieved 86.63% accuracy, providing substantial margin, and the prior 83-unit/29-step model achieved 86.01%; together these results support a one-unit width reduction while retaining the accuracy-restoring max readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2843238009115798385, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 28185, "peak_hidden_elements": 128000, "recurrent_macs": 543459930, "recurrent_steps": 21190, "total_inference_macs": 545083410, "training_seconds": 42.974731041816995, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4236403763659893}

RECENT RESULT
hypothesis: An 82-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.2% versus the successful 83-unit model.
change: Reduce the GRU and recurrent summary width from 83 to 82 and resize the classifier input from 249 to 246, preserving the frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 83-unit/26-step model achieved 86.87% accuracy, giving 1.87 percentage points of margin, while an 82-unit model previously achieved 85.52% at 30 steps; this supports testing the smallest structural width reduction with the stronger max-augmented readout intact.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2781791684497696190, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 27600, "peak_hidden_elements": 126464, "recurrent_macs": 531699480, "recurrent_steps": 21190, "total_inference_macs": 533303400, "training_seconds": 36.711444332962856, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4176021178075872}

RECENT RESULT
hypothesis: An 81-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 2.2%, from 533.30M to 521.65M.
change: Reduce the GRU and recurrent summary width from 82 to 81 and resize the concatenated classifier input from 246 to 243, preserving the frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 82-unit/26-step model achieved 86.50% accuracy, leaving 1.50 percentage points of margin, and every prior one-unit max-readout compression from 85 through 82 units passed; an earlier 81-unit model also passed at 31 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2721008541465343541, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 27021, "peak_hidden_elements": 124928, "recurrent_macs": 520066170, "recurrent_steps": 21190, "total_inference_macs": 521650530, "training_seconds": 31.502919625025243, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4182519362748035}

RECENT RESULT
hypothesis: A 96-unit full-rank single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame max-augmented schedule while reducing total dense inference MACs by about 9.2%, from 521.65M to 473.82M.
change: Replace the 81-unit three-gate GRU with a wider 96-unit recurrent update using one learned retention gate and one full-rank normalized proposal, while preserving mean-max-terminal classification and the successful frame schedule.
mechanism: Full-rank single-retention-gate recurrence
evidence_used: The current 81-unit GRU reaches 86.99%, providing accuracy margin, while the 128-unit rank-32 recurrent bottleneck failed at 82.70%. This motivates challenging the assumption that three gates are necessary without repeating the failed low-rank assumption: both retained state and proposal remain full-rank, and the wider state compensates for removing the reset gate.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2471524730303178038, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25008, "peak_hidden_elements": 147968, "recurrent_macs": 471943680, "recurrent_steps": 21190, "total_inference_macs": 473821440, "training_seconds": 116.72218300006352, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.3903332482086369}

RECENT RESULT
hypothesis: A 92-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 7.5%, from 473.82M to 438.48M.
change: Reduce the recurrent, temporal-summary, and classifier width from 96 to 92 units while preserving the successful recurrence, readout, schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 96-unit single-gate model achieved 87.73% accuracy, exceeding the requirement by 2.73 percentage points; this substantial margin motivates a moderate four-unit structural compression before altering the proven temporal schedule or gating mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2287194258776893862, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 23232, "peak_hidden_elements": 141824, "recurrent_macs": 436683520, "recurrent_steps": 21190, "total_inference_macs": 438483040, "training_seconds": 112.9495802919846, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40820705671251917}

RECENT RESULT
hypothesis: A 90-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 3.9%, from 438.48M to 421.32M.
change: Reduce the recurrent, temporal-summary, and classifier width from 92 to 90 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 92-unit single-gate model achieved 86.13% accuracy at 438.48M MACs, while the 96-unit version achieved 87.73%; this supports a measured two-unit compression that tests the lower-width boundary without changing the proven temporal or gating design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2197681749356749958, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 22368, "peak_hidden_elements": 138752, "recurrent_macs": 419562000, "recurrent_steps": 21190, "total_inference_macs": 421322400, "training_seconds": 92.76778687513433, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41911935630751534}

RECENT RESULT
hypothesis: An 88-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.0%, from 421.32M to 404.50M.
change: Reduce the recurrent, temporal-summary, and classifier width from 90 to 88 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 90-unit model achieved 86.26% accuracy, and the 92-unit model achieved 86.13%; both exceed the threshold by more than one percentage point, supporting another measured two-unit compression without changing the proven temporal or gating design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2109937724165271510, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 21520, "peak_hidden_elements": 135680, "recurrent_macs": 402779520, "recurrent_steps": 21190, "total_inference_macs": 404500800, "training_seconds": 102.02311599999666, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40685732788834833}

RECENT RESULT
hypothesis: An 86-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.1%, from 404.50M to 388.02M.
change: Reduce the recurrent, temporal-summary, and classifier width from 88 to 86 units while preserving the recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 88-unit model achieved 86.13% accuracy, while the 90- and 92-unit models also exceeded 86%; the stable results across successive two-unit reductions motivate continuing the same controlled compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2023962183202458518, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 20688, "peak_hidden_elements": 132608, "recurrent_macs": 386336080, "recurrent_steps": 21190, "total_inference_macs": 388018240, "training_seconds": 96.42121758288704, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.40881503988628737}

RECENT RESULT
hypothesis: An 84-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 4.2%, from 388.02M to 371.87M.
change: Reduce the recurrent, temporal-summary, and classifier width from 86 to 84 units while preserving the successful recurrence, readout, frame schedule, and training procedure.
mechanism: Margin-guided single-gate width compression
evidence_used: The 86-unit model achieved 87.24% accuracy, the strongest margin among the compressed single-gate designs, while every preceding two-unit reduction from 92 through 86 units remained above 86%; this supports continuing the established two-unit compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1939755126468310982, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 19872, "peak_hidden_elements": 129536, "recurrent_macs": 370231680, "recurrent_steps": 21190, "total_inference_macs": 371874720, "training_seconds": 86.15392604190856, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.42746424762749236}

RECENT RESULT
hypothesis: An 83-unit single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame schedule while reducing total dense inference MACs by approximately 2.1%, from 371.87M to 363.93M.
change: Reduce the recurrent, temporal-summary, and classifier width from 84 to 83 units while preserving the recurrence, readout, frame schedule, and training procedure.
mechanism: Boundary-aware single-gate width compression
evidence_used: The 84-unit model passed at 85.64% accuracy, while every tested single-gate width from 86 through 96 also passed; because the latest result has narrower margin, a one-unit reduction is the most controlled test of the compression boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1898314779686986760, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 362306620, "recurrent_steps": 21190, "total_inference_macs": 363930100, "training_seconds": 82.48713024985045, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4196209772964197}

RECENT RESULT
hypothesis: Removing one additional densely spaced early frame from the 83-unit single-gate model will preserve at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and total inference MACs by approximately 3.8%.
change: Expand the existing schedule deletion from two to three early indices, preserving the recurrent architecture, temporal readout, training procedure, and full recording coverage.
mechanism: Redundant early-frame pruning
evidence_used: The current 83-unit single-gate model achieved 86.26% accuracy at 26 steps, leaving 1.26 percentage points of margin; its schedule already benefits from removing two adjacent early frames, motivating a controlled test that removes one more nearby frame while retaining the first and final frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1825628377341731875, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 19470, "peak_hidden_elements": 128000, "recurrent_macs": 348371750, "recurrent_steps": 20375, "total_inference_macs": 349995230, "training_seconds": 85.62386012496427, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.43377429985561256}



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


class KeywordSingleGate(nn.Module):
    """A full-rank single-gate recurrence with multi-view temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 83
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].fill_(1.0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        peak = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
        hidden = retention * hidden + (1.0 - retention) * proposal
        return (
            hidden,
            summary + hidden,
            torch.maximum(peak, hidden),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, peak, hidden), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        base_steps = min(28, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 27:
            del indices[1:4]
        return indices


def build_model() -> nn.Module:
    return KeywordSingleGate()


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


