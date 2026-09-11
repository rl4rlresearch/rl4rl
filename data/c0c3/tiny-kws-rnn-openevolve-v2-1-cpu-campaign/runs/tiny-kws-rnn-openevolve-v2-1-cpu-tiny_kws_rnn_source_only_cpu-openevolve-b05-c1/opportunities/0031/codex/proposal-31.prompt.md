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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2471524730303178038, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25008, "peak_hidden_elements": 147968, "recurrent_macs": 471943680, "recurrent_steps": 21190, "total_inference_macs": 473821440, "training_seconds": 116.72218300006352, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.3903332482086369}
prior_hypothesis: A 96-unit full-rank single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame max-augmented schedule while reducing total dense inference MACs by about 9.2%, from 521.65M to 473.82M.

## Recent verification evidence

RECENT RESULT
hypothesis: An 85-unit GRU processing 27 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.5% versus the verified 84-unit, 28-step model.
change: Increase the GRU and dual-timescale readout width from 84 to 85 while reducing the causal schedule from 28 to 27 frames.
mechanism: Width-for-time Pareto continuation
evidence_used: The consecutive diagonal designs from 81 units/31 steps through 84 units/28 steps all passed; the current 84-unit, 28-step model achieved 85.64% accuracy, supporting one more incremental width-for-time trade.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3079054328063406973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 589183875, "recurrent_steps": 22005, "total_inference_macs": 590292275, "training_seconds": 76.18085374985822, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4303115563889954}

RECENT RESULT
hypothesis: A persistent 128-unit gated state with a learned rank-32 recurrent bottleneck will retain at least 85% accuracy on the 28-frame schedule while reducing total dense inference MACs by approximately 8% versus the verified 84-unit GRU.
change: Replace the full-rank GRU with a custom GRU-like update whose input transformation remains wide but whose recurrent gate transformation passes through a shared 32-dimensional learned bottleneck; preserve the mean-plus-terminal readout and endpoint-preserving schedule.
mechanism: Wide-state low-rank gated recurrence
evidence_used: The 84-unit, 28-step GRU passed at 85.64% accuracy and 599.16M MACs, while the failed reset-based hierarchy reached only 82.94%, indicating that persistent state and retained observations are load-bearing. This patch preserves both while directly testing the shared assumption that every gate needs an independent full-rank hidden-to-hidden matrix.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2873106687621594964, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 26544, "peak_hidden_elements": 131584, "recurrent_macs": 549140480, "recurrent_steps": 22820, "total_inference_macs": 550809600, "training_seconds": 142.08633808302693, "validation_accuracy": 0.8269938650306748, "validation_cross_entropy": 0.4821070946067389}

RECENT RESULT
hypothesis: An 85-unit GRU using 27 frames formed by removing only the second frame from the successful 28-frame schedule will achieve at least 85% accuracy while reducing total inference MACs by approximately 1.5%.
change: Increase the GRU width from 84 to 85 and replace uniform 27-frame resampling with a nested schedule that preserves 27 of the verified model’s 28 frame locations.
mechanism: Nested endpoint-preserving frame pruning
evidence_used: The 84-unit/28-step model passed at 85.64%, while the uniformly resampled 85-unit/27-step model narrowly failed at 84.54% despite lower MACs; that resampling changed seven frame locations, so pruning one likely low-information leading frame isolates step reduction from broad temporal realignment.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3079054328063406973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 589183875, "recurrent_steps": 22005, "total_inference_macs": 590292275, "training_seconds": 55.72199074993841, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4282363236316143}

RECENT RESULT
hypothesis: An 86-unit GRU processing 26 frames formed by removing one additional early frame from the successful 85-unit/27-step schedule will retain at least 85% accuracy while reducing total inference MACs by approximately 1.6%.
change: Increase GRU and readout width from 85 to 86, resize the classifier accordingly, and prune the first two nonzero frames from the verified 28-frame base schedule.
mechanism: Nested frame pruning with width-for-time trade
evidence_used: The nested 85-unit/27-step design achieved 86.63% accuracy, substantially outperforming uniform 27-frame resampling; this supports preserving its frame alignment while making the next incremental width-for-time trade.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3028631271644885638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 579504120, "recurrent_steps": 21190, "total_inference_macs": 580625560, "training_seconds": 75.63254175009206, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4487786977569018}

RECENT RESULT
hypothesis: An 86-unit GRU retaining the early frame lost by the failed 26-step design and instead pruning the penultimate frame will achieve at least 85% accuracy at 580.63M total inference MACs.
change: Increase the GRU/readout width from 85 to 86 and create a 26-frame schedule by removing the already-proven redundant second frame plus the penultimate frame from the successful 28-frame base schedule.
mechanism: Onset-preserving boundary frame pruning
evidence_used: The 85-unit/27-step schedule passed at 86.63%, while the 86-unit/26-step design that removed both earliest nonzero frames missed the threshold by only 0.09 points; preserving potential onset information while pruning a likely trailing-silence frame directly tests whether pruning location caused that failure.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3028631271644885638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 579504120, "recurrent_steps": 21190, "total_inference_macs": 580625560, "training_seconds": 77.47596350009553, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4552927496974454}

RECENT RESULT
hypothesis: Adding temporal max pooling to the closest-failing 86-unit, 26-step GRU will raise validation accuracy from 84.91% to at least 85% while keeping total inference MACs near 581.19M, below the current 590.29M design.
change: Use the previously tested early-pruned 26-frame schedule and concatenate mean, maximum, and terminal GRU outputs for classification.
mechanism: Transient-preserving max-augmented recurrent readout
evidence_used: The early-pruned 86-unit/26-step model missed the threshold by only 0.09 points. A max summary preserves transient command evidence that mean and terminal summaries can dilute after removing frames, at a cost of only 560,720 additional validation MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3031556072484602246, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29976, "peak_hidden_elements": 132608, "recurrent_macs": 579504120, "recurrent_steps": 21190, "total_inference_macs": 581186280, "training_seconds": 66.93220995785668, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4158541907562069}

RECENT RESULT
hypothesis: An 85-unit GRU with the successful 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.1%, from 581.19M to 569.02M.
change: Reduce the GRU and summary width from 86 to 85 and resize the concatenated classifier input from 258 to 255, preserving the verified frame schedule and training procedure.
mechanism: Max-supported recurrent width compression
evidence_used: The 86-unit, 26-step model reached 86.01% after max pooling rescued the same schedule’s 84.91% mean-terminal result, while an 85-unit model previously reached 86.63% at 27 steps; this supports testing the smallest width reduction without removing the accuracy-restoring readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2968120203109251413, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29373, "peak_hidden_elements": 131072, "recurrent_macs": 567362250, "recurrent_steps": 21190, "total_inference_macs": 569024850, "training_seconds": 34.57787450007163, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41563148732565663}

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
        self.hidden_size = 96
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
            del indices[1:3]
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


