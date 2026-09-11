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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3079054328063406973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 589183875, "recurrent_steps": 22005, "total_inference_macs": 590292275, "training_seconds": 55.72199074993841, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4282363236316143}
prior_hypothesis: An 85-unit GRU using 27 frames formed by removing only the second frame from the successful 28-frame schedule will achieve at least 85% accuracy while reducing total inference MACs by approximately 1.5%.

## Recent verification evidence

RECENT RESULT
hypothesis: An 82-unit GRU with the full 32-frame schedule and mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.3% versus the verified 84-unit model.
change: Reduce the GRU and summary width from 84 to 82 and resize the concatenated classifier input from 168 to 164, leaving training and temporal coverage unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 84-unit model achieved 87.73% accuracy, 2.73 points above the requirement, and all tested dual-readout widths from 84 through 91 passed; this supports another cautious two-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3419024165611784064, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26944, "peak_hidden_elements": 84480, "recurrent_macs": 654399360, "recurrent_steps": 26080, "total_inference_macs": 655468640, "training_seconds": 181.46486658300273, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223508308270226}

RECENT RESULT
hypothesis: An 80-unit GRU with all 32 frames and the mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.5% versus the verified 82-unit model.
change: Reduce the GRU and summary width from 82 to 80 and resize the concatenated classifier input from 164 to 160, leaving training and temporal coverage unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 82-unit model achieved 86.01% accuracy at 655.47M total MACs, and every tested dual-readout width from 82 through 91 passed, making the next two-unit reduction the most direct probe of the minimum viable width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 68.48943950003013, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4433465594894316}

RECENT RESULT
hypothesis: An 81-unit GRU using all 32 frames and the mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.2% versus the verified 82-unit model.
change: Reduce the GRU state and summary width from 82 to 81 and resize the concatenated classifier input from 164 to 162, preserving temporal coverage and training.
mechanism: One-unit dual-timescale GRU width compression
evidence_used: The 82-unit model passed at 86.01% while the 80-unit model narrowly failed at 84.29%; testing 81 units directly resolves the remaining minimum-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3344271697638578933, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 640081440, "recurrent_steps": 26080, "total_inference_macs": 641137680, "training_seconds": 111.35484620905481, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.43995596294754125}

RECENT RESULT
hypothesis: The 81-unit GRU processing 31 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1% versus the verified 32-step model.
change: Reduce the causal frame schedule from 32 to 31 unique increasing indices while preserving both recording endpoints, the mean-plus-terminal readout, and training procedure.
mechanism: Single-frame endpoint-preserving temporal compression
evidence_used: The 81-unit 32-step model passed at 85.64% accuracy, while 24-frame processing has previously passed at higher width; with the recurrent-width boundary already isolated between 80 and 81 units, removing one frame is the most cautious probe of the remaining temporal redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3239935379229828373, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 620078895, "recurrent_steps": 25265, "total_inference_macs": 621135135, "training_seconds": 71.20170233398676, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4404342464142782}

RECENT RESULT
hypothesis: An 82-unit GRU processing 30 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.1% versus the verified 81-unit, 31-step model.
change: Increase the GRU and readout width from 81 to 82 while reducing the causal schedule from 31 to 30 frames.
mechanism: Width-for-time Pareto trade
evidence_used: The 82-unit, 32-step model reached 86.01% accuracy, while the 81-unit, 31-step model narrowly passed at 85.15%; trading one additional hidden unit for one fewer recurrent step is the smallest unexplored diagonal reduction and still lowers exact dense MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3205683750709968874, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 26944, "peak_hidden_elements": 84480, "recurrent_macs": 613499400, "recurrent_steps": 24450, "total_inference_macs": 614568680, "training_seconds": 98.14459629217163, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44018247639474695}

RECENT RESULT
hypothesis: An 83-unit GRU processing 29 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.2% versus the verified 82-unit, 30-step model.
change: Increase the GRU and dual-timescale readout width from 82 to 83 while reducing the causal schedule from 30 to 29 frames.
mechanism: Width-for-time Pareto continuation
evidence_used: The adjacent diagonal designs—81 units at 31 steps and 82 units at 30 steps—both passed at 85.15% and 85.52%, respectively; continuing that width-for-time trade is the smallest evidence-supported reduction in exact dense MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167504046643746721, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27521, "peak_hidden_elements": 85504, "recurrent_macs": 606166845, "recurrent_steps": 23635, "total_inference_macs": 607249165, "training_seconds": 89.7734805829823, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4487377236957199}

RECENT RESULT
hypothesis: An 84-unit GRU processing 28 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.3% versus the verified 83-unit, 29-step model.
change: Increase the GRU and dual-timescale readout width from 83 to 84 while reducing the causal schedule from 29 to 28 frames.
mechanism: Width-for-time Pareto continuation
evidence_used: The adjacent diagonal designs—81 units at 31 steps, 82 units at 30 steps, and 83 units at 29 steps—all passed; the latest reached 86.01% accuracy, supporting one more incremental width-for-time trade.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3125319746078960044, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 598066560, "recurrent_steps": 22820, "total_inference_macs": 599161920, "training_seconds": 76.79220579215325, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4281637367295341}

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
    """A one-layer causal GRU with mean and terminal-state readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
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
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        base_steps = min(28, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 27:
            del indices[1]
        return indices


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


