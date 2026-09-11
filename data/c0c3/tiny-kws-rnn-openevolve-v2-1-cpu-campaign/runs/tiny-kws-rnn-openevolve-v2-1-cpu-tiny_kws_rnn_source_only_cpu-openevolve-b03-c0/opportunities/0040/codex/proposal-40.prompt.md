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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 64.51925350003876, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.509380672899492}
prior_hypothesis: Retaining frame 28 while omitting adjacent frame 27 will recover at least 85% accuracy at 25 recurrent steps because frame 28 distinguished the passing 26-step schedule from the narrowly failing truncation.

## Recent verification evidence

RECENT RESULT
hypothesis: Omitting only input frame 0 will preserve at least 85% validation accuracy while reducing recurrent execution from 32 to 31 steps and cutting recurrent MACs by 3.125%.
change: Keep the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged, but schedule frames 1–31 for 32-frame inputs.
mechanism: Earliest-frame recurrent step pruning
evidence_used: The current design passes at 85.52% accuracy; removing one recurrent step preserves all learned dimensions and the remaining 31-frame causal sequence, while offering a larger structural MAC reduction than another single-feature or classifier-coordinate reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4497289405334864312, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 861637560, "recurrent_steps": 25265, "total_inference_macs": 862185240, "training_seconds": 58.394637958845124, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4544558659652991}

RECENT RESULT
hypothesis: Omitting frames 0 and 1 will retain at least 85% validation accuracy while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 3.23% versus the current design.
change: Schedule frames 2–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental earliest-frame recurrent step pruning
evidence_used: Omitting frame 0 improved accuracy from 85.52% to 85.64% while reducing execution to 31 steps, indicating that the earliest context is expendable and motivating an incremental test of the adjacent frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4352307707814887137, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 833842800, "recurrent_steps": 24450, "total_inference_macs": 834390480, "training_seconds": 96.66559787490405, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4559282361364072}

RECENT RESULT
hypothesis: Omitting frames 0–2 will retain at least 85% validation accuracy while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 3.33%.
change: Schedule frames 3–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental earliest-frame recurrent step pruning
evidence_used: Successive removal of frame 0 and then frame 1 preserved passing accuracy at 85.64% and 85.40%, respectively, while each reduced recurrent execution by one step; removing the next-earliest frame is the most direct test of the remaining temporal redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4207326010294909962, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 806048040, "recurrent_steps": 23635, "total_inference_macs": 806595720, "training_seconds": 88.400646999944, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45559915179855254}

RECENT RESULT
hypothesis: Omitting frames 0–3 will retain at least 85% validation accuracy while reducing recurrent execution from 29 to 28 steps and recurrent MACs by approximately 3.45%.
change: Schedule frames 4–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental earliest-frame recurrent step pruning
evidence_used: Removing each of the first three frames successively preserved passing accuracy, with the latest 29-step model reaching 85.52%; this supports testing whether the next-earliest frame is also redundant.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4062344312774932787, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 778253280, "recurrent_steps": 22820, "total_inference_macs": 778800960, "training_seconds": 83.42982604191639, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4741135860513324}

RECENT RESULT
hypothesis: Omitting frames 0–2 and the final frame will retain at least 85% validation accuracy while matching the failed 28-step model’s lower inference cost, because it restores frame 3 and removes a likely trailing-context frame instead.
change: Run the verified recurrent model on frames 3 through 30 for standard 32-frame inputs, preserving 28 causal recurrent steps.
mechanism: Opposite-end frame pruning
evidence_used: The 29-step schedule omitting frames 0–2 passed at 85.52%, while additionally omitting frame 3 narrowly failed at 84.79%; relocating the fourth omission to the opposite temporal boundary directly tests whether frame 3 is more informative than the final frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4062344312774932787, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 778253280, "recurrent_steps": 22820, "total_inference_macs": 778800960, "training_seconds": 68.7644304591231, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4670081390193635}

RECENT RESULT
hypothesis: Omitting frame 30 in addition to frames 0–2 and 31 will retain at least 85% validation accuracy while reducing recurrent execution from 28 to 27 steps and recurrent MACs by approximately 3.57%.
change: Schedule frames 3–29 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental trailing-frame recurrent step pruning
evidence_used: The 29-step schedule using frames 3–31 passed at 85.52%; removing the final frame also preserved 85.52%, whereas removing frame 3 failed at 84.79%. This indicates trailing context is the better-supported target for the next step reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3917362615254955612, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 750458520, "recurrent_steps": 22005, "total_inference_macs": 751006200, "training_seconds": 63.025688665919006, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4790260829808522}

RECENT RESULT
hypothesis: Omitting frame 29 in addition to frames 0–2 and 30–31 will retain at least 85% validation accuracy while reducing recurrent execution from 27 to 26 steps and recurrent MACs by approximately 3.70%.
change: Schedule frames 3–28 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental trailing-frame recurrent step pruning
evidence_used: Successively removing trailing frames 31 and 30 preserved passing accuracy at 85.52% and 85.28%, respectively, while removing frame 3 failed; this makes the next trailing frame the best-supported target for another step reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3772380917734978437, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 722663760, "recurrent_steps": 21190, "total_inference_macs": 723211440, "training_seconds": 70.6754772500135, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4919525708157592}

RECENT RESULT
hypothesis: Omitting frame 28 in addition to frames 0–2 and 29–31 will retain at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and recurrent MACs by approximately 3.85%.
change: Schedule frames 3–27 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental trailing-frame recurrent step pruning
evidence_used: Successive removal of trailing frames 31, 30, and 29 preserved passing accuracy, with the current 26-step model reaching 85.40%; this makes the next trailing frame the most directly supported target for further cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 60.87797804106958, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.5098119255955234}

RECENT RESULT
hypothesis: Retaining frame 28 while omitting the locally redundant middle frame 16 will recover at least 85% accuracy at the lower 25-step inference cost.
change: Change the 32-frame schedule from contiguous frames 3–28 to frames 3–15 and 17–28, leaving the model and training procedure unchanged.
mechanism: Interior-frame temporal thinning
evidence_used: The 26-step schedule over frames 3–28 reached 85.40%, while truncating it to frames 3–27 missed the threshold by only one validation example; preserving the later endpoint and relocating the omission to an interior frame tests whether temporal coverage matters more than one densely sampled middle frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 57.50335258315317, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.5045694082061206}

RECENT RESULT
hypothesis: Retaining frame 28 while omitting adjacent frame 27 will recover at least 85% accuracy at 25 recurrent steps because frame 28 distinguished the passing 26-step schedule from the narrowly failing truncation.
change: Use frames 3–26 and 28 for standard 32-frame inputs, reducing execution from 26 to 25 steps without changing model dimensions or training.
mechanism: Penultimate-frame pruning
evidence_used: Frames 3–28 achieved 85.40%, while removing frame 28 scored 84.91%, only one validation example below threshold; this motivates preserving frame 28 and pruning its temporally redundant predecessor instead.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 64.51925350003876, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.509380672899492}

RECENT RESULT
hypothesis: Omitting frames 25 and 27 while retaining frames 26 and 28 will preserve at least 85% validation accuracy with 24 recurrent steps, because spaced omissions maintain late temporal coverage better than contiguous tail truncation.
change: Change the 32-frame schedule from frames 3–26 and 28 to frames 3–24, 26, and 28, reducing execution from 25 to 24 steps without changing model capacity or training.
mechanism: Alternating late-tail temporal thinning
evidence_used: The 25-step design passed only when frame 28 was retained and adjacent frame 27 was omitted; removing frame 28 narrowly failed, motivating another alternating omission in the locally redundant late region while preserving frames 26 and 28.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482417522695024087, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 667074240, "recurrent_steps": 19560, "total_inference_macs": 667621920, "training_seconds": 60.96644412493333, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.5228541181131374}

RECENT RESULT
hypothesis: Reducing the classifier input from 96 to 95 recurrent-summary coordinates will preserve at least 85% validation accuracy while lowering total inference MACs at the successful 25-step schedule.
change: Keep the GRU, feature pooling, and frame schedule unchanged, but remove one classifier input coordinate.
mechanism: Incremental recurrent-readout coordinate pruning
evidence_used: The current 25-step design passes at 85.03%, whereas the tested 24-step schedule falls to 84.66%; a one-coordinate readout reduction is a smaller structural change that improves exact MAC cost without further reducing temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627369462066922750, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35400, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695410975, "training_seconds": 58.80939258309081, "validation_accuracy": 0.8294478527607362, "validation_cross_entropy": 0.5300466712998466}



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
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        pooled_frame = torch.cat(
            (
                frame[:, :2].mean(dim=1, keepdim=True),
                frame[:, 2:18],
                frame[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        output, hidden = self.gru(
            self.input_norm(pooled_frame).unsqueeze(1),
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
        pooled_frames = torch.cat(
            (
                frames[:, :, :2].mean(dim=2, keepdim=True),
                frames[:, :, 2:18],
                frames[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        outputs, hidden = self.gru(
            self.input_norm(pooled_frames), hidden.transpose(0, 1).contiguous()
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
        averaged = summary / count.clamp_min(1.0)
        logits = self.classifier(averaged[:, :96])
        return torch.cat((logits, -logits.sum(dim=1, keepdim=True)), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 8:
            return list(range(available_frames))
        schedule = list(range(3, available_frames - 3))
        del schedule[-2]
        return schedule


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


