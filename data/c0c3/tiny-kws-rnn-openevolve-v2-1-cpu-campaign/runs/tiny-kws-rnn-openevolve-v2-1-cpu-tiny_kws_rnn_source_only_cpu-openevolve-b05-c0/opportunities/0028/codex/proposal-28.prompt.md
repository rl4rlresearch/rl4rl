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
verified_results: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1446365031245192291, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 522, "p95_recurrent_steps": 25, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277285820, "training_seconds": 51.014770542038605, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43589297803633054}
prior_hypothesis: Processing frames 2–23, 25, 27, and 29 will achieve at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps.

## Recent verification evidence

RECENT RESULT
hypothesis: A 58-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.9% versus the verified 60-unit model.
change: Reduce the GRU hidden width from 60 to 58 and resize the classifier and recurrent state tensors accordingly.
mechanism: Two-unit recurrent-width refinement with full temporal coverage
evidence_used: The 60-unit full-sequence model achieved 86.50% accuracy, while the attempted 56-unit model produced no accuracy evidence because training timed out; testing the midpoint isolates a smaller, lower-risk structural reduction while retaining the successful full-frame dual readout.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1850242617537167856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 786, "p95_recurrent_steps": 32, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 353957760, "recurrent_steps": 26080, "total_inference_macs": 354714080, "training_seconds": 207.61574637517333, "validation_accuracy": 0.8846625766871166, "validation_cross_entropy": 0.37396759782100747}

RECENT RESULT
hypothesis: A 58-unit GRU trained with batch size 96 will retain at least 85% validation accuracy, finish within the verification time limit, and reduce total inference MACs by approximately 5.7% versus the verified 60-unit model.
change: Reduce the GRU and temporal-summary width from 60 to 58, resize the classifier accordingly, and increase training batch size from 64 to 96 to reduce optimizer-step overhead.
mechanism: Throughput-assisted recurrent-width reduction
evidence_used: The 58-unit, batch-64 run reached 88.47% accuracy and 354,714,080 inference MACs, but its 207.6-second training time exceeded the limit; prior larger-batch runs completed substantially faster.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1850242617537167856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 522, "p95_recurrent_steps": 32, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 353957760, "recurrent_steps": 26080, "total_inference_macs": 354714080, "training_seconds": 108.16636183299124, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4278718913259682}

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 31 frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1%.
change: Omit only the earliest input frame while preserving the remaining contiguous causal sequence, model width, optimizer, and training procedure.
mechanism: Single-frame causal prefix trimming
evidence_used: The current 58-unit, batch-96 model achieved 85.77% accuracy over all 32 frames; prior 26–27-frame reductions were unreliable, motivating the smallest possible temporal reduction at the likely low-information recording boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1792545819495457061, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 522, "p95_recurrent_steps": 31, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 342896580, "recurrent_steps": 25265, "total_inference_macs": 343652900, "training_seconds": 78.78889991599135, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.427536357107338}

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 30 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.2% versus the verified 31-frame model.
change: Omit the first two input frames while preserving the contiguous final 30-frame sequence and all training settings.
mechanism: Incremental causal prefix trimming
evidence_used: Omitting only the earliest frame retained 85.77% accuracy and reduced execution to 31 steps; one additional prefix-frame removal is the smallest and most informative temporal-cost reduction, though the 0.77-point margin makes it a boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1734849021453746266, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 522, "p95_recurrent_steps": 30, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 331835400, "recurrent_steps": 24450, "total_inference_macs": 332591720, "training_seconds": 105.38211370795034, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4365738570324482}

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 29 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.3% versus the verified 30-frame model.
change: Omit the first three input frames while preserving the contiguous final 29-frame sequence and all training settings.
mechanism: Incremental causal prefix trimming
evidence_used: The final-30-frame model achieved 85.52% accuracy after the 31-frame model achieved 85.77%; removing one additional earliest frame is the smallest structural cost reduction and directly tests the remaining temporal boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1677152223412035471, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 522, "p95_recurrent_steps": 29, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 320774220, "recurrent_steps": 23635, "total_inference_macs": 321530540, "training_seconds": 68.43077654205263, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.447176777658287}

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 28 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.4% versus the verified 29-frame model.
change: Omit the first four input frames while preserving the contiguous final 28-frame sequence and all other model and training settings.
mechanism: Incremental causal prefix trimming
evidence_used: The final-29-frame model achieved 85.28% accuracy after successive one-frame prefix trims through 31 and 30 frames remained viable; removing one additional earliest frame is the smallest direct test of the remaining temporal boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1619455425370324676, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 522, "p95_recurrent_steps": 28, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 309713040, "recurrent_steps": 22820, "total_inference_macs": 310469360, "training_seconds": 74.87525541614741, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4737021978647431}

RECENT RESULT
hypothesis: A 59-unit GRU processing the final 28 frames will recover validation accuracy to at least 85% while reducing total inference MACs by approximately 0.5% and recurrent steps from 29 to 28 versus the verified 58-unit, 29-frame model.
change: Increase recurrent and summary width from 58 to 59 units, resize the classifier, and omit the first four frames.
mechanism: Near-isocompute width–time reallocation
evidence_used: The 58-unit, 28-frame model narrowly missed the target at 84.66%, while 29 frames achieved 85.28%; adding one hidden unit is the smallest capacity increase and still leaves the 28-step recurrent computation below the current model’s MAC count.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1668445839435566769, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 522, "p95_recurrent_steps": 28, "parameters": 15329, "peak_hidden_elements": 60928, "recurrent_macs": 319092060, "recurrent_steps": 22820, "total_inference_macs": 319861420, "training_seconds": 59.07418387499638, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.4990342403482074}

RECENT RESULT
hypothesis: Processing frames 2–29 will achieve at least 85% validation accuracy while matching the failed final-28-frame model’s 310,469,360 MACs and 28 recurrent steps.
change: Replace four-frame prefix trimming with two-frame trimming at each boundary, preserving 28 contiguous central frames.
mechanism: Symmetric boundary trimming
evidence_used: Frames 2–31 achieved 85.52% accuracy, whereas removing two additional leading frames fell narrowly to 84.66%; removing the likely redundant final two frames instead directly tests whether retaining earlier command onset information recovers accuracy at identical cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1619455425370324676, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 522, "p95_recurrent_steps": 28, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 309713040, "recurrent_steps": 22820, "total_inference_macs": 310469360, "training_seconds": 65.47450574999675, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4348838946570648}

RECENT RESULT
hypothesis: Processing frames 2–28 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.6% versus the verified 28-frame model.
change: Preserve the informative starting frame and omit one additional trailing frame, producing a contiguous 27-frame schedule.
mechanism: Incremental trailing-boundary trimming
evidence_used: Frames 2–29 achieved 86.50% accuracy at 28 steps, outperforming the 29-step final-frame schedule; this suggests earlier command-onset information is more valuable than the recording tail and motivates the smallest further trailing trim.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1561758627328613881, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 522, "p95_recurrent_steps": 27, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 298651860, "recurrent_steps": 22005, "total_inference_macs": 299408180, "training_seconds": 55.067767082946375, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.44136404961896086}

RECENT RESULT
hypothesis: Processing frames 2–27 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.7% versus the verified 27-step model.
change: Omit one additional trailing frame, producing a contiguous 26-frame schedule while preserving the model and training procedure.
mechanism: Incremental trailing-boundary trimming
evidence_used: Frames 2–28 achieved 86.26% accuracy after frames 2–29 achieved 86.50%, indicating that incremental removal from the less-informative recording tail has preserved a meaningful margin above the target.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1504061829286903086, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 522, "p95_recurrent_steps": 26, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 287590680, "recurrent_steps": 21190, "total_inference_macs": 288347000, "training_seconds": 60.41266495781019, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4519053921377732}

RECENT RESULT
hypothesis: Processing frames 2–26 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.8% versus the verified 26-step model.
change: Omit one additional trailing frame, producing a contiguous 25-frame schedule while preserving model capacity and training procedure.
mechanism: Incremental trailing-boundary trimming
evidence_used: Frames 2–27 achieved 85.15% accuracy after successive trailing trims at 28 and 27 endpoints also met the target; this is the smallest structural reduction that directly tests the remaining temporal boundary, though the narrow margin makes it a high-information threshold test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1446365031245192291, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 522, "p95_recurrent_steps": 25, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277285820, "training_seconds": 54.23268970893696, "validation_accuracy": 0.8331288343558282, "validation_cross_entropy": 0.46658252177794285}

RECENT RESULT
hypothesis: Processing frames 2–23, 25, 27, and 29 will achieve at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps.
change: Replace the contiguous 26-frame schedule with 25 frames spanning the stronger frames-2–29 window, omitting alternating late frames.
mechanism: Sparse tail sampling with preserved temporal coverage
evidence_used: Frames 2–29 achieved 86.50%, while contiguous frames 2–26 fell to 83.31%; preserving the later endpoint and broader temporal span directly tests whether coverage, rather than the 25-step count itself, caused that failure.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1446365031245192291, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 522, "p95_recurrent_steps": 25, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277285820, "training_seconds": 51.014770542038605, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43589297803633054}



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

BATCH_SIZE = 96
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
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
        if available_frames >= 8:
            full_window = list(range(2, available_frames - 2))
            return full_window[:-6] + full_window[-5::2]
        return list(range(available_frames))


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


