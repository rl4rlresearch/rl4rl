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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3682664352605295548, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 705411840, "recurrent_steps": 22820, "total_inference_macs": 706011680, "training_seconds": 115.29540041694418, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4833382167699147}
prior_hypothesis: Processing frames 4–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.4% versus the 29-step design.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.
change: Omit frame 1 in addition to frame 0, producing an explicit 30-step causal schedule while preserving model capacity and training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: The 31-step model achieved 85.40% accuracy; prior 30-step attempts either timed out or accidentally reproduced the 31-step implementation, so the temporal-compression boundary remains unresolved.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.
change: Change the explicit causal schedule from frames 1–31 to frames 2–31, yielding 30 recurrent steps while preserving model capacity and training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: The 31-step model achieved 85.40% accuracy; previous intended 30-step tests either timed out or reproduced the unchanged 31-step implementation, so this remains the smallest unresolved temporal-compression test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3945488316597653338, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 755798400, "recurrent_steps": 24450, "total_inference_macs": 756398240, "training_seconds": 109.3020062497817, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4675700837117763}

RECENT RESULT
hypothesis: Processing frames 3–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.3% versus the 30-step design.
change: Omit frame 2 in addition to frames 0–1, producing a 29-step causal schedule without changing model capacity or training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: The 92-unit GRU retained 85.52% accuracy after progressive leading-frame pruning reduced execution from 32 to 30 steps, indicating that the earliest boundary frames are lower-value and motivating the smallest remaining temporal reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3814076334601474443, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 730605120, "recurrent_steps": 23635, "total_inference_macs": 731204960, "training_seconds": 153.4947164170444, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.46314061638767734}

RECENT RESULT
hypothesis: Processing frames 4–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.4% versus the 29-step design.
change: Omit frame 3 in addition to frames 0–2, producing a 28-step causal schedule without changing model capacity or training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: Progressive leading-frame pruning from 32 to 29 steps preserved or improved accuracy, with the 29-step model reaching 85.77%; this supports testing the smallest further temporal reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3682664352605295548, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 705411840, "recurrent_steps": 22820, "total_inference_macs": 706011680, "training_seconds": 115.29540041694418, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4833382167699147}

RECENT RESULT
hypothesis: Processing frames 5–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.6% versus the 28-step design.
change: Omit frame 4 in addition to frames 0–3, producing a 27-step causal schedule without changing model capacity or training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: Progressive leading-frame pruning from 32 to 28 steps has consistently retained at least 85% accuracy; the current 28-step model achieves 85.40%, making one additional leading-frame removal the smallest informative test of the temporal-compression boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3551252370609116653, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 680218560, "recurrent_steps": 22005, "total_inference_macs": 680818400, "training_seconds": 149.98637083289213, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.5108104682407496}

RECENT RESULT
hypothesis: Seven causal four-frame memory segments with learned frame embeddings and sum/max statistics will retain at least 85% accuracy while reducing total inference MACs by roughly 55% versus the 28-step GRU.
change: Replace dense hidden-to-hidden GRU updates with a two-layer per-frame encoder that recurrently accumulates coarse temporal sum and maximum memories, then predicts from all ordered segments with a small nonlinear head.
mechanism: Nonlinear segmented recurrent memory pooling
evidence_used: The 92-unit GRU remained accurate through 28 steps but failed only when a fifth leading frame was removed, suggesting that preserving temporal coverage matters more than repeatedly applying three dense recurrent gates; the new design keeps the verified 28-frame schedule while challenging that gate-heavy assumption.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1648771452391748084, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 99664, "peak_hidden_elements": 459264, "recurrent_macs": 245360640, "recurrent_steps": 22820, "total_inference_macs": 316089600, "training_seconds": 45.82743033301085, "validation_accuracy": 0.8147239263803681, "validation_cross_entropy": 0.5251271347326735}

RECENT RESULT
hypothesis: Processing frames 4–30 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.6% versus the 28-step design.
change: Preserve the earliest frame retained by the successful 28-step model while omitting the final frame, producing a 27-step causal schedule.
mechanism: Trailing-boundary frame pruning
evidence_used: Frames 4–31 achieved 85.40% accuracy, whereas frames 5–31 fell to 83.68%; this suggests frame 4 may be load-bearing and makes removing the opposite boundary frame the most informative 27-step test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3551252370609116653, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 680218560, "recurrent_steps": 22005, "total_inference_macs": 680818400, "training_seconds": 82.2063117090147, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4935036314045725}

RECENT RESULT
hypothesis: Reducing the verified 28-step GRU from 92 to 91 hidden units will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2%.
change: Keep the successful frames 4–31 schedule and training procedure, but structurally reduce the GRU state and classifier input width to 91.
mechanism: Incremental recurrent-width compression
evidence_used: The 92-unit, 28-step model achieved 85.40%, while both tested 27-step schedules failed; this motivates preserving temporal coverage and probing the smallest possible dense recurrent-width reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3610139494573967025, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31625, "peak_hidden_elements": 93696, "recurrent_macs": 691514460, "recurrent_steps": 22820, "total_inference_macs": 692107780, "training_seconds": 107.99345216713846, "validation_accuracy": 0.8331288343558282, "validation_cross_entropy": 0.5072902000754889}

RECENT RESULT
hypothesis: Processing 27 frames while retaining both boundary frames 4 and 31 and omitting only redundant adjacent frame 5 will preserve at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.6% versus the verified 28-step model.
change: Keep the verified 92-unit GRU and training procedure, but change the schedule from frames 4–31 to frames 4 and 6–31.
mechanism: Edge-span-preserving interior frame thinning
evidence_used: Both 27-step boundary-pruning schedules failed, while frames 4–31 achieved 85.40%; this suggests preserving the full successful temporal span and testing redundancy within that span is more promising than removing either endpoint.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3551252370609116653, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 680218560, "recurrent_steps": 22005, "total_inference_macs": 680818400, "training_seconds": 87.55411537503824, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4959009650294766}

RECENT RESULT
hypothesis: Dropping the lowest log-mel band while preserving the verified 92-unit GRU and 28-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.9%.
change: Change the recurrent input width from 20 to 19 and feed bands 1–19 to both recurrent execution paths, leaving temporal coverage, hidden capacity, and training unchanged.
mechanism: Low-frequency boundary-band pruning
evidence_used: The verified 28-step, 92-unit model achieved 85.40%, while reducing either recurrent steps or hidden width failed; pruning one frequency-boundary input feature tests a distinct structural cost axis without sacrificing the temporal coverage or recurrent capacity those failures indicate are important.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3649811357126625750, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31970, "peak_hidden_elements": 94720, "recurrent_macs": 699113520, "recurrent_steps": 22820, "total_inference_macs": 699713360, "training_seconds": 99.3089335418772, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4852116309791986}

RECENT RESULT
hypothesis: Dropping only the highest log-mel band will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.9% versus the verified 20-band, 28-step model.
change: Reduce the GRU input width from 20 to 19 and feed bands 0–18 through both recurrent execution paths, preserving hidden width, temporal schedule, classifier, and training.
mechanism: High-frequency boundary-band pruning
evidence_used: Dropping the lowest band narrowly missed the target at 84.79%, showing that single-band structural pruning is near the feasible boundary; testing the opposite spectral boundary is the most direct way to determine whether the lowest-frequency information was specifically load-bearing.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3649811357126625750, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31970, "peak_hidden_elements": 94720, "recurrent_macs": 699113520, "recurrent_steps": 22820, "total_inference_macs": 699713360, "training_seconds": 89.86903308401816, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.47585936002204754}

RECENT RESULT
hypothesis: Removing the highest mel band while reducing label smoothing from 0.03 to 0.02 will recover at least one validation decision and reach 85% accuracy, while retaining the previously measured reduction to approximately 699.7M total MACs.
change: Use bands 0–18 with a 19-input GRU in both recurrent paths and slightly reduce label smoothing to strengthen class margins.
mechanism: High-frequency band pruning with sharper supervision
evidence_used: The prior high-band-pruned model reached 84.91% accuracy—just below the requirement—while improving validation cross-entropy from 0.4833 to 0.4759 and reducing total MACs from 706.0M to 699.7M, indicating that its representation remained strong enough for a small loss adjustment to plausibly cross the threshold.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3649811357126625750, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31970, "peak_hidden_elements": 94720, "recurrent_macs": 699113520, "recurrent_steps": 22820, "total_inference_macs": 699713360, "training_seconds": 85.23320354195312, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4718180276133531}



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
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
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
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))


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


