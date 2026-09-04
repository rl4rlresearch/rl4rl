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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1413341989043073665, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 269928000, "recurrent_steps": 18745, "total_inference_macs": 270954900, "training_seconds": 31.087566582951695, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.460217977886551}
prior_hypothesis: Processing frames 3–22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing total dense inference MACs from 282,690,900 to approximately 270,954,900.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1536540722251086223, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294573600, "training_seconds": 56.65610625012778, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4427040802189178}
prior_hypothesis: A 60-unit GRU processing frames 3–26 and frame 28 will achieve at least 85% validation accuracy while reducing total dense inference MACs from 306,309,600 to approximately 294,573,600.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352125227200076850, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259218900, "training_seconds": 36.48290670802817, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4870919163241708}
prior_hypothesis: Processing frames 3–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 22 recurrent steps and approximately 259,218,900 total MACs.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067295, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 30.942437916994095, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4581207088166219}
prior_hypothesis: Disabling LayerNorm affine parameters and the seven-logit classifier bias will retain at least 85% validation accuracy at 294,426,900 MACs while reducing learned parameters from 16,067 to 16,020.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the lowest mel band only from the GRU reset-gate input will retain at least 85% accuracy while reducing total inference MACs from 294,426,900 to approximately 293,204,400.
change: Replace the fused GRU with an equivalent Linear-based GRU whose update and candidate gates retain all 20 bands while its reset gate uses 19; also adopt the qualified bias-free seven-logit head.
mechanism: Gate-selective spectral pruning
evidence_used: Removing the lowest band from every GRU gate narrowly missed at 84.42%; retaining that band in the update and candidate paths tests a substantially smaller structural reduction, while the bias-free head already qualified at 85.52%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1529398766712244735, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15960, "peak_hidden_elements": 92672, "recurrent_macs": 292177500, "recurrent_steps": 20375, "total_inference_macs": 293204400, "training_seconds": 64.02440058393404, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.483909878408982}

RECENT RESULT
hypothesis: Processing frames 3–25 plus frame 28 will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 282,690,900.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, then remove frame 26 from the proven endpoint-preserving schedule for 24 recurrent steps.
mechanism: Iterative late-frame pruning with endpoint preservation
evidence_used: The bias-free 25-step reference achieved 85.52% accuracy, and removing frame 27 while preserving frame 28 previously qualified; pruning the next-latest contiguous frame is the most direct test of further temporal redundancy without disrupting the important endpoint or early continuity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1474558750886070480, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 281664000, "recurrent_steps": 19560, "total_inference_macs": 282690900, "training_seconds": 39.57340499991551, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4616131928800805}

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–24, 26, and 28 will achieve at least 85% validation accuracy while reducing total inference MACs to approximately 282,690,900.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, and use a 24-step schedule that retains informative frame 26 while distributing two-frame gaps across the late sequence.
mechanism: Distributed late-frame pruning with endpoint preservation
evidence_used: The qualified 25-step design reached 85.52%; the 24-step frames 3–25 plus 28 variant narrowly missed at 84.79%. Replacing frame 25 with later frame 26 preserves the proven endpoint and improves late-time coverage without adding MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1474558750886070480, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 281664000, "recurrent_steps": 19560, "total_inference_macs": 282690900, "training_seconds": 36.8413354170043, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4530906724052195}

RECENT RESULT
hypothesis: Processing frames 3–22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing total dense inference MACs from 282,690,900 to approximately 270,954,900.
change: Remove frame 23 from the qualified 24-step schedule, yielding 23 recurrent steps while preserving evenly spaced late observations and the important frame-28 endpoint.
mechanism: Distributed alternating late-frame pruning
evidence_used: The 24-step schedule retaining frames 26 and 28 qualified at 85.15%, whereas clustering the omitted late frames missed at 84.79%; this motivates extending the successful distributed-pruning pattern by retaining frames 24, 26, and 28.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1413341989043073665, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 269928000, "recurrent_steps": 18745, "total_inference_macs": 270954900, "training_seconds": 31.087566582951695, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.460217977886551}

RECENT RESULT
hypothesis: Processing frames 3–20, 22, 24, 26, and 28 will retain at least 85% validation accuracy while reducing recurrent execution from 23 to 22 steps and total dense inference MACs from 270,954,900 to approximately 259,218,900.
change: Replace the current 25-step schedule with a 22-step schedule that preserves the continuous early context and samples every second late frame through the proven frame-28 endpoint.
mechanism: Extended alternating late-frame pruning
evidence_used: The 23-step frames 3–22, 24, 26, and 28 schedule achieved 85.52% accuracy; the prior successful transition from 24 to 23 steps showed that extending distributed late-frame pruning can remove a step without losing qualification.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352125227200076850, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259218900, "training_seconds": 31.1265710410662, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4731880047569977}

RECENT RESULT
hypothesis: Processing frames 3–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 22 recurrent steps and approximately 259,218,900 total MACs.
change: Remove frame 22 instead of frame 21 from the qualified 23-step schedule, preserve the proven late frames 24, 26, and 28, and remove the seven-logit classifier bias.
mechanism: Early-continuity-preserving temporal pruning
evidence_used: The 23-step schedule reached 85.52% accuracy, while the 22-step schedule that omitted frame 21 but retained frame 22 narrowly missed at 84.66%; preserving uninterrupted early recurrence through frame 21 tests whether continuity is more valuable than the final contiguous frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352125227200076850, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259218900, "training_seconds": 36.48290670802817, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4870919163241708}

RECENT RESULT
hypothesis: Processing frames 4–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution to 21 recurrent steps and approximately 247,482,900 total MACs.
change: Adopt the qualified non-affine normalization and bias-free seven-logit head, then remove only frame 3 from the qualified 22-step schedule while preserving continuous frames through 21 and the proven late frames 24, 26, and 28.
mechanism: Earliest-boundary temporal pruning
evidence_used: The 22-step frames 3–21, 24, 26, and 28 design achieved 85.40%; replacing frame 21 with frame 22 missed at 84.66%, and prior removal of frame 26 also missed, motivating preservation of the later contiguous context and all three late observations while pruning the earliest selected boundary frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1290908465357080035, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 246456000, "recurrent_steps": 17115, "total_inference_macs": 247482900, "training_seconds": 27.439611959038302, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.5186664604701878}

RECENT RESULT
hypothesis: Exiting examples whose penultimate-step posterior is at least 0.95 will retain at least 85% validation accuracy and reduce total inference MACs whenever more than 8.75% of examples exit before the final GRU step.
change: Evaluate the learned classifier only at steps 21 and 22, and add a high-confidence exit after step 21 while preserving the qualified 22-frame schedule and final classifier behavior.
mechanism: Confidence-gated penultimate early exit
evidence_used: The qualified 22-step design reached 85.40%, while unconditional removal of one step fell to 84.29%; selectively skipping the final step only for already-confident examples is a narrower temporal reduction that preserves all 22 steps for uncertain examples.
result: the implementation could not be verified

RECENT RESULT
hypothesis: The qualified 22-step schedule will retain at least 85% validation accuracy when one final-hidden coordinate is omitted from the classifier, reducing total inference MACs from 259,218,900 to 259,213,195 and parameters from 16,020 to 16,013.
change: Adopt the qualified frames 3–21, 24, 26, and 28 schedule and reduce the bias-free classifier input from 180 to 179 features by excluding one coordinate only from its final-hidden branch.
mechanism: Single-coordinate classifier-path pruning
evidence_used: The 22-step design achieved 85.40% accuracy at 259,218,900 MACs, while removing seven classifier biases preserved qualification. This equally small seven-weight reduction retains both complete mean and maximum summaries and 59 of 60 final-hidden coordinates, avoiding the aggressive recurrent-input and classifier-feature reductions that failed.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1352095469051998338, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 16013, "peak_hidden_elements": 92672, "recurrent_macs": 258192000, "recurrent_steps": 17930, "total_inference_macs": 259213195, "training_seconds": 30.330491374945268, "validation_accuracy": 0.8269938650306748, "validation_cross_entropy": 0.502046325309145}

RECENT RESULT
hypothesis: Processing frames 3–21, 26, and 28 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total dense inference MACs to approximately 247,482,900.
change: Remove frame 24 from the qualified 22-step schedule, preserving uninterrupted early recurrence through frame 21 and the proven late frames 26 and 28.
mechanism: Late-bridge frame pruning
evidence_used: The 22-step frames 3–21, 24, 26, and 28 design achieved 85.40%, while removing frame 3 failed and prior evidence identifies frames 26 and 28 as important; frame 24 is therefore the most informative remaining single-frame pruning candidate.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1290908465357080035, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 246456000, "recurrent_steps": 17115, "total_inference_macs": 247482900, "training_seconds": 31.302681958070025, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.5169063638324386}

RECENT RESULT
hypothesis: A 59-unit GRU using the qualified 22-frame schedule will retain at least 85% validation accuracy while reducing total dense inference MACs from 259,218,900 to approximately 251,724,975.
change: Adopt the qualified non-affine normalization, bias-free seven-logit head, and frames 3–21, 24, 26, and 28 schedule, while reducing GRU width from 60 to 59 units.
mechanism: Minimal recurrent-width contraction
evidence_used: The 60-unit 22-step design achieved 85.40% accuracy, while tested 21-step schedules failed; preserving all qualified observations and making the smallest possible recurrent-width reduction targets MACs without discarding additional temporal or spectral information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1313035774116954481, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 15576, "peak_hidden_elements": 91136, "recurrent_macs": 250715190, "recurrent_steps": 17930, "total_inference_macs": 251724975, "training_seconds": 44.02463495801203, "validation_accuracy": 0.8294478527607362, "validation_cross_entropy": 0.5033503058497891}

RECENT RESULT
hypothesis: Processing frames 3–11, 13–21, 24, 26, and 28 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total dense inference MACs to approximately 247,482,900.
change: Remove only the central frame 12 from the qualified 22-step schedule, preserving both temporal boundaries of the dense early segment and all three proven late observations.
mechanism: Interior-frame temporal pruning
evidence_used: Removing frame 3 or frame 24 from the qualified schedule failed, indicating that the early boundary and late bridge are important; pruning one interior frame tests redundancy within the densely sampled 3–21 region while introducing only a single two-frame gap.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1290908465357080035, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 246456000, "recurrent_steps": 17115, "total_inference_macs": 247482900, "training_seconds": 37.79332912503742, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4972788804879218}



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
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 9)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
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

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with online temporal summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]


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

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
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

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]


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
