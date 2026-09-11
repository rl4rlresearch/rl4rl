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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4076900298593832233, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 780991680, "recurrent_steps": 25265, "total_inference_macs": 781591520, "training_seconds": 195.1400665419642, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.46826342249209163}
prior_hypothesis: Omitting only the earliest frame will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125% versus the verified 92-unit, 32-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit baseline.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 128 to 112 while preserving the full causal frame schedule.
mechanism: Full-resolution recurrent width compression
evidence_used: Temporal subsampling reduced accuracy from 87.24% to 83.19% even at 24 steps, indicating that all 32 frames are valuable; compressing hidden width instead tests an independent structural cost reduction without discarding temporal evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 107.5079114169348, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.44323348530962425}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 12.8% versus the verified 112-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 112 to 104 while preserving the full causal frame schedule and training procedure.
mechanism: Full-resolution GRU width compression
evidence_used: The 112-unit full-resolution GRU achieved 85.89% accuracy, while temporal subsampling failed at both 16 and 24 steps; this motivates preserving all 32 frames and testing the next structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 113.093671125127, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.45115726213513707}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13.7% versus the verified 104-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 104 to 96 while preserving the full causal frame schedule and training procedure.
mechanism: Full-resolution GRU width compression
evidence_used: The 104-unit full-resolution GRU achieved 86.01% accuracy, slightly improving on the 112-unit model’s 85.89%, while both temporal-subsampling experiments failed; this supports another width reduction without discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 100.82147183292545, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4543663258932851}

RECENT RESULT
hypothesis: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the verified 96-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 96 to 92 while preserving the full causal frame schedule and training procedure.
mechanism: Boundary-seeking full-resolution GRU width compression
evidence_used: The 96-unit full-resolution GRU achieved 85.15% accuracy after the 104-unit model achieved 86.01%; testing the midpoint width of 92 is the most direct probe of the feasible compression boundary without discarding temporally valuable frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4208312280590011128, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 806784800, "training_seconds": 118.667672208976, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.46424440723255367}

RECENT RESULT
hypothesis: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.8% versus the verified 92-unit model.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 92 to 88 while preserving the full causal frame schedule and training procedure.
mechanism: Boundary-seeking full-resolution GRU width compression
evidence_used: The 92-unit full-resolution GRU achieved 85.89% accuracy with 806,184,960 recurrent MACs, leaving enough observed accuracy margin to test the next four-unit structural reduction without discarding valuable frames.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3881686847280337792, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29792, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744166720, "training_seconds": 103.23932075011544, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.47434615620806175}

RECENT RESULT
hypothesis: A 90-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.9% versus the verified 92-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 92 to 90 while retaining all 32 causal frames and the existing training procedure.
mechanism: Boundary-bisection GRU width compression
evidence_used: The 92-unit model achieved 85.89% accuracy, while the 88-unit model achieved 84.66%; width 90 directly probes the midpoint of the observed feasibility boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4043367116954867888, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31008, "peak_hidden_elements": 92672, "recurrent_macs": 774576000, "recurrent_steps": 26080, "total_inference_macs": 775162800, "training_seconds": 94.96799183310941, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46152874531190086}

RECENT RESULT
hypothesis: A 91-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs versus the verified 92-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 92 to 91 while preserving all 32 causal frames and the existing training procedure.
mechanism: Boundary-bisection GRU width compression
evidence_used: The 92-unit model achieved 85.89% accuracy, while widths 90 and 88 missed the threshold; width 91 is the only untested integer at the observed feasibility boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4125431587027362865, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31625, "peak_hidden_elements": 93696, "recurrent_macs": 790302240, "recurrent_steps": 26080, "total_inference_macs": 790895560, "training_seconds": 106.7818828751333, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.4779986048037289}

RECENT RESULT
hypothesis: Selective exits at steps 24 and 28 will preserve at least 85% accuracy while reducing mean recurrent steps and total MACs, because only high-confidence examples stop early while ambiguous examples retain all 32 frames.
change: Keep the verified 92-unit GRU, train its shared classifier on 24-, 28-, and 32-step summaries, cache logits only at those checkpoints, and exit at conservative confidence thresholds.
mechanism: Auxiliary-trained confidence-gated recurrent checkpoints
evidence_used: Uniform 16- and 24-frame execution failed at 83.19%, while the full 32-step 92-unit GRU reached 85.89%; this challenges the load-bearing assumption that every example needs the same temporal budget without again discarding later evidence for every recording.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Omitting only the earliest frame will preserve at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125% versus the verified 92-unit, 32-step model.
change: Retain the verified 92-unit GRU and training procedure, but process frames 1–31 instead of all 32 frames.
mechanism: Minimal leading-frame temporal pruning
evidence_used: The 92-unit model reached 85.89% accuracy at 32 steps, whereas uniform reduction to 24 steps was too aggressive at 83.19%; removing a single likely low-information boundary frame is the smallest direct probe of temporal compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4076900298593832233, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 780991680, "recurrent_steps": 25265, "total_inference_macs": 781591520, "training_seconds": 195.1400665419642, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.46826342249209163}

RECENT RESULT
hypothesis: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.
change: Omit one additional earliest frame, reducing the causal schedule from 31 to 30 recurrent steps while leaving model capacity and training unchanged.
mechanism: Incremental leading-boundary frame pruning
evidence_used: Omitting frame 0 preserved 85.40% accuracy and reduced inference cost versus the 32-step model; pruning one more likely low-information boundary frame is the smallest informative test of the temporal-compression limit.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Processing frames 2–31 will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the verified 31-step design.
change: Omit the second-earliest frame, reducing the causal schedule from 31 to 30 steps without changing capacity or training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: The 31-step 92-unit GRU achieved 85.40% accuracy; the prior 30-step attempt timed out and therefore provided no accuracy evidence, making this smallest additional temporal reduction the unresolved boundary test.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.
change: Omit one additional earliest frame, reducing the causal schedule from 31 to 30 recurrent steps while preserving model capacity and training.
mechanism: Incremental leading-boundary frame pruning
evidence_used: The 31-step model achieved 85.40% accuracy; the intended 30-step test timed out and the subsequent edit reproduced the 31-step implementation, so this compression boundary remains unresolved.
result: the edit reproduced a previously verified implementation



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
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))


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


