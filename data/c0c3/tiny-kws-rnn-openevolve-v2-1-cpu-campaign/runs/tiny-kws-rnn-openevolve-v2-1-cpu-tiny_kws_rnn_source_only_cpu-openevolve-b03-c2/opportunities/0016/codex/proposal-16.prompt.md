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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5381782338181254185, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 45225, "peak_hidden_elements": 114176, "recurrent_macs": 1031029605, "recurrent_steps": 23635, "total_inference_macs": 1031753325, "training_seconds": 112.0694795830641, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4746506111753499}
prior_hypothesis: A 111-unit GRU using 29 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the qualified 111-unit, 30-step design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5014660316337748473, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 959924115, "recurrent_steps": 22005, "total_inference_macs": 961371555, "training_seconds": 136.92549666692503, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.4030273999173217}
prior_hypothesis: Removing the penultimate input frame from the qualified dual-readout model will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.57%.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5567230865923986945, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 45225, "peak_hidden_elements": 114176, "recurrent_macs": 1066582350, "recurrent_steps": 24450, "total_inference_macs": 1067306070, "training_seconds": 120.5662536250893, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4578503193299463}
prior_hypothesis: A 111-unit GRU on the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.64% versus the 112-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 29 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.33% versus the qualified 30-step design.
change: Replace the full 32-frame schedule with 29 unique, approximately uniform indices spanning the first and final input frames.
mechanism: Uniform 29-frame causal subsampling
evidence_used: The 30-step design achieved 86.01% accuracy while the 28-step design achieved 83.93%; 29 steps directly tests the remaining pass/fail boundary and, if qualified, improves the dominant recurrent cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7010815633786821987, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1343224320, "recurrent_steps": 23635, "total_inference_macs": 1344058880, "training_seconds": 84.30788024980575, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4803711405561014}

RECENT RESULT
hypothesis: A 120-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 11.3% versus the current design and below the qualified 29-step, 128-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 120 while preserving the full causal frame schedule and training procedure.
mechanism: Moderate recurrent-width reduction
evidence_used: The current 128-unit, 32-step model achieves 87.24% accuracy with a 2.24-point margin, and recurrent computation accounts for virtually all MACs; a moderate width reduction targets that dominant cost while retaining all temporal evidence, unlike the failed 28-step schedule.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6860358437346396608, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1314432000, "recurrent_steps": 26080, "total_inference_macs": 1315214400, "training_seconds": 106.96294012502767, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4206474725454131}

RECENT RESULT
hypothesis: The 120-unit GRU using 30 uniformly distributed frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.3% versus the current 32-step model.
change: Change the frame schedule from all 32 frames to 30 approximately uniform causal frames spanning the complete recording.
mechanism: Joint moderate width reduction and uniform causal frame subsampling
evidence_used: The 120-unit model has 86.75% accuracy at 32 steps, while the 128-unit model retains 86.01% at 30 steps; these independent margins make their combination the most informative lower-cost candidate.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6431841104852922978, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1232280000, "recurrent_steps": 24450, "total_inference_macs": 1233062400, "training_seconds": 97.09306433307938, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4458835718821894}

RECENT RESULT
hypothesis: A 112-unit GRU processing 30 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 12% versus the qualified 120-unit, 30-step design.
change: Reduce the GRU, summary state, and classifier width from 128 to 112, and use the qualified 30-frame causal schedule.
mechanism: Joint recurrent-width reduction with qualified 30-frame subsampling
evidence_used: The 120-unit, 30-step model achieved 85.77% accuracy, while reducing width from 128 to 120 cost only 0.49 points at 32 steps; this supports testing the next 8-unit structural reduction while retaining the successful 30-frame schedule.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5660237832161349458, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1084406400, "recurrent_steps": 24450, "total_inference_macs": 1085136640, "training_seconds": 96.53904662490822, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4421053927368913}

RECENT RESULT
hypothesis: A 110-unit GRU processing the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the qualified 112-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 110 while preserving the 30-frame causal schedule and training procedure.
mechanism: Conservative recurrent-width boundary refinement
evidence_used: The 112-unit, 30-step model achieved 85.276% accuracy, and the reduction from 120 to 112 units cost only 0.491 percentage points; a smaller two-unit reduction directly probes the remaining width boundary with less risk than another eight-unit step.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5474989109208643138, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 44488, "peak_hidden_elements": 113152, "recurrent_macs": 1048905000, "recurrent_steps": 24450, "total_inference_macs": 1049622200, "training_seconds": 114.2407449579332, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.47666789797917464}

RECENT RESULT
hypothesis: A 111-unit GRU on the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.64% versus the 112-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 112 to 111 without changing the training procedure or frame schedule.
mechanism: Single-unit recurrent-width boundary refinement
evidence_used: The 112-unit model qualified at 85.276% accuracy, while 110 units achieved 84.540%; testing 111 units directly resolves the remaining width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5567230865923986945, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 45225, "peak_hidden_elements": 114176, "recurrent_macs": 1066582350, "recurrent_steps": 24450, "total_inference_macs": 1067306070, "training_seconds": 120.5662536250893, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4578503193299463}

RECENT RESULT
hypothesis: A 111-unit GRU using 29 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the qualified 111-unit, 30-step design.
change: Reduce the GRU, summary state, and classifier width from 120 to 111 and reduce the uniform causal schedule from 30 to 29 frames.
mechanism: Joint recurrent-width and causal-step boundary reduction
evidence_used: The 111-unit, 30-step design qualified at 85.40%, while the 128-unit, 29-step design qualified at 85.03%; combining these individually qualified reductions directly tests the next lower-cost boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5381782338181254185, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 45225, "peak_hidden_elements": 114176, "recurrent_macs": 1031029605, "recurrent_steps": 23635, "total_inference_macs": 1031753325, "training_seconds": 112.0694795830641, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4746506111753499}

RECENT RESULT
hypothesis: A 111-unit GRU using the qualified 29-frame schedule with only its second, edge-adjacent frame removed will retain at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.45%.
change: Reduce the GRU width to 111 and use a 28-step schedule nested within the qualified 29-step schedule by removing frame index 1.
mechanism: Nested edge-frame causal subsampling
evidence_used: The 111-unit, 29-step design qualified at 85.03%, whereas the failed uniform 28-step design changed four interior omissions at once; a nested schedule isolates one additional, likely redundant boundary-frame removal while preserving every selected interior frame from the qualified design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5196333810438521425, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 45225, "peak_hidden_elements": 114176, "recurrent_macs": 995476860, "recurrent_steps": 22820, "total_inference_macs": 996200580, "training_seconds": 137.46215675002895, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4754460621465203}

RECENT RESULT
hypothesis: Doubling the optimizer-update frequency with batch size 64 will recover the 0.215-point accuracy deficit of the nested 28-step design, reaching at least 85% while reducing recurrent MACs by approximately 3.45% versus the qualified 29-step model.
change: Use batch size 64 and remove the second selected frame from the qualified 29-frame schedule, yielding the previously tested nested 28-step inference path.
mechanism: Higher-update training for nested 28-step recurrence
evidence_used: The 111-unit nested 28-step model reached 84.785% versus the 85% requirement, with cross-entropy nearly identical to the qualified 29-step model; retaining the same 50,000-example exposure at half the batch size approximately doubles optimizer steps without adding inference MACs.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing frame 0 instead of frame 1 from the qualified 111-unit, 29-frame design will retain at least 85% validation accuracy while reducing recurrent MACs and steps by approximately 3.45%.
change: Use the qualified uniform 29-frame schedule as a base, then omit its first frame to execute 28 recurrent steps while preserving frame 1 and every qualified interior and final frame.
mechanism: Nested leading-edge frame omission
evidence_used: The qualified 29-step model achieved 85.03% accuracy; the nested 28-step variant that removed frame 1 narrowly missed at 84.785%, only two validation examples short, motivating an isolated test of the alternative leading-edge omission at identical inference cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A learned readout combining the final hidden state with the mean recurrent output will recover the nested 28-step model’s 0.215-point accuracy deficit and reach at least 85%, while remaining below 997 million total validation MACs.
change: Reduce the GRU to 111 units, use the qualified 29-frame schedule minus frame 1, and classify concatenated final-state and mean-output features.
mechanism: Dual-view temporal readout on a nested 28-step GRU
evidence_used: The 111-unit nested 28-step model reached 84.785% with cross-entropy nearly identical to the qualified 29-step model; adding the complementary final hidden state costs only 888 parameters and approximately 0.724 million validation MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5200108844080481233, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 995476860, "recurrent_steps": 22820, "total_inference_macs": 996924300, "training_seconds": 157.15188074996695, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.39772155387269936}

RECENT RESULT
hypothesis: Removing the penultimate input frame from the qualified dual-readout model will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.57%.
change: Reduce the causal schedule from 28 to 27 steps by omitting frame 30 in addition to frame 1, while preserving both endpoints, every interior selected frame, and the 111-unit dual-view readout.
mechanism: Nested trailing-edge frame omission
evidence_used: The current 28-step dual-readout design achieved 87.48% accuracy, a 2.48-point margin; moreover, the earlier single-readout transition from 29 steps to the nested 28-step schedule lost only 0.245 points, supporting an isolated additional edge-adjacent omission.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5014660316337748473, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 959924115, "recurrent_steps": 22005, "total_inference_macs": 961371555, "training_seconds": 136.92549666692503, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.4030273999173217}



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
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
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
        steps = min(29, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
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
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 4:
            return [
                frame
                for frame in schedule
                if frame not in (1, available_frames - 2)
            ]
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
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
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
        steps = min(30, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
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
