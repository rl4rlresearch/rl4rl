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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3278633724897919381, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 627406560, "recurrent_steps": 22005, "total_inference_macs": 628554080, "training_seconds": 64.4013512081001, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45655335970451494}
prior_hypothesis: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy on the most recent 27 frames while reducing total inference MACs below 651,791,360 and recurrent steps from 28 to 27 per example.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3157424536528656486, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 604169280, "recurrent_steps": 21190, "total_inference_macs": 605316800, "training_seconds": 70.86741162510589, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44337790085494155}
prior_hypothesis: An 88-unit dual-readout GRU using 26 evenly thinned frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 628,554,080 to approximately 605,316,800.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3521052101636445171, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 673881120, "recurrent_steps": 23635, "total_inference_macs": 675028640, "training_seconds": 74.68331041699275, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.42446389227557035}
prior_hypothesis: An 88-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 30-frame model’s 698,265,920.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3642261290005708066, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 697118400, "recurrent_steps": 24450, "total_inference_macs": 698265920, "training_seconds": 84.52542316704057, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4217787853779237}
prior_hypothesis: An 88-unit dual-readout GRU over the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 712,728,910 to 698,265,920.

## Recent verification evidence

RECENT RESULT
hypothesis: A 92-unit dual-readout GRU processing the most recent 30 frames will achieve at least 85% validation accuracy while reducing total inference MACs below the qualified 94-unit model’s 787,244,360.
change: Reduce the GRU hidden width from 98 to 92, resize the recurrent state, temporal summary, and mean-plus-final classifier accordingly, and use the qualified 30-frame schedule.
mechanism: Dual-timescale recurrent width refinement
evidence_used: The 94-unit, 30-frame dual-readout model achieved 86.99% accuracy, retaining 1.99 percentage points of margin after the preceding reduction from 96 units; another two-unit reduction directly tests the next structural cost boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3948617173309908314, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 755798400, "recurrent_steps": 24450, "total_inference_macs": 756998080, "training_seconds": 106.37251258315518, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.3886520011293376}

RECENT RESULT
hypothesis: A 90-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below 756,998,080.
change: Reduce the GRU hidden width from 92 to 90 and resize the recurrent state, temporal summary, and mean-plus-final classifier accordingly.
mechanism: Dual-timescale recurrent width refinement
evidence_used: The 92-unit, 30-frame model achieved 87.61% accuracy, improving on the qualified 94-unit model and leaving 2.61 percentage points of margin for the next two-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3793908812613770778, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 31728, "peak_hidden_elements": 92672, "recurrent_macs": 726165000, "recurrent_steps": 24450, "total_inference_macs": 727338600, "training_seconds": 101.429184040986, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4073098305544239}

RECENT RESULT
hypothesis: Two independent 52-unit GRU branches over the qualified 30-frame schedule will achieve at least 85% validation accuracy while reducing predicted total inference MACs from 727,338,600 to approximately 550,600,960.
change: Replace the single densely connected 94-unit GRU with two parallel 52-unit GRUs, concatenate their temporal outputs, and retain the mean-plus-final readout.
mechanism: Parallel grouped-GRU temporal subspaces
evidence_used: Dense 90-, 92-, 94-, and 96-unit GRUs all qualified, suggesting total recurrent width is useful; the failed low-rank experiment challenged dense recurrent mixing but timed out. Standard parallel GRUs cleanly test whether full cross-channel recurrent connectivity is unnecessary while preserving gated dynamics and efficient sequence execution.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2872018389797723810, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 24800, "peak_hidden_elements": 107008, "recurrent_macs": 549244800, "recurrent_steps": 24450, "total_inference_macs": 550600960, "training_seconds": 76.44758554222062, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4542129984662577}

RECENT RESULT
hypothesis: An 89-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 90-unit model’s 727,338,600.
change: Reduce the GRU hidden width from 96 to 89 and resize its classifier, hidden state, and temporal summary while preserving the qualified 30-frame schedule and training procedure.
mechanism: Single-unit recurrent width refinement
evidence_used: The 90-unit, 30-frame dual-readout GRU achieved 86.01% accuracy; its 1.01-point margin motivates a conservative one-unit reduction after the successful sequence of width refinements.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3717702446548730069, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 31109, "peak_hidden_elements": 91648, "recurrent_macs": 711568350, "recurrent_steps": 24450, "total_inference_macs": 712728910, "training_seconds": 95.21129420911893, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4104718729030867}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU over the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 712,728,910 to 698,265,920.
change: Reduce the GRU hidden width from 89 to 88 and resize its state, temporal summary, and mean-plus-final classifier.
mechanism: Single-unit recurrent width refinement
evidence_used: The 89-unit model achieved 86.26% accuracy, while every tested dense width from 90 through 96 also qualified; this 1.26-point margin supports the next conservative one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3642261290005708066, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 697118400, "recurrent_steps": 24450, "total_inference_macs": 698265920, "training_seconds": 84.52542316704057, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4217787853779237}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 30-frame model’s 698,265,920.
change: Reduce hidden width from 90 to 88, resize the classifier and recurrent state, and omit the first three input frames.
mechanism: Incremental temporal trimming of a qualified narrow GRU
evidence_used: The 88-unit, 30-frame model achieved 86.38% accuracy, while trimming the 98-unit model from 30 to 29 frames reduced accuracy by only 0.74 points and still qualified; applying the same single-frame trim to the narrower qualified design tests a larger structural cost reduction with 1.38 points of available margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3521052101636445171, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 673881120, "recurrent_steps": 23635, "total_inference_macs": 675028640, "training_seconds": 74.68331041699275, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.42446389227557035}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU processing the most recent 28 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 29-frame model’s 675,028,640.
change: Reduce the GRU width from 94 to 88, resize its state and classifier, and omit the first four input frames.
mechanism: Incremental temporal trimming of the qualified narrow GRU
evidence_used: The 88-unit model qualified at 86.38% over 30 frames and 86.50% over 29 frames; the latter leaves 1.50 percentage points of margin and motivates testing the next single-frame structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3399842913267182276, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 650643840, "recurrent_steps": 22820, "total_inference_macs": 651791360, "training_seconds": 66.04627275001258, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.43144288794394653}

RECENT RESULT
hypothesis: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy on the most recent 27 frames while reducing total inference MACs below 651,791,360 and recurrent steps from 28 to 27 per example.
change: Omit one additional leading input frame, changing the causal schedule from the most recent 28 frames to the most recent 27 while preserving the model and training procedure.
mechanism: Incremental trailing-window temporal trimming
evidence_used: The same 88-unit model qualified at 86.50% with 29 frames and 85.89% with 28 frames; the current 0.89-point margin supports testing the next single-frame structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3278633724897919381, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 627406560, "recurrent_steps": 22005, "total_inference_macs": 628554080, "training_seconds": 64.4013512081001, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45655335970451494}

RECENT RESULT
hypothesis: An 86-unit dual-readout GRU over the most recent 28 frames will achieve at least 85% validation accuracy while reducing predicted total inference MACs below the qualified 88-unit, 27-frame model’s 628,554,080 to approximately 625,202,800.
change: Reduce GRU width from 88 to 86, resize its state and classifier, and expand the trailing window from 27 qualified frames to 28 frames.
mechanism: Width–time exchange with dual-timescale readout
evidence_used: The 88-unit model achieved 85.89% over 28 frames versus 85.03% over 27; additionally, exchanging two units for one frame previously improved accuracy when moving from the 98-unit, 29-frame model to the 96-unit, 30-frame model.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3261152938565302908, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 624081360, "recurrent_steps": 22820, "total_inference_macs": 625202800, "training_seconds": 59.73556770803407, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.45403945080341734}

RECENT RESULT
hypothesis: An 86-unit GRU over 28 frames with mean–final multiplicative features will achieve at least 85% validation accuracy while keeping total inference MACs below the qualified 88-unit, 27-frame model.
change: Use the near-qualified 86-unit, 28-frame architecture and augment its linear classifier with the elementwise product of mean and final recurrent outputs.
mechanism: Parameter-free cross-timescale interaction readout
evidence_used: The 86-unit, 28-frame model missed qualification by one validation example while achieving lower cross-entropy than the qualified 88-unit, 27-frame model; adding an inexpensive interaction feature targets this narrow accuracy gap with a predicted 625,763,520 total MACs, still below 628,554,080.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3264077739405019516, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29976, "peak_hidden_elements": 88576, "recurrent_macs": 624081360, "recurrent_steps": 22820, "total_inference_macs": 625763520, "training_seconds": 52.25696949986741, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.46472425402307804}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU using 26 evenly thinned frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 628,554,080 to approximately 605,316,800.
change: Replace the 128-unit baseline with the qualified 88-unit mean-plus-final GRU and process 26 samples spanning the latest 28 frames instead of truncating the window to 26 contiguous frames.
mechanism: Span-preserving temporal thinning
evidence_used: The 88-unit model achieved 85.89% on 28 frames and 85.03% on 27, whereas reducing width to 86 missed qualification; retaining width and removing two distributed, temporally redundant frames tests a lower-cost alternative while preserving the stronger 28-frame temporal span.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3157424536528656486, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 604169280, "recurrent_steps": 21190, "total_inference_macs": 605316800, "training_seconds": 70.86741162510589, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44337790085494155}

RECENT RESULT
hypothesis: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy using 25 evenly distributed frames across the latest 28-frame window, reducing predicted total inference MACs from 605,316,800 to approximately 582,079,520 and recurrent steps from 26 to 25 per example.
change: Reduce the frame-schedule target from 26 to 25 while preserving the qualified GRU, training procedure, and 28-frame temporal span.
mechanism: Incremental span-preserving temporal thinning
evidence_used: The 26-frame span-preserving schedule qualified at 85.03% and achieved lower cross-entropy than the qualified 27-frame contiguous schedule; this indicates that distributed sampling preserves useful temporal coverage and motivates testing the next single-step cost boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3036215348159393591, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 580932000, "recurrent_steps": 20375, "total_inference_macs": 582079520, "training_seconds": 70.22041741595604, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4608848805807851}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the recurrent model represents time, updates state, controls computation, or forms command predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
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
        start = max(available_frames - 27, 0)
        return list(range(start, available_frames))


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
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
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
        target_steps = min(26, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
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
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
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
        start = max(available_frames - 29, 0)
        return list(range(start, available_frames))


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
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
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
        start = max(available_frames - 30, 0)
        return list(range(start, available_frames))


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
