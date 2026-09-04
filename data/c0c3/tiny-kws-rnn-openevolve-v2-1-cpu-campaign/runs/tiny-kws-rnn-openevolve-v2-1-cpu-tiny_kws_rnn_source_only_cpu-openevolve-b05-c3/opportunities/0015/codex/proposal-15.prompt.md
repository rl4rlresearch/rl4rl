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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 103.09386774990708, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.47749484945659987}
prior_hypothesis: A 24-frame uniform schedule will recover at least 85% accuracy while reducing total inference MACs to approximately 1.112 billion, below the qualified 112-unit model.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4700835137295352156, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 58016, "peak_hidden_elements": 129536, "recurrent_macs": 899564400, "recurrent_steps": 16300, "total_inference_macs": 901207440, "training_seconds": 132.47330612502992, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4095238434025115}
prior_hypothesis: A 126-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 914,570,180 to approximately 901,207,440.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4631643191848815981, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 57181, "peak_hidden_elements": 128512, "recurrent_macs": 886312500, "recurrent_steps": 16300, "total_inference_macs": 887942500, "training_seconds": 61.309398707933724, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.42285024256794}
prior_hypothesis: A 125-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 126-unit design’s 901,207,440 MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: A 112-unit GRU retaining all 32 frames will preserve at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit baseline.
change: Reduce the recurrent hidden width and classifier input width from 128 to 112 without temporal subsampling.
mechanism: Full-resolution narrow-state GRU
evidence_used: Both 16-step and 20-step schedules achieved only 84.66%, suggesting temporal evidence should be preserved; the full 32-step 128-unit model reached 87.24%, leaving a 2.24-point margin for a modest width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 135.83657795889303, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4385198019764906}

RECENT RESULT
hypothesis: A 24-frame uniform schedule will recover at least 85% accuracy while reducing total inference MACs to approximately 1.112 billion, below the qualified 112-unit model.
change: Retain the 128-unit GRU but process 24 uniformly distributed frames, including both recording endpoints.
mechanism: Three-of-four-frame causal recurrence
evidence_used: The 20-step model missed the target by only three validation examples, while the 32-step 128-unit model reached 87.24%; 24 steps add temporal resolution while costing less than the qualified full-resolution 112-unit model.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 103.09386774990708, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.47749484945659987}

RECENT RESULT
hypothesis: Processing 22 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 8.3% versus the qualified 24-frame design.
change: Replace full 32-frame recurrence with 22 unique increasing indices spanning the recording endpoints.
mechanism: Midpoint uniform temporal subsampling
evidence_used: The 20-frame design reached 84.66% while the 24-frame design reached 85.77%; testing their midpoint is the most informative next search step and should cost approximately 1.020 billion total MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5319600561618720122, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1018997760, "recurrent_steps": 17930, "total_inference_macs": 1019832320, "training_seconds": 92.00011508283205, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.482319500694977}

RECENT RESULT
hypothesis: Processing 23 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by about 4.2% versus the qualified 24-frame design.
change: Replace full 32-frame recurrence with 23 unique increasing indices spanning both recording endpoints.
mechanism: Boundary-search temporal subsampling
evidence_used: The 24-frame design achieved 85.77% accuracy, while 22 frames achieved 84.42%; testing 23 frames directly probes the qualification boundary with an estimated total cost near 1.066 billion MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5561202714785591817, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1065315840, "recurrent_steps": 18745, "total_inference_macs": 1066150400, "training_seconds": 85.33179225004278, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4889097248849693}

RECENT RESULT
hypothesis: A 124-unit GRU processing the qualified 24-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 1.112 billion to approximately 1.049 billion.
change: Apply the proven 24-frame uniform schedule and reduce the GRU hidden and classifier widths from 128 to 124.
mechanism: Width-reduced three-of-four-frame causal GRU
evidence_used: The 128-unit 24-frame design achieved 85.77% accuracy, while reducing width all the way to 112 at full resolution still achieved 85.15%; a modest four-unit reduction should fit within the qualified 24-frame design’s 0.77-point margin.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5469649646721563072, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 1047790080, "recurrent_steps": 19560, "total_inference_macs": 1048598560, "training_seconds": 95.9977690000087, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.49073465733440375}

RECENT RESULT
hypothesis: A 126-unit GRU processing the qualified 24-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs to approximately 1.080 billion.
change: Use 24 uniformly distributed causal frames and reduce the GRU hidden and classifier widths from 128 to 126.
mechanism: Boundary-search width reduction on qualified temporal schedule
evidence_used: The 128-unit 24-frame design achieved 85.77%, while the 124-unit variant achieved 84.29%; testing the midpoint directly probes the narrowest qualified width boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5635002922101783360, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 57000, "peak_hidden_elements": 129536, "recurrent_macs": 1079477280, "recurrent_steps": 19560, "total_inference_macs": 1080298800, "training_seconds": 100.35242737480439, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.47351122897095477}

RECENT RESULT
hypothesis: A 127-unit GRU using the qualified 24-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs below the 1,112,468,480-MAC 128-unit design.
change: Process 24 uniformly distributed causal frames and reduce the GRU hidden state, temporal summary, and classifier input from 128 to 127 units.
mechanism: One-unit boundary width reduction on the qualified 24-frame GRU
evidence_used: The 128-unit 24-frame model achieved 85.77%, whereas the 126-unit variant achieved 84.66%; testing the intervening 127-unit width directly resolves the remaining qualification boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5718597811218315953, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1095496920, "recurrent_steps": 19560, "total_inference_macs": 1096324960, "training_seconds": 153.24448733287863, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4708847233122843}

RECENT RESULT
hypothesis: A 128-unit GRU processing 20 uniform frames with a learned terminal-state residual head will achieve at least 85% accuracy while keeping total inference MACs below 0.93 billion.
change: Replace the assumption that uniformly averaging recurrent outputs is sufficient with a dual-timescale prediction: retain the proven mean summary and add a zero-initialized classifier head over the terminal hidden state, which preserves ordered whole-utterance context without adding recurrent MACs.
mechanism: Residual endpoint-aware temporal readout
evidence_used: The 20-step mean-only model reached 84.66%, just three validation examples below qualification. Unlike repeating temporal or width boundary searches, the endpoint head directly addresses the shared equal-averaging bottleneck while preserving the original predictor at initialization and adding only 834,560 validation MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4840749447232461924, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 59704, "peak_hidden_elements": 131584, "recurrent_macs": 926361600, "recurrent_steps": 16300, "total_inference_macs": 928030720, "training_seconds": 171.0220195830334, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4255875101850077}

RECENT RESULT
hypothesis: Reducing the qualified dual-readout design from 20 to 19 uniformly distributed frames will retain at least 85% validation accuracy while lowering total inference cost to approximately 882 million MACs and 19 recurrent steps per example.
change: Add the proven zero-initialized terminal-state residual classifier and use 19 uniformly distributed causal frames.
mechanism: Nineteen-step endpoint-aware temporal readout
evidence_used: The 20-step dual-readout model achieved 85.89% accuracy at 928,030,720 MACs, outperforming the 20-step mean-only model by 1.23 percentage points; its 0.89-point qualification margin motivates testing one fewer recurrent step.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 127-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 928,030,720 to approximately 914,570,180.
change: Replace the current full-resolution 112-unit mean-only model with a 127-unit GRU processing 20 uniformly distributed frames and combining mean-state and terminal-state classifier heads.
mechanism: One-unit width reduction of endpoint-aware temporal readout
evidence_used: The 128-unit 20-frame dual-readout design achieved 85.89% accuracy; the 127-unit 24-frame mean-only design also qualified at 85.03%, making a one-unit reduction the most supported next cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4770537222423234137, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 58857, "peak_hidden_elements": 130560, "recurrent_macs": 912914100, "recurrent_steps": 16300, "total_inference_macs": 914570180, "training_seconds": 98.75726579199545, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4156104620248994}

RECENT RESULT
hypothesis: A 126-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 914,570,180 to approximately 901,207,440.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 127 to 126 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: One-unit width reduction of the qualified dual-readout GRU
evidence_used: The current 127-unit design achieved 86.26% accuracy, providing a 1.26-point qualification margin; the dual-readout mechanism also qualified at 128 units, so testing the adjacent 126-unit width is the most direct next cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4700835137295352156, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 58016, "peak_hidden_elements": 129536, "recurrent_macs": 899564400, "recurrent_steps": 16300, "total_inference_macs": 901207440, "training_seconds": 132.47330612502992, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4095238434025115}

RECENT RESULT
hypothesis: A 125-unit GRU with the proven 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 126-unit design’s 901,207,440 MACs.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 128 to 125 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: One-unit boundary width reduction of the qualified dual-readout GRU
evidence_used: The 126-unit 20-frame dual-readout model achieved 86.87% accuracy, a 1.87-point margin, while successively tested 128- and 127-unit variants also qualified; the adjacent 125-unit width is the most direct next cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4631643191848815981, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 57181, "peak_hidden_elements": 128512, "recurrent_macs": 886312500, "recurrent_steps": 16300, "total_inference_macs": 887942500, "training_seconds": 61.309398707933724, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.42285024256794}



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
        steps = min(24, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
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
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 126, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(126, 8)
        self.endpoint_classifier = nn.Linear(126, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 126, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 126, device=device, dtype=dtype)
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
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
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
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(125, 8)
        self.endpoint_classifier = nn.Linear(125, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
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
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
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
