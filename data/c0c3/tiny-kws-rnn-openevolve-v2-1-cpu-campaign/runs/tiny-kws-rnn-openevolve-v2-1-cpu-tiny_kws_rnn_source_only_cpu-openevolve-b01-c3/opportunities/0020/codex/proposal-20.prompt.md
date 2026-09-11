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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2588686809637269777, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 495611280, "recurrent_steps": 13040, "total_inference_macs": 496282840, "training_seconds": 75.18150233407505, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.46482491522478914}
prior_hypothesis: A 103-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 505,169,600 to approximately 496,282,840.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2497201760115921605, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 38125, "peak_hidden_elements": 103936, "recurrent_macs": 478085520, "recurrent_steps": 13040, "total_inference_macs": 478744040, "training_seconds": 85.76904862490483, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.45338834984902227}
prior_hypothesis: A 101-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 487,474,320 to approximately 478,744,040.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2635041502015558832, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 504491520, "recurrent_steps": 13040, "total_inference_macs": 505169600, "training_seconds": 115.78899174998514, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.47056287519770895}
prior_hypothesis: A 104-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 514,134,600 to approximately 505,169,600.

## Recent verification evidence

RECENT RESULT
hypothesis: A 112-unit GRU trained with batch size 64 on the proven even-index schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.3% versus the qualified 116-unit design.
change: Reduce recurrent, summary-state, and classifier width from the current 124 units to 112 while preserving the successful training regime and 16-frame schedule.
mechanism: Further width trimming of the qualified 16-step causal GRU
evidence_used: The 116-unit design achieved 86.50% accuracy, 1.50 points above the requirement, after successive four-unit reductions from 124 to 120 to 116 all qualified with batch size 64.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3020571063864630528, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 578350080, "recurrent_steps": 13040, "total_inference_macs": 579080320, "training_seconds": 88.51149762491696, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4576998564363257}

RECENT RESULT
hypothesis: A 110-unit GRU trained with batch size 64 on the proven even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.3% versus the qualified 112-unit design.
change: Reproduce the successful batch-64 training regime and reduce recurrent, summary-state, and classifier width from 128 to 110 units while retaining all 16 scheduled frames.
mechanism: Boundary-seeking width trim of the qualified 16-step causal GRU
evidence_used: The 112-unit design achieved 85.644% accuracy at 579,080,320 total MACs; a two-unit reduction is a conservative boundary test expected to cost about 560,133,200 MACs while using its remaining 0.644-point accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2921740002931902728, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 44488, "peak_hidden_elements": 113152, "recurrent_macs": 559416000, "recurrent_steps": 13040, "total_inference_macs": 560133200, "training_seconds": 81.36387941683643, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4566807939962375}

RECENT RESULT
hypothesis: A 64-unit fast GRU encoding all 16 scheduled frames into four local chunks, followed by a 112-unit slow GRU operating only at chunk boundaries, will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 28% versus the qualified 110-unit monolithic GRU.
change: Replace the assumption that every frame needs the same full-width recurrent update with a two-timescale hierarchy: a resettable fast GRU processes each four-frame chunk, a slow GRU models the resulting four-chunk command sequence, and prediction combines the slow state with the mean fast representation.
mechanism: Four-frame hierarchical dual-rate GRU
evidence_used: Reducing temporal coverage from 16 to 15 frames collapsed accuracy to 80.25%, while the current 16-frame model reaches 86.01%; this suggests preserving all observations but challenges the unsupported assumption that all 16 require an expensive 110-unit transition. The proposed hierarchy keeps complete coverage and is expected to reduce recurrent MACs from 559,416,000 to about 403,092,480.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 108-unit GRU with the proven batch-64 training regime and 16-frame even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.3% versus the qualified 110-unit design.
change: Reduce recurrent, summary-state, and classifier width from 112 to 108 units while preserving the validated schedule and training procedure.
mechanism: Boundary-seeking recurrent width reduction
evidence_used: The 110-unit design achieved 86.01% accuracy at 560,133,200 total MACs, improving accuracy over the wider 112-unit design; its 1.01-point margin supports testing another conservative two-unit reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2824541388979481512, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 540794880, "recurrent_steps": 13040, "total_inference_macs": 541499040, "training_seconds": 177.5632731250953, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4438787542237826}

RECENT RESULT
hypothesis: A 106-unit GRU with the proven batch-64 training regime and 16-frame even-index schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 3.4% versus the qualified 108-unit design.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 116 to 106 units while preserving the validated training procedure and frame schedule.
mechanism: Boundary-seeking recurrent width reduction
evidence_used: The 108-unit design qualified at 85.399% accuracy and 541,499,040 total MACs after the 110-unit design qualified at 86.012%; this supports another two-unit boundary test, projected near 523,177,840 total MACs.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 107-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 541,499,040 to approximately 532,299,320.
change: Reduce the recurrent state, temporal summary, and classifier width from 108 to 107 units without changing training or temporal coverage.
mechanism: One-unit boundary width trim
evidence_used: The 108-unit design qualified at 85.399% accuracy, and the 110-unit design qualified at 86.012%; a one-unit reduction is the smallest informative capacity test below the current boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2776554249620885873, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 42313, "peak_hidden_elements": 110080, "recurrent_macs": 531601680, "recurrent_steps": 13040, "total_inference_macs": 532299320, "training_seconds": 108.52942700008862, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.45476566618936926}

RECENT RESULT
hypothesis: A 106-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 532,299,320 to approximately 523,177,840.
change: Reduce the recurrent state, temporal summary, and classifier width from 110 to 106 units while preserving the validated optimizer, augmentation, and even-index frame schedule.
mechanism: One-unit boundary width trim
evidence_used: The adjacent 107-unit design achieved 86.135% accuracy at 532,299,320 MACs; the prior 106-unit attempt timed out without producing contrary accuracy evidence, so retrying this smallest structural reduction directly tests the next unresolved cost boundary.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: A 105-unit GRU using the qualified batch-64, 16-frame regime will retain at least 85% validation accuracy while reducing total inference MACs from 532,299,320 to approximately 514,134,600.
change: Reduce recurrent-state, temporal-summary, and classifier width from the current 116 units to 105 while preserving training and temporal coverage.
mechanism: Nonduplicate boundary-width trim
evidence_used: The 107-unit design achieved 86.135% accuracy with 532,299,320 MACs; because the subsequent 106-unit submission reproduced an already verified implementation, 105 units is the next untested lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2681804306138924533, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 40893, "peak_hidden_elements": 108032, "recurrent_macs": 513450000, "recurrent_steps": 13040, "total_inference_macs": 514134600, "training_seconds": 113.32499116589315, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.47585878898761025}

RECENT RESULT
hypothesis: A 104-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 514,134,600 to approximately 505,169,600.
change: Reduce the recurrent state, temporal summary, and classifier width from 105 to 104 units while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 105-unit design achieved 85.644% accuracy at 514,134,600 MACs, leaving a 0.644-point margin and making a one-unit reduction the smallest informative test below the current cost boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2635041502015558832, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 504491520, "recurrent_steps": 13040, "total_inference_macs": 505169600, "training_seconds": 115.78899174998514, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.47056287519770895}

RECENT RESULT
hypothesis: A 103-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 505,169,600 to approximately 496,282,840.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 107 to 103 units while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 104-unit design achieved 86.626% accuracy at 505,169,600 MACs, leaving a 1.626-point margin and motivating the smallest untested structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2588686809637269777, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 495611280, "recurrent_steps": 13040, "total_inference_macs": 496282840, "training_seconds": 75.18150233407505, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.46482491522478914}

RECENT RESULT
hypothesis: A 102-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 496,282,840 to approximately 487,474,320.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from the current 110 units to 102 while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 103-unit design qualified at 85.644% accuracy and 496,282,840 MACs, leaving a 0.644-point margin; 102 units is the next untested lower-cost boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2542740229004057368, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 38808, "peak_hidden_elements": 104960, "recurrent_macs": 486809280, "recurrent_steps": 13040, "total_inference_macs": 487474320, "training_seconds": 79.22702487488277, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46595370988904333}

RECENT RESULT
hypothesis: A 101-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 487,474,320 to approximately 478,744,040.
change: Reduce the GRU, recurrent state, temporal summary, and classifier width from 102 to 101 units while preserving the qualified batch-64 training regime and 16-frame schedule.
mechanism: One-unit recurrent width boundary test
evidence_used: The adjacent 102-unit design achieved 85.276% accuracy at 487,474,320 MACs; 101 units is the next untested structural reduction, while the failed 15-frame result indicates temporal coverage should remain unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2497201760115921605, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 38125, "peak_hidden_elements": 103936, "recurrent_macs": 478085520, "recurrent_steps": 13040, "total_inference_macs": 478744040, "training_seconds": 85.76904862490483, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.45338834984902227}

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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
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
        return list(range(0, available_frames, 2))


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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 101, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(101, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
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
        return list(range(0, available_frames, 2))


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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
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
        return list(range(0, available_frames, 2))


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
