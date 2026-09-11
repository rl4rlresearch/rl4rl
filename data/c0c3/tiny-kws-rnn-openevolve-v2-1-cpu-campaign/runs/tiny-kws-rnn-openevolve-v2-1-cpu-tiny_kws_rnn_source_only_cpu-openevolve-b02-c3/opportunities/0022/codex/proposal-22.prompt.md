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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3717702446548730069, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 31109, "peak_hidden_elements": 91648, "recurrent_macs": 711568350, "recurrent_steps": 24450, "total_inference_macs": 712728910, "training_seconds": 95.21129420911893, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4104718729030867}
prior_hypothesis: An 89-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 90-unit model’s 727,338,600.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3793908812613770778, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 31728, "peak_hidden_elements": 92672, "recurrent_macs": 726165000, "recurrent_steps": 24450, "total_inference_macs": 727338600, "training_seconds": 101.429184040986, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4073098305544239}
prior_hypothesis: A 90-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below 756,998,080.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4106386372094120674, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 34264, "peak_hidden_elements": 96768, "recurrent_macs": 786018600, "recurrent_steps": 24450, "total_inference_macs": 787244360, "training_seconds": 124.8966925831046, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4043488016889139}
prior_hypothesis: A 94-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 96-unit model’s 818,077,440.

## Recent verification evidence

RECENT RESULT
hypothesis: A 128-dimensional GRU-like state with rank-32 learned recurrent mixing and dual-timescale readout will retain at least 85% accuracy while reducing total inference MACs below the current 906,045,280.
change: Replace the dense 98-unit standard GRU with a custom GRU update whose recurrent gates share a 32-dimensional bottleneck, while retaining all 32 frames and concatenating mean and final states for classification.
mechanism: Shared-bottleneck low-rank GRU recurrence
evidence_used: Dropping frames failed even at 24 steps, showing full temporal coverage is load-bearing, while the 98-unit mean-plus-final readout reached 87.12%. This challenges the remaining assumption that every gate needs full-rank hidden-to-hidden mixing: a wider state preserves representation capacity while low-rank recurrent matrices structurally reduce MACs.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 90-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 906,045,280 to approximately 775,749,600.
change: Reduce the qualified dual-readout GRU width from 98 to 90 while preserving all 32 causal frames and expand the classifier input to the concatenated 180-dimensional mean/final representation.
mechanism: Dual-timescale readout with narrower recurrent state
evidence_used: The 98-unit dual-timescale model achieved 87.12% accuracy versus 84.54% for the 98-unit mean-only model, showing a 2.58-point gain and enough margin to test a meaningful eight-unit structural reduction without revisiting failed temporal subsampling.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 96-unit GRU classifying concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 98-unit model’s 906,045,280.
change: Reduce GRU width from 100 to 96 and use the qualified mean-plus-final recurrent readout with a 192-dimensional classifier input.
mechanism: Conservative dual-timescale width reduction
evidence_used: The 98-unit dual-timescale model achieved 87.12%, outperforming the 98-unit mean-only model by 2.58 points; a two-unit reduction is a conservative test of how much of that accuracy margin can be converted into structural MAC savings.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Omitting only the first frame will retain at least 85% accuracy from the current 87.12% baseline while reducing recurrent MACs and executed steps by 3.125%.
change: Process the most recent 31 of 32 frames, preserving uniform spacing and the final input frame.
mechanism: Single-edge-frame causal trimming
evidence_used: Reductions to 24 and 16 frames were too aggressive, but the qualified 98-unit dual-readout model has 2.12 percentage points of accuracy margin; a one-frame edge reduction is the most conservative untested temporal optimization.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4578588665885341461, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 876493380, "recurrent_steps": 25265, "total_inference_macs": 877771300, "training_seconds": 133.51997066708282, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40440371460709834}

RECENT RESULT
hypothesis: A 98-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 31-frame design.
change: Replace the 99-unit mean-only model with the qualified 98-unit mean-plus-final readout and omit the first two input frames.
mechanism: Incremental edge-frame trimming with dual-timescale readout
evidence_used: The 98-unit dual-readout model achieved 87.12% over 32 frames and 86.50% over the most recent 31 frames; its remaining 1.50-point margin motivates testing one additional edge-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4431107283926769866, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 848219400, "recurrent_steps": 24450, "total_inference_macs": 849497320, "training_seconds": 133.96698258304968, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.41693727458181556}

RECENT RESULT
hypothesis: A 98-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 30-frame design.
change: Replace the current 100-unit mean-only GRU with the qualified 98-unit mean-plus-final architecture and omit the first three input frames.
mechanism: Incremental edge-frame trimming with dual-timescale readout
evidence_used: The same 98-unit dual-readout architecture passed at 87.12% with 32 frames, 86.50% with 31 frames, and 85.77% with 30 frames; its remaining 0.77-point margin supports testing one further edge-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4283625901968198271, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 819945420, "recurrent_steps": 23635, "total_inference_macs": 821223340, "training_seconds": 151.8656526671257, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.42601396057503355}

RECENT RESULT
hypothesis: A 96-unit dual-readout GRU processing 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the current 98-unit, 29-frame model.
change: Reduce recurrent width from 98 to 96, resize the mean-plus-final classifier, and restore one input frame for a predicted total of approximately 818.1M inference MACs.
mechanism: Width–time tradeoff with dual-timescale readout
evidence_used: The 98-unit dual-readout model achieved 85.77% at 30 frames, providing 0.77 percentage points of margin; exchanging two hidden units for that additional frame is the smallest width–time tradeoff that improves on the current 821.2M-MAC design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4267216408966407858, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 816825600, "recurrent_steps": 24450, "total_inference_macs": 818077440, "training_seconds": 120.89504654193297, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.3885523509394172}

RECENT RESULT
hypothesis: A 94-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 96-unit model’s 818,077,440.
change: Reduce the GRU hidden width from 98 to 94 and resize the recurrent state, temporal summary, and mean-plus-final classifier accordingly, while preserving the 30-frame schedule and training procedure.
mechanism: Dual-timescale recurrent width refinement
evidence_used: The 96-unit dual-readout model reached 87.48% accuracy over 30 frames with 818,077,440 MACs, leaving 2.48 percentage points of margin; reducing width by two units previously remained qualified and substantially improved cost, motivating another two-unit reduction without removing temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4106386372094120674, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 34264, "peak_hidden_elements": 96768, "recurrent_macs": 786018600, "recurrent_steps": 24450, "total_inference_macs": 787244360, "training_seconds": 124.8966925831046, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4043488016889139}

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
        self.gru = nn.GRU(20, 89, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(178, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 89, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 89, device=device, dtype=dtype)
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
        self.gru = nn.GRU(20, 90, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 90, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 90, device=device, dtype=dtype)
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
        self.gru = nn.GRU(20, 94, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(188, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 94, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 94, device=device, dtype=dtype)
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
