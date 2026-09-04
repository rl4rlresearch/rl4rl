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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5114622186734460013, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 979772625, "recurrent_steps": 20375, "total_inference_macs": 980535465, "training_seconds": 70.04316212516278, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4050237082264906}
prior_hypothesis: A 117-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.56% versus the passing 118-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 127-unit GRU with the passing 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.45%.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 127 while preserving the verified schedule and training procedure.
mechanism: Additive-readout recurrent-width trim
evidence_used: The current 128-unit additive-readout model reached 86.503% accuracy, 1.503 points above the threshold; additive fusion improved the same 25-step schedule from 84.540% with mean-only classification, providing evidence that the stronger readout may support the conservative one-unit width reduction that previously missed under the weaker architecture.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5956692754075935273, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1141142625, "recurrent_steps": 20375, "total_inference_macs": 1141970665, "training_seconds": 136.1219446659088, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40036090312559913}

RECENT RESULT
hypothesis: A 126-unit GRU with the passing 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.46% versus the verified 127-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 127 to 126 while preserving the successful schedule, additive readout, and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 127-unit additive-readout model achieved 86.380% accuracy, 1.380 percentage points above the requirement; this supports testing the smallest remaining structural width reduction without changing its verified temporal coverage or readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5869616161634217595, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 57000, "peak_hidden_elements": 129536, "recurrent_macs": 1124455500, "recurrent_steps": 20375, "total_inference_macs": 1125277020, "training_seconds": 135.7988055839669, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.40380626280614934}

RECENT RESULT
hypothesis: A 125-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.47% versus the passing 126-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 126 to 125 while preserving the successful frame schedule, additive readout, and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 126-unit additive-readout model achieved 85.890% accuracy after the 127-unit version achieved 86.380%; both passed with identical temporal coverage, supporting the next isolated one-channel reduction. The earlier 125-unit result used a weaker mean-only readout and missed by just 0.092 percentage points, so it does not rule out 125 units with the stronger verified additive readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5783177243794182173, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 56173, "peak_hidden_elements": 128512, "recurrent_macs": 1107890625, "recurrent_steps": 20375, "total_inference_macs": 1108705625, "training_seconds": 111.40233691688627, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4153137955928873}

RECENT RESULT
hypothesis: A 124-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.48% versus the passing 125-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 125 to 124 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Successive additive-readout widths of 127, 126, and 125 units all passed; the 125-unit model achieved 85.031% accuracy, making a one-channel trim the smallest isolated structural test despite its narrow margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5697376000555829007, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 1091448000, "recurrent_steps": 20375, "total_inference_macs": 1092256480, "training_seconds": 115.06732637505047, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.39709994076219807}

RECENT RESULT
hypothesis: A 123-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 124-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 124 to 123 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 124-unit model achieved 86.012% accuracy, more than one percentage point above the requirement, and every successive additive-readout width from 128 through 124 has passed; a one-channel trim is therefore the smallest isolated structural cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5612212431919158097, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 54537, "peak_hidden_elements": 126464, "recurrent_macs": 1075127625, "recurrent_steps": 20375, "total_inference_macs": 1075929585, "training_seconds": 98.42873675003648, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4085447440118146}

RECENT RESULT
hypothesis: A 122-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 123-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 123 to 122 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 123 has passed; the 123-unit model achieved 85.276% accuracy, so another isolated one-channel reduction is the most direct test of the remaining width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5527686537884169443, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 53728, "peak_hidden_elements": 125440, "recurrent_macs": 1058929500, "recurrent_steps": 20375, "total_inference_macs": 1059724940, "training_seconds": 108.86680608382449, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.40788664086464727}

RECENT RESULT
hypothesis: A 121-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 122-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 122 to 121 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 122 has passed; the 122-unit model achieved 85.644% accuracy, so another isolated one-channel reduction is the most informative test of the width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5443798318450863045, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 52925, "peak_hidden_elements": 124416, "recurrent_macs": 1042853625, "recurrent_steps": 20375, "total_inference_macs": 1043642545, "training_seconds": 100.74323433404788, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40282296958876534}

RECENT RESULT
hypothesis: A 120-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 121-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 121 to 120 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 121 has passed, and the latest 121-unit model achieved 86.503% accuracy, providing a 1.503-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5360547773619238903, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1026900000, "recurrent_steps": 20375, "total_inference_macs": 1027682400, "training_seconds": 90.98146774992347, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4024813037708493}

RECENT RESULT
hypothesis: A 119-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.54% versus the passing 120-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 120 to 119 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 120 has passed, and the 120-unit model achieved 86.135% accuracy, providing a 1.135-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5277934903389297017, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 51337, "peak_hidden_elements": 122368, "recurrent_macs": 1011068625, "recurrent_steps": 20375, "total_inference_macs": 1011844505, "training_seconds": 79.4365180421155, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4005632459020322}

RECENT RESULT
hypothesis: A 118-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.55% versus the passing 119-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 119 to 118 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 119 has passed, and the latest 119-unit model achieved 86.135% accuracy, leaving a 1.135-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5195959707761037387, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 50552, "peak_hidden_elements": 121344, "recurrent_macs": 995359500, "recurrent_steps": 20375, "total_inference_macs": 996128860, "training_seconds": 50.26036816602573, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.41964874267578123}

RECENT RESULT
hypothesis: A 117-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.56% versus the passing 118-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 118 to 117 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 118 has passed; the 118-unit model achieved 85.153% accuracy, making the next isolated one-channel reduction the most informative test of the recurrent-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5114622186734460013, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 979772625, "recurrent_steps": 20375, "total_inference_macs": 980535465, "training_seconds": 70.04316212516278, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4050237082264906}

RECENT RESULT
hypothesis: A 116-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 117-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 117 to 116 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 117 has passed; the current 117-unit model achieved 85.767% accuracy, providing evidence for one more isolated channel reduction as the next test of the width boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5033922340309564895, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49000, "peak_hidden_elements": 119296, "recurrent_macs": 964308000, "recurrent_steps": 20375, "total_inference_macs": 965064320, "training_seconds": 73.46372124995105, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4151908453256806}



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
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
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
        temporal_mean = summary / count.clamp_min(1.0)
        return self.classifier(temporal_mean + hidden[:, 0, :])

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(i * (available_frames - 1) / (steps - 1))
            for i in range(steps)
        ]
        return schedule[1:-3] + schedule[-1:] if len(schedule) > 4 else schedule


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


