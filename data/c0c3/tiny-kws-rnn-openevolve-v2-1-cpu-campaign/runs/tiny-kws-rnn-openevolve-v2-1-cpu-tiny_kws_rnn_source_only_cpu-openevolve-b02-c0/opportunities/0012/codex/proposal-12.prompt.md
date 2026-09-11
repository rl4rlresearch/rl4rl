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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4621942036471711615, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36480, "peak_hidden_elements": 101888, "recurrent_macs": 885437190, "recurrent_steps": 25265, "total_inference_macs": 886082670, "training_seconds": 181.31405062507838, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4569374412115366}
prior_hypothesis: Merging only the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by 7,503,705 versus the verified 99-unit, 31-step design.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing the 16 odd-indexed frames will retain at least 85% validation accuracy while approximately halving recurrent MACs and executed recurrent steps from the 87.24%-accurate baseline.
change: Change the causal frame schedule from all 32 frames to every second frame, preserving the final frame.
mechanism: Uniform causal frame decimation
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, leaving a 2.24-point margin for temporal decimation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 73.47678979113698, "validation_accuracy": 0.8257668711656442, "validation_cross_entropy": 0.5140387599453604}

RECENT RESULT
hypothesis: Processing 24 uniformly distributed frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 25% versus the 32-frame baseline.
change: Skip every fourth input frame while retaining the final frame, yielding 24 recurrent steps for the fixed 32-frame input.
mechanism: Three-of-four causal frame subsampling
evidence_used: The 32-step model achieved 87.24% accuracy, while aggressive 16-step decimation achieved 82.58%; testing the 24-step midpoint is the most informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 89.96426475001499, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.4546638910024444}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 22% versus the 128-unit baseline.
change: Reduce the GRU state, temporal summary, and classifier input width from 128 to 112 without temporal subsampling.
mechanism: Width-reduced full-resolution recurrence
evidence_used: The 128-unit, 32-step model achieved 87.24% accuracy, while reducing temporal coverage to 24 or 16 steps failed; preserving every frame while trimming recurrent width targets compute without discarding causal observations.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 116.78561591706239, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4414740345960746}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13% versus the verified 112-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 112 to 104 while preserving full causal temporal resolution and the existing training procedure.
mechanism: Incremental recurrent-width reduction
evidence_used: The 112-unit full-resolution GRU achieved 86.13% accuracy, whereas reducing temporal coverage failed at both 24 and 16 steps; this motivates preserving all 32 frames and testing a conservative 8-unit width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 112.49923920817673, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by roughly 14% versus the verified 104-unit model.
change: Reduce the GRU hidden state and classifier input width from 104 to 96 while preserving all frames and the existing training procedure.
mechanism: Incremental full-resolution GRU width reduction
evidence_used: Reducing width from 112 to 104 lowered accuracy by only 0.25 percentage points, from 86.13% to 85.89%, while reducing recurrent MACs from 1,156,700,160 to 1,008,983,040; this supports testing the next conservative eight-unit reduction without discarding temporal information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 125.66230945801362, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.44714848805058954}

RECENT RESULT
hypothesis: A 100-unit full-resolution GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the verified 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 100 while preserving all 32 causal frames and the established training procedure.
mechanism: Boundary-seeking recurrent-width reduction
evidence_used: The 104-unit model achieved 85.89% accuracy, while the 96-unit model achieved 84.29%; testing the midpoint directly probes the narrowest width likely to satisfy the threshold without confounding temporal or training changes.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 114.55130762490444, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4471566978407784}

RECENT RESULT
hypothesis: A 99-unit full-resolution GRU will retain at least 85% validation accuracy while reducing exact recurrent MACs below the verified 100-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 99 while preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width boundary reduction
evidence_used: The 100-unit model achieved 85.40% accuracy while the 96-unit model achieved 84.29%; a one-unit reduction is the smallest available probe of the accuracy boundary and preserves full temporal resolution, unlike the unsuccessful 24- and 16-step designs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 111.973967500031, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45291225690783166}

RECENT RESULT
hypothesis: A 98-unit full-resolution GRU will retain at least 85% validation accuracy while reducing exact recurrent and classifier MACs below the verified 99-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 99 to 98 while preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width boundary reduction
evidence_used: The 99-unit model achieved 85.40% accuracy, while the 96-unit model achieved 84.29%; testing 98 units is the most informative next probe of the minimum viable width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 92.19683279120363, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4501637090203221}

RECENT RESULT
hypothesis: Processing 31 frames with the verified 99-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125%.
change: Omit only the earliest input frame while retaining the remaining 31 causal frames, including the final frame.
mechanism: Single leading-frame omission
evidence_used: The 99-unit 32-step model achieved 85.40% accuracy; prior 24- and 16-step schedules were too aggressive, so removing one likely low-information boundary frame is the smallest temporal-cost probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4661082503522968417, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893586375, "training_seconds": 107.63260129187256, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4587176457504553}

RECENT RESULT
hypothesis: Pooling the verified 99-unit recurrent summary into 33 groups before classification will retain at least 85% accuracy while reducing exact classifier MACs and learned parameters.
change: Preserve the 99-unit GRU and 31-step schedule, but average each consecutive group of three recurrent features and replace the 99-to-8 classifier with a 33-to-8 classifier.
mechanism: Parameter-free grouped recurrent readout
evidence_used: The 99-unit, 31-step model achieved 85.03% accuracy, while even a 98-unit model using all 32 frames achieved only 84.54%; this motivates preserving recurrent capacity and structurally reducing the readout instead.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4658837888925046369, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36249, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893156055, "training_seconds": 110.998170417035, "validation_accuracy": 0.8245398773006135, "validation_cross_entropy": 0.5709395519794862}

RECENT RESULT
hypothesis: Merging only the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by 7,503,705 versus the verified 99-unit, 31-step design.
change: Preserve the 99-unit state, 31-frame schedule, and full classifier while replacing the GRU’s 20-dimensional input with 19 features formed by retaining bands 0–17 and averaging bands 18–19.
mechanism: Adjacent high-frequency mel-band pooling
evidence_used: The 99-unit, 31-step model met the threshold at 85.03%, whereas reducing hidden width to 98 or compressing the recurrent readout failed; this motivates preserving recurrent capacity and readout information while conservatively exploiting locality in the ordered mel inputs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4621942036471711615, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36480, "peak_hidden_elements": 101888, "recurrent_macs": 885437190, "recurrent_steps": 25265, "total_inference_macs": 886082670, "training_seconds": 181.31405062507838, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4569374412115366}



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
        self.gru = nn.GRU(19, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)

    def _input_features(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_band = normalized[..., 18:].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :18], high_band), dim=-1)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self._input_features(frame).unsqueeze(1),
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
            self._input_features(frames), hidden.transpose(0, 1).contiguous()
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


