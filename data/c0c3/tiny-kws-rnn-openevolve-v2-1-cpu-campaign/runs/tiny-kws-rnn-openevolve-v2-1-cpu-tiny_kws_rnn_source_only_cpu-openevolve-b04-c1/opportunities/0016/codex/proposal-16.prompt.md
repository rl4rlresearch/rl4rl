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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 192.7032862920314, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.39644080788079944}
prior_hypothesis: An 80-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the verified 84-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Sampling 27 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-step model.
change: Reduce the evenly spaced full-recording frame schedule from 28 steps to 27 steps.
mechanism: Boundary-search uniform causal frame decimation
evidence_used: The 26-step model narrowly missed the target at 84.66%, while 28 steps achieved 85.52%; testing 27 steps directly resolves the remaining compute–accuracy boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6527611327453078597, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1250588160, "recurrent_steps": 22005, "total_inference_macs": 1251422720, "training_seconds": 114.3457491248846, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.46580634555933664}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.8% versus the verified 128-unit, 28-step model.
change: Reduce the GRU and classifier width from 128 to 112 units, resize recurrent state tensors accordingly, and restore the complete 32-frame causal schedule.
mechanism: Width-for-temporal-resolution tradeoff
evidence_used: The 128-unit model achieved 87.24% with 32 frames, while reducing temporal coverage to 27 frames fell to 84.05%; restoring all frames provides accuracy headroom while structural width reduction lowers dense MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 153.1809397500474, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4392715641325968}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.7% versus the verified 112-unit model.
change: Reduce the GRU hidden state and classifier width from 112 to 104 units while preserving the full 32-frame schedule and training procedure.
mechanism: Recurrent width reduction with full temporal resolution
evidence_used: The 112-unit, 32-step model achieved 86.26% accuracy with 1,157,430,400 total MACs, leaving 1.26 percentage points of accuracy headroom; prior frame-decimation results show temporal resolution is valuable, so reducing width while retaining every frame is the most grounded next compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 83.05264195799828, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44775579721649733}

RECENT RESULT
hypothesis: A 100-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the verified 104-unit model.
change: Reduce recurrent width from 104 to 100, preserve all 32 frames, and classify from both the sequence-average output and final hidden output.
mechanism: Width reduction with dual-timescale recurrent readout
evidence_used: The 104-unit full-resolution GRU achieved 85.03% accuracy; retaining every frame while enriching its temporal readout may compensate for a modest width reduction, which structurally lowers recurrent MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4904142805945688328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38248, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 940184000, "training_seconds": 110.3439824578818, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.40625231222141006}

RECENT RESULT
hypothesis: A 96-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 7% versus the verified 100-unit model.
change: Reduce the GRU hidden width from 100 to 96 and resize the classifier and recurrent state tensors accordingly, preserving all 32 frames and the successful dual-timescale readout.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 100-unit dual-readout model achieved 85.64% accuracy with 940,184,000 MACs, outperforming the wider 104-unit model’s 85.03%; this suggests the enriched readout provides enough accuracy headroom to test another modest structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 89.91233604191802, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4015452285485765}

RECENT RESULT
hypothesis: A 92-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 7.5% versus the verified 96-unit model.
change: Reduce the GRU hidden width from 96 to 92 and resize the classifier and recurrent state tensors accordingly, while preserving all 32 frames and the successful dual-timescale readout.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 96-unit dual-readout model achieved 85.89% accuracy with 872,532,480 total MACs, improving on the 100-unit model’s accuracy despite its smaller width; this provides 0.89 percentage points of headroom for another four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4211441137302266104, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807384640, "training_seconds": 101.63187612500042, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.3870472633034174}

RECENT RESULT
hypothesis: A 128-dimensional residual state with an input-conditioned update gate and rank-64 learned recurrent transform will retain at least 85% accuracy while reducing dense MACs by roughly 30% versus the verified 92-unit GRU.
change: Replace the three-gate full-rank GRU with a residual recurrent update using one input projection and a 128→64→128 recurrent bottleneck, while preserving all 32 frames and the successful mean-plus-final readout.
mechanism: Input-gated low-rank residual recurrence
evidence_used: The 92-unit GRU achieved 87.12% accuracy, leaving 2.12 points of headroom, but spends nearly all 807.4M MACs on three full-rank gated transforms. Prior frame decimation harmed accuracy, so this patch preserves temporal resolution and instead challenges the assumption that every gate requires a full-rank recurrent matrix.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2934051375212373776, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23856, "peak_hidden_elements": 131584, "recurrent_macs": 560824320, "recurrent_steps": 26080, "total_inference_macs": 562493440, "training_seconds": 134.95084954216145, "validation_accuracy": 0.8233128834355828, "validation_cross_entropy": 0.49733295089628066}

RECENT RESULT
hypothesis: An 88-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 7.8% versus the verified 92-unit model.
change: Reduce the GRU hidden width from 92 to 88 and resize the classifier and recurrent state tensors accordingly, preserving all 32 frames and the successful dual-timescale readout.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 92-unit model achieved 87.12% accuracy with 807,384,640 MACs—its best observed accuracy among the width-reduction sequence and 2.12 percentage points above threshold—supporting another modest four-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 120.34821412502788, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42309365301775786}

RECENT RESULT
hypothesis: An 84-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 8.1% versus the verified 88-unit model.
change: Reduce the GRU hidden width from 88 to 84 and resize the classifier and recurrent state tensors accordingly, while preserving the full 32-frame schedule and training procedure.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 88-unit dual-readout model achieved 85.77% accuracy with 744,740,480 MACs, and every tested dual-readout width from 92 through 100 exceeded the accuracy requirement; the next four-unit reduction directly continues the established width search.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3570977772028654184, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 684600000, "training_seconds": 27.641658791108057, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.42847105623022913}

RECENT RESULT
hypothesis: An 86-unit GRU with mean-plus-final readout will achieve at least 85% validation accuracy while reducing dense inference MACs by approximately 4% versus the verified 88-unit model.
change: Reduce the GRU hidden width from 88 to 86 and resize the classifier and recurrent state tensors, preserving all 32 frames and the established training procedure.
mechanism: Boundary-search recurrent width reduction
evidence_used: The 88-unit model achieved 85.77% accuracy, while 84 units narrowly missed at 84.79%; testing the midpoint directly resolves the remaining width boundary with a meaningful structural cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3726196272406137448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 713235840, "recurrent_steps": 26080, "total_inference_macs": 714357280, "training_seconds": 159.8941866250243, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4322061175949003}

RECENT RESULT
hypothesis: An 84-unit GRU classified from mean, maximum, and final recurrent outputs will recover the 0.22-point accuracy shortfall of the prior 84-unit model and reach at least 85% while retaining substantially fewer MACs than the verified 88-unit model.
change: Reduce the GRU width to 84 and add a running elementwise maximum of recurrent outputs to the state and classifier features, preserving all 32 causal steps and the existing training procedure.
mechanism: Salience-preserving max-augmented recurrent readout
evidence_used: The 84-unit mean-plus-final model narrowly missed at 84.79%, while the 88-unit version reached 85.77%; adding a parameter-light maximum readout exposes salient transient speech evidence that temporal averaging may suppress, while the 84-unit recurrent core preserves the prior model’s structural MAC reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3573834554244191336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 685147680, "training_seconds": 144.75289745815098, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.39589817538583205}

RECENT RESULT
hypothesis: An 80-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the verified 84-unit model.
change: Reduce the GRU hidden width from 84 to 80 and resize all recurrent state tensors and the three-way classifier input accordingly, while preserving all 32 causal frames and the successful max-augmented readout.
mechanism: Width reduction with salience-preserving temporal readout
evidence_used: The 84-unit mean-plus-final model missed the threshold at 84.79%, but adding the maximum recurrent output raised accuracy to 86.75% with the same recurrent core; that 1.75-point margin motivates testing a four-unit structural reduction without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 192.7032862920314, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.39644080788079944}



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
    """A compact causal GRU with mean and final-state temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return hidden.transpose(0, 1), summary + output, maximum, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        sequence_maximum = outputs.amax(dim=1)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, sequence_maximum),
            sequence_maximum,
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            maximum,
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(32, available_frames)
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


