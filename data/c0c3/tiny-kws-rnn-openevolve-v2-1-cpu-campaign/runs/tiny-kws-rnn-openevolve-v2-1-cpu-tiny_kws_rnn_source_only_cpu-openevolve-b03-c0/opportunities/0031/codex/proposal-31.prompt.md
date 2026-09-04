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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4207326010294909962, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 806048040, "recurrent_steps": 23635, "total_inference_macs": 806595720, "training_seconds": 88.400646999944, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45559915179855254}
prior_hypothesis: Omitting frames 0–2 will retain at least 85% validation accuracy while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 3.33%.

## Recent verification evidence

RECENT RESULT
hypothesis: Folding all 98 recurrent summary coordinates into a 96-dimensional readout will retain at least 85% accuracy while reducing classifier inference by 6,520 MACs versus the verified 97-coordinate design.
change: Reduce the classifier input from 97 to 96 and fold the final two summary coordinates into the first two with variance-preserving fixed additions, preserving a direct classification path from every recurrent coordinate.
mechanism: Fixed orthogonal recurrent-coordinate folding
evidence_used: The cropped 96-coordinate readout narrowly failed at 84.66% despite achieving lower cross-entropy than the passing 97-coordinate model; retaining information and direct classifier gradients from the two previously discarded coordinates specifically targets that failure while keeping the lower 96-input MAC count.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722669116634940256, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36096, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905393280, "training_seconds": 111.14206658303738, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4578865753361053}

RECENT RESULT
hypothesis: A seven-output classifier with the eighth logit derived as the negative sum will retain at least 85% accuracy while reducing exact classifier MACs by 79,055, because it preserves all seven independent softmax decision dimensions.
change: Replace the 97-to-8 classifier with a 97-to-7 classifier and construct an eight-logit zero-sum output without another learned matrix operation.
mechanism: Zero-sum softmax logit parameterization
evidence_used: The 97-coordinate readout passed at 85.52%, while reducing its input to 96 failed; exploiting softmax’s redundant common-logit dimension reduces MACs without discarding another recurrent feature.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722290763037942031, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36006, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905320745, "training_seconds": 112.10978870815597, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.44277687306784413}

RECENT RESULT
hypothesis: A 97-unit GRU with the seven-output zero-sum classifier will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.86% versus the current 98-unit model.
change: Reduce the GRU hidden state and temporal summary from 98 to 97 units, retaining all 32 frames and classifying from the complete 97-dimensional summary.
mechanism: Zero-sum-head-compensated recurrent compression
evidence_used: The 97-unit model with a conventional eight-output head narrowly missed at 84.42%, while changing the 98-unit model from an eight-output head to the seven-output zero-sum parameterization improved accuracy from 85.52% to 86.26%; a comparable gain would place the compressed model above 85%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634546737846463780, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35355, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888499145, "training_seconds": 104.70380000001751, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4641465731193683}

RECENT RESULT
hypothesis: Classifying from 96 recurrent summary coordinates with the seven-output zero-sum head will retain at least 85% validation accuracy while reducing exact classifier MACs; the zero-sum head improved the 97-coordinate model from 85.52% to 86.26%, enough to plausibly recover the cropped 96-coordinate model’s 84.66% result.
change: Reduce the seven-output classifier input width from 97 to 96 and omit one additional recurrent summary coordinate, while preserving the 98-unit GRU and all 32 causal steps.
mechanism: Zero-sum-head-assisted recurrent readout pruning
evidence_used: The 96-coordinate conventional head missed narrowly at 84.66%, while the seven-output zero-sum parameterization raised the passing 97-coordinate design to 86.26% and reduced MACs, motivating their combination.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722261004889863519, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35999, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905315040, "training_seconds": 89.66861245804466, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4541908217354055}

RECENT RESULT
hypothesis: Replacing one learned classifier row with a fixed, scaled readout of the two unused GRU summary coordinates will retain at least 85% validation accuracy while reducing classifier MACs by 14.3% and preserving seven independent softmax decision dimensions.
change: Reduce the classifier from seven outputs to six, construct a seventh logit from the final two recurrent summary coordinates, and continue deriving the eighth as the negative sum.
mechanism: Recurrent-coordinate logit substitution
evidence_used: The seven-output zero-sum head improved accuracy to 86.26% while removing one dense output row, and the current 96-coordinate version still passes at 85.28%; the GRU’s two omitted coordinates provide a no-dense-MAC source for another trainable decision score.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4721852893144786782, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35902, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905236800, "training_seconds": 76.31658116704784, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4696192454706672}

RECENT RESULT
hypothesis: Combining the final two mel bands into one input while retaining the seven-output zero-sum head will achieve at least 85% validation accuracy and reduce recurrent MACs by about 0.85%.
change: Reduce the GRU input and input normalization from 20 to 19 features by averaging the two highest mel bands at every recurrent step.
mechanism: Zero-sum-head-assisted adjacent-band spectral pooling
evidence_used: The prior 19-input averaged-band model narrowly missed at 84.91%, while the seven-output zero-sum head improved the 97-coordinate design from 85.52% to 86.26% and enabled the current 96-coordinate design to pass at 85.28%.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4682266053872352503, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35703, "peak_hidden_elements": 100864, "recurrent_macs": 897099840, "recurrent_steps": 26080, "total_inference_macs": 897647520, "training_seconds": 79.88326287502423, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.47828744525558375}

RECENT RESULT
hypothesis: Averaging mel bands 16–17 in addition to the already pooled bands 18–19 will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.85%.
change: Reduce the normalized GRU input from 19 to 18 features using two separate adjacent-band averages for the four highest mel bands.
mechanism: Incremental adjacent-band spectral pooling
evidence_used: Pooling bands 18–19 with the seven-output zero-sum head passed at 85.52%; this tests one additional structural input reduction while preserving more high-frequency detail than averaging all four bands together.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4642271102854841487, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 889432320, "recurrent_steps": 26080, "total_inference_macs": 889980000, "training_seconds": 79.98308766703121, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.46927257139989936}

RECENT RESULT
hypothesis: Averaging only mel bands 17–19 into one feature will retain at least 85% validation accuracy while reducing recurrent MACs to the prior 18-input level, because it concentrates both removed spectral degrees of freedom in the least-resolved high-frequency region.
change: Reduce the GRU input from 19 to 18 features by preserving mel bands 0–16 individually and averaging the final three bands.
mechanism: Concentrated highest-band spectral pooling
evidence_used: The 19-input model pooling bands 18–19 passed at 85.52%, whereas separately pooling bands 16–17 caused the 18-input model to fail at 84.29%; pooling the final three bands tests the same structural reduction without altering the more informative band 16.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4642271102854841487, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 889432320, "recurrent_steps": 26080, "total_inference_macs": 889980000, "training_seconds": 78.90725358412601, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4694507996728815}

RECENT RESULT
hypothesis: Averaging mel bands 0–1 while retaining the successful 18–19 average will achieve at least 85% validation accuracy at the lower 18-input inference cost, because it preserves bands 2–17 that were damaged by both failed 18-input designs.
change: Reduce the GRU input from 19 to 18 features by separately averaging the lowest and highest mel-band pairs while preserving every middle band.
mechanism: Dual-edge adjacent-band spectral pooling
evidence_used: The 19-input model pooling bands 18–19 passed at 85.52%, while further pooling bands 16–17 scored 84.29% and pooling bands 17–19 scored 84.54%; this motivates moving the additional compression away from bands 16–17.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4642271102854841487, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 889432320, "recurrent_steps": 26080, "total_inference_macs": 889980000, "training_seconds": 61.24184108292684, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45050564631362633}

RECENT RESULT
hypothesis: Omitting only input frame 0 will preserve at least 85% validation accuracy while reducing recurrent execution from 32 to 31 steps and cutting recurrent MACs by 3.125%.
change: Keep the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged, but schedule frames 1–31 for 32-frame inputs.
mechanism: Earliest-frame recurrent step pruning
evidence_used: The current design passes at 85.52% accuracy; removing one recurrent step preserves all learned dimensions and the remaining 31-frame causal sequence, while offering a larger structural MAC reduction than another single-feature or classifier-coordinate reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4497289405334864312, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 861637560, "recurrent_steps": 25265, "total_inference_macs": 862185240, "training_seconds": 58.394637958845124, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4544558659652991}

RECENT RESULT
hypothesis: Omitting frames 0 and 1 will retain at least 85% validation accuracy while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 3.23% versus the current design.
change: Schedule frames 2–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental earliest-frame recurrent step pruning
evidence_used: Omitting frame 0 improved accuracy from 85.52% to 85.64% while reducing execution to 31 steps, indicating that the earliest context is expendable and motivating an incremental test of the adjacent frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4352307707814887137, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 833842800, "recurrent_steps": 24450, "total_inference_macs": 834390480, "training_seconds": 96.66559787490405, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4559282361364072}

RECENT RESULT
hypothesis: Omitting frames 0–2 will retain at least 85% validation accuracy while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 3.33%.
change: Schedule frames 3–31 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental earliest-frame recurrent step pruning
evidence_used: Successive removal of frame 0 and then frame 1 preserved passing accuracy at 85.64% and 85.40%, respectively, while each reduced recurrent execution by one step; removing the next-earliest frame is the most direct test of the remaining temporal redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4207326010294909962, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 806048040, "recurrent_steps": 23635, "total_inference_macs": 806595720, "training_seconds": 88.400646999944, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45559915179855254}



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
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        pooled_frame = torch.cat(
            (
                frame[:, :2].mean(dim=1, keepdim=True),
                frame[:, 2:18],
                frame[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        output, hidden = self.gru(
            self.input_norm(pooled_frame).unsqueeze(1),
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
        pooled_frames = torch.cat(
            (
                frames[:, :, :2].mean(dim=2, keepdim=True),
                frames[:, :, 2:18],
                frames[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        outputs, hidden = self.gru(
            self.input_norm(pooled_frames), hidden.transpose(0, 1).contiguous()
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
        averaged = summary / count.clamp_min(1.0)
        logits = self.classifier(averaged[:, :96])
        return torch.cat((logits, -logits.sum(dim=1, keepdim=True)), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(3, available_frames))


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


