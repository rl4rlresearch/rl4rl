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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2981222290754815470, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27170, "peak_hidden_elements": 99840, "recurrent_macs": 570271800, "recurrent_steps": 22820, "total_inference_macs": 571536680, "training_seconds": 64.14281774987467, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4372256132722632}
prior_hypothesis: Compressing normalized bands 0–3 from four features to a learned three-dimensional subspace will retain at least 85% validation accuracy while reducing total inference by approximately 3.01M MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 4–5 will retain at least 85% validation accuracy because the lower GRU still receives all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 61 to 60 and replace its six individual low-band skip features with four individual bands plus an average of bands 4–5 in both recurrent execution paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: Six successive upper-skip reductions have met the accuracy requirement, and the latest fold of bands 6–7 achieved 85.28% while saving exactly 3,286,080 recurrent MACs; applying the same localized fold to the next adjacent pair is the most direct test of the compression boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3047642477266039188, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27728, "peak_hidden_elements": 99840, "recurrent_macs": 583005360, "recurrent_steps": 22820, "total_inference_macs": 584270240, "training_seconds": 79.93251470895484, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46573597726646376}

RECENT RESULT
hypothesis: Removing the highest remaining upper-skip aggregate (bands 16–17) will retain at least 85% validation accuracy while saving exactly 3,286,080 recurrent MACs, because the lower GRU still receives all 19 features and prior failure on bands 4–5 suggests low-frequency skips are more important.
change: Reduce the upper GRU input width from 61 to 60 and omit the normalized bands 16–17 average from both recurrent execution paths.
mechanism: Frequency-aware upper-skip pruning
evidence_used: Deleting the highest-band feature only from the upper skip retained 85.77%, whereas folding low-band skip features 4–5 missed the requirement at 84.91%; this motivates preserving low bands and pruning the highest remaining redundant skip feature.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3047642477266039188, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27728, "peak_hidden_elements": 99840, "recurrent_macs": 583005360, "recurrent_steps": 22820, "total_inference_macs": 584270240, "training_seconds": 76.2451717921067, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4468598722680215}

RECENT RESULT
hypothesis: Omitting the highest remaining upper-skip aggregate, bands 14–15, will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 60 to 59 and remove the normalized bands 14–15 average from both recurrent execution paths.
mechanism: Frequency-aware upper-skip pruning
evidence_used: Removing the bands 16–17 upper-skip aggregate achieved 85.52% accuracy and saved exactly 3,286,080 recurrent MACs, while folding lower-frequency bands 4–5 failed; this supports continuing frequency-aware pruning from the high end.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3030501783972820164, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27584, "peak_hidden_elements": 99840, "recurrent_macs": 579719280, "recurrent_steps": 22820, "total_inference_macs": 580984160, "training_seconds": 69.0041252921801, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4619088389390817}

RECENT RESULT
hypothesis: Omitting the highest remaining upper-skip aggregate, bands 12–13, will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 59 to 58 and remove the normalized bands 12–13 average from both recurrent execution paths.
mechanism: Frequency-aware upper-skip pruning
evidence_used: Removing the bands 14–15 aggregate improved accuracy to 85.89% while saving exactly 3,286,080 recurrent MACs; continuing high-to-low pruning is better supported than folding bands 4–5, which failed at 84.91%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3013361090679601140, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27440, "peak_hidden_elements": 99840, "recurrent_macs": 576433200, "recurrent_steps": 22820, "total_inference_macs": 577698080, "training_seconds": 66.39395120786503, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4612052589837759}

RECENT RESULT
hypothesis: Replacing normalized bands 4–5 with a learned 2-to-1 projection will retain at least 85% validation accuracy by learning the discriminative direction that fixed averaging lost, while reducing total inference by approximately 3.24M MACs.
change: Reduce the upper GRU input width from 59 to 58 and compress bands 4–5 through a bias-free linear projection initialized to their average in both recurrent paths.
mechanism: Learned adjacent-band compression
evidence_used: Fixed averaging of bands 4–5 narrowly missed the requirement at 84.91%, while retaining both features achieved 85.89%; a learned scalar projection tests whether the failure came from the fixed equal-weight aggregation rather than the one-dimensional bottleneck.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3013599155864229182, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27442, "peak_hidden_elements": 99840, "recurrent_macs": 576478840, "recurrent_steps": 22820, "total_inference_macs": 577743720, "training_seconds": 63.66614504088648, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.43291244974896953}

RECENT RESULT
hypothesis: Updating the upper GRU 14 times from concatenated adjacent 58-feature inputs will retain at least 85% accuracy while saving approximately 78.9M recurrent MACs, because neither frame’s lower representation or acoustic skip is discarded.
change: Widen the upper GRU input to 116, buffer alternating upper inputs, execute one upper update per pair, track its count separately, and make sequence execution equivalent through vectorized temporal pairing.
mechanism: Lossless pair-concatenated pyramidal upper recurrence
evidence_used: The prior half-rate model narrowly missed at 84.79% after averaging acoustics and discarding every first lower output, while the current model reaches 86.50%. This challenges the assumption that upper recurrence must run every frame while directly addressing the failed design’s temporal information loss.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing only the upper GRU width from 48 to 47 will retain at least 85% validation accuracy while saving approximately 10.49M total inference MACs.
change: Shrink the upper recurrent state by one unit and resize its state initialization and classifier input accordingly.
mechanism: One-unit upper-state contraction
evidence_used: The current 49+48 hierarchy reaches 86.50%, while extensive upper-input compression retained accuracy and reducing the load-bearing lower width failed; this supports testing a small structural reduction in the more redundant upper stage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2958895177367914081, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 26961, "peak_hidden_elements": 98816, "recurrent_macs": 566004460, "recurrent_steps": 22820, "total_inference_macs": 567256300, "training_seconds": 65.60542587493546, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.47396693317436733}

RECENT RESULT
hypothesis: Replacing the separate bands 10–11 and 12–13 upper-skip averages with one learned four-band projection will retain at least 85% validation accuracy while reducing total inference by approximately 3.19M MACs.
change: Reduce the upper GRU input width from 58 to 57 and fuse normalized bands 10–13 through a bias-free learned projection initialized to their average in both recurrent paths.
mechanism: Learned high-band skip fusion
evidence_used: Removing bands 12–13 entirely failed at 84.42%, but learned 2-to-1 compression of bands 4–5 recovered from a fixed-average failure to 86.50%; this tests whether a learned shared high-band direction preserves information that outright pruning lost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2996934592940266242, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27302, "peak_hidden_elements": 99840, "recurrent_macs": 573284040, "recurrent_steps": 22820, "total_inference_macs": 574548920, "training_seconds": 54.59529716707766, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4326310701896808}

RECENT RESULT
hypothesis: Replacing the separate bands 6–7 and 8–9 upper-skip averages with one learned four-band projection will retain at least 85% validation accuracy while reducing total inference by approximately 3.19M MACs.
change: Add a bias-free 4-to-1 projection initialized to averaging, use it to fuse normalized bands 6–9 in both recurrent paths, and reduce the upper GRU input width from 57 to 56.
mechanism: Learned mid-band skip fusion
evidence_used: The learned four-band fusion of bands 10–13 retained 85.15% accuracy while saving approximately 3.19M MACs, and the separate prior folds of bands 6–7 and 8–9 each met the accuracy requirement; this directly tests whether those two already-compressed features can share one learned direction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2980270030016303302, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27162, "peak_hidden_elements": 99840, "recurrent_macs": 570089240, "recurrent_steps": 22820, "total_inference_macs": 571354120, "training_seconds": 62.314682125113904, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.4675696718180838}

RECENT RESULT
hypothesis: A learned 2-to-1 projection of normalized bands 2–3 will retain at least 85% validation accuracy while reducing total inference by approximately 3.24M MACs.
change: Add an averaging-initialized bias-free projection for bands 2–3, reduce the upper GRU input width from 57 to 56, and use the projection in both recurrent paths.
mechanism: Learned low-band adjacent compression
evidence_used: Fixed compression of bands 4–5 missed at 84.91%, while replacing it with a learned projection reached 86.50%; this tests the same information-preserving mechanism on the next adjacent low-band pair.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2980031964831675260, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27160, "peak_hidden_elements": 99840, "recurrent_macs": 570043600, "recurrent_steps": 22820, "total_inference_macs": 571308480, "training_seconds": 66.69610220915638, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4459659131758052}

RECENT RESULT
hypothesis: Replacing four individual upper-skip bands 0–3 with a learned 4-to-3 projection will retain at least 85% validation accuracy while saving approximately 3.01M total MACs; unlike the narrowly failing bands 2–3 scalar projection, the unrestricted three-dimensional subspace can learn mixtures across all four low bands.
change: Add an averaging-initialized bias-free 4-to-3 projection for normalized bands 0–3, use it in both recurrent paths, and reduce the upper GRU input width from 57 to 56.
mechanism: Learned low-band subspace compression
evidence_used: The constrained learned projection of bands 2–3 missed the threshold by only 0.09 percentage points at 84.91%, while the current 57-input model achieved 85.15%; this projection initially reproduces that constrained representation but adds cross-band degrees of freedom at modest MAC cost.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Compressing normalized bands 0–3 from four features to a learned three-dimensional subspace will retain at least 85% validation accuracy while reducing total inference by approximately 3.01M MACs.
change: Add a bias-free 4-to-3 projection initialized to preserve bands 0–1 and average bands 2–3, use it in both recurrent paths, and reduce the upper GRU input width from 57 to 56.
mechanism: Learned low-band subspace compression
evidence_used: The learned bands 2–3 scalar projection missed the threshold by only 0.09 percentage points, while retaining four low-band dimensions achieved 85.15%; a three-dimensional learned projection starts from that tested compression but can recover discriminative mixtures across all four bands.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2981222290754815470, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27170, "peak_hidden_elements": 99840, "recurrent_macs": 570271800, "recurrent_steps": 22820, "total_inference_macs": 571536680, "training_seconds": 64.14281774987467, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4372256132722632}



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
    """A raw-skip hierarchy of two compact causal recurrent stages."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(19)
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_band03 = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.upper_band03.weight.copy_(
                torch.tensor(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 0.5, 0.5),
                    )
                )
            )
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_band1013 = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.upper_band1013.weight, 0.25)
        self.upper_gru = nn.GRU(56, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)

    @staticmethod
    def _fold_bands(frames: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                frames[..., :18],
                frames[..., 18:20].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return lower_hidden, upper_hidden, lower_summary, upper_summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                self.upper_band03(normalized[..., :4]),
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
        upper_output, upper_hidden = self.upper_gru(
            upper_input.unsqueeze(1),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        upper_output = upper_output[:, 0, :]
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_output,
            upper_summary + upper_output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                self.upper_band03(normalized[..., :4]),
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                self.upper_band1013(normalized[..., 10:14]),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
        upper_outputs, upper_hidden = self.upper_gru(
            upper_inputs,
            upper_hidden.transpose(0, 1).contiguous(),
        )
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary + upper_outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        divisor = count.clamp_min(1.0)
        features = torch.cat(
            (
                lower_summary / divisor,
                upper_summary / divisor,
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))


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


