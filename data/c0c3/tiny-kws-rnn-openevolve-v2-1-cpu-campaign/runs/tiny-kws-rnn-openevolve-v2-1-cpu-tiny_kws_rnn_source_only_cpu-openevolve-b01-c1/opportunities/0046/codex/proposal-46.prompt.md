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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3064783170559258212, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27872, "peak_hidden_elements": 99840, "recurrent_macs": 586291440, "recurrent_steps": 22820, "total_inference_macs": 587556320, "training_seconds": 62.073358125053346, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46391484547246453}
prior_hypothesis: Folding normalized upper-skip bands 6–7 will retain at least 85% validation accuracy because the lower GRU still receives all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing only the upper GRU from 49 to 48 units will preserve at least 85% validation accuracy while lowering total inference MACs, because the verified 49+49 model reached 86.50% and prior failure from reducing the lower stage indicates lower-stage capacity is more load-bearing.
change: Preserve the 49-unit lower GRU, folded 19-band input, and 28-step schedule while reducing the upper recurrent state to 48 units and adjusting its state tensors and classifier width.
mechanism: Incremental upper-stage width compression
evidence_used: The current 49+49 hierarchy achieved 86.50%, providing 1.50 percentage points of margin; by contrast, the 48+50 hierarchy achieved only 84.54%, motivating compression of the upper rather than lower stage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3184768023611791380, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28880, "peak_hidden_elements": 99840, "recurrent_macs": 609294000, "recurrent_steps": 22820, "total_inference_macs": 610558880, "training_seconds": 53.20526549988426, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4449613793495974}

RECENT RESULT
hypothesis: Reducing the upper GRU from 48 to 47 units will preserve at least 85% validation accuracy while lowering exact inference MACs below the verified 610.56M, because the preceding 49-to-48 reduction retained 86.01% accuracy.
change: Preserve the 49-unit lower GRU, folded 19-band input, 28-step schedule, summaries, and training procedure while reducing the upper recurrent state by one unit and adjusting its classifier dimensions.
mechanism: Incremental upper-stage width compression
evidence_used: The 49+48 hierarchy achieved 86.01% accuracy, leaving 1.01 percentage points of margin, while the failed 48+50 result indicates the lower stage is more sensitive to compression than the upper stage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3126493067346055649, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28369, "peak_hidden_elements": 98816, "recurrent_macs": 598135020, "recurrent_steps": 22820, "total_inference_macs": 599386860, "training_seconds": 54.5244680410251, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.45730922207510544}

RECENT RESULT
hypothesis: Removing only the folded highest-band feature from the upper GRU’s raw acoustic skip will preserve at least 85% validation accuracy because the lower GRU still receives all 19 features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input from 68 to 67 features and pass it the first 18 normalized acoustic features plus the complete 49-unit lower recurrent output in both execution paths.
mechanism: Redundant upper-stage spectral-skip pruning
evidence_used: The 49+48 hierarchy achieved 86.01% accuracy, while removing a spectral feature from the entire recurrent path failed; retaining all spectral information in the more load-bearing lower stage while pruning only its redundant direct copy is a narrower structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167627330318572356, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28736, "peak_hidden_elements": 99840, "recurrent_macs": 606007920, "recurrent_steps": 22820, "total_inference_macs": 607272800, "training_seconds": 48.76711895898916, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43410169917381614}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 16–17 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 67 to 66 and replace its 18-feature acoustic skip with the first 16 normalized bands plus the average of bands 16–17 in both recurrent paths.
mechanism: Upper-skip adjacent-band folding
evidence_used: Removing the folded highest-band feature only from the upper skip retained 85.77% accuracy and saved 3,286,080 recurrent MACs; prior whole-path experiments also showed adjacent-band folding preserves information better than deletion.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3150486637025353332, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28592, "peak_hidden_elements": 99840, "recurrent_macs": 602721840, "recurrent_steps": 22820, "total_inference_macs": 603986720, "training_seconds": 56.07529004220851, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4429512913241708}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 14–15 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 66 to 65 and replace its 17-feature acoustic skip with the first 14 normalized bands plus averages of bands 14–15 and 16–17 in both recurrent paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: Folding upper-skip bands 16–17 retained 85.77% accuracy and saved 3,286,080 recurrent MACs; applying the same information-preserving reduction to the next adjacent pair is the narrowest supported structural compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3133345943732134308, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28448, "peak_hidden_elements": 99840, "recurrent_macs": 599435760, "recurrent_steps": 22820, "total_inference_macs": 600700640, "training_seconds": 48.845527875004336, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4513846368146089}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 12–13 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 65 to 64 and replace its 16-feature acoustic skip with the first 12 normalized bands plus averages of bands 12–13, 14–15, and 16–17 in both recurrent paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: The two preceding upper-skip folds each preserved at least 85.64% accuracy while saving 3,286,080 recurrent MACs; repeating the same localized compression on the next adjacent pair is the narrowest evidence-supported reduction.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Updating the upper GRU from averaged adjacent acoustic frames and every second lower-GRU output will retain at least 85% validation accuracy while reducing total inference MACs from 600.70M to approximately 415.04M.
change: Keep the load-bearing lower GRU at all 28 causal steps, but clock the upper GRU once per frame pair, add pending-acoustic and upper-update-count state, and make sequence execution exactly match repeated recurrent steps.
mechanism: Pyramidal two-frame upper recurrence
evidence_used: The 49+48 hierarchy reached 86.01%, upper-skip pruning and folding repeatedly retained at least 85%, and reducing the lower width failed; this challenges the shared assumption that the comparatively redundant upper stage must execute at every frame while preserving full-rate lower recurrence.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2164896772665267588, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28448, "peak_hidden_elements": 108544, "recurrent_macs": 413772240, "recurrent_steps": 22820, "total_inference_macs": 415037120, "training_seconds": 45.23730045789853, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4628247781765242}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 12–13 will preserve at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs, because the lower GRU continues to receive all 19 folded spectral features.
change: Reduce the upper GRU input width from 65 to 64 and replace its 16-feature acoustic skip with the first 12 normalized bands plus averages of bands 12–13, 14–15, and 16–17 in both recurrent paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: The two preceding upper-skip folds each preserved at least 85.64% accuracy while saving 3,286,080 recurrent MACs; the previous attempt at this exact next fold was not evaluated because its search text matched twice.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3116205250438915284, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28304, "peak_hidden_elements": 99840, "recurrent_macs": 596149680, "recurrent_steps": 22820, "total_inference_macs": 597414560, "training_seconds": 49.012873667059466, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4473254525588334}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 10–11 into one feature will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs, because the lower GRU still receives all 19 folded spectral features.
change: Reduce the upper GRU input width from 64 to 63 and replace its 15-feature acoustic skip with the first 10 normalized bands plus averages of bands 10–11, 12–13, 14–15, and 16–17 in both recurrent paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: Three successive upper-skip folds preserved at least 85.40% accuracy and each saved exactly 3,286,080 recurrent MACs; applying the same localized, information-preserving reduction to the next adjacent pair is the narrowest supported compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3099064557145696260, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28160, "peak_hidden_elements": 99840, "recurrent_macs": 592863600, "recurrent_steps": 22820, "total_inference_macs": 594128480, "training_seconds": 56.83195999986492, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44030621300445744}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 8–9 will retain at least 85% validation accuracy while reducing recurrent inference by exactly 3,286,080 MACs, because the lower GRU still receives all 19 folded spectral features.
change: Reduce the upper GRU input width from 63 to 62 and replace its 14-feature acoustic skip with the first eight normalized bands plus averages of bands 8–9, 10–11, 12–13, 14–15, and 16–17 in both recurrent paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: Four successive upper-skip adjacent-band folds preserved at least 85.40% accuracy and each saved exactly 3,286,080 recurrent MACs; the current model recovered to 85.52%, making the next identical localized fold the narrowest evidence-supported compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3081923863852477236, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28016, "peak_hidden_elements": 99840, "recurrent_macs": 589577520, "recurrent_steps": 22820, "total_inference_macs": 590842400, "training_seconds": 43.1869987079408, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46442407713345957}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 6–7 will retain at least 85% validation accuracy because the lower GRU still receives all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 62 to 61 and replace its eight individual low-band skip features with six individual bands plus an average of bands 6–7 in both recurrent execution paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: Five successive upper-skip reductions preserved at least 85% accuracy and each adjacent-band fold saved exactly 3,286,080 recurrent MACs; applying the same localized fold to the next pair directly tests the compression boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3064783170559258212, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27872, "peak_hidden_elements": 99840, "recurrent_macs": 586291440, "recurrent_steps": 22820, "total_inference_macs": 587556320, "training_seconds": 62.073358125053346, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46391484547246453}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 4–5 will retain at least 85% validation accuracy because the lower GRU still receives all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 61 to 60 and replace its six individual low-band skip features with four individual bands plus an average of bands 4–5 in both recurrent execution paths.
mechanism: Progressive upper-skip adjacent-band folding
evidence_used: Six successive upper-skip reductions have met the accuracy requirement, and the latest fold of bands 6–7 achieved 85.28% while saving exactly 3,286,080 recurrent MACs; applying the same localized fold to the next adjacent pair is the most direct test of the compression boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3047642477266039188, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27728, "peak_hidden_elements": 99840, "recurrent_macs": 583005360, "recurrent_steps": 22820, "total_inference_macs": 584270240, "training_seconds": 79.93251470895484, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46573597726646376}



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
        self.upper_gru = nn.GRU(61, 48, num_layers=1, batch_first=True)
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
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
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
                normalized[..., :6],
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
                normalized[..., 14:16].mean(dim=-1, keepdim=True),
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
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


