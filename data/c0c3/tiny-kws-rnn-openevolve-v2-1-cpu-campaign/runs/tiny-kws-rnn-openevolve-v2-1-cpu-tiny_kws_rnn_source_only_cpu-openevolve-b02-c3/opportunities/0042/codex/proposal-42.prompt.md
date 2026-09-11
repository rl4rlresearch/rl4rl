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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2656212299687902993, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24688, "peak_hidden_elements": 82432, "recurrent_macs": 508315500, "recurrent_steps": 22005, "total_inference_macs": 509228300, "training_seconds": 56.25552050000988, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4362127292375623}
prior_hypothesis: A 70-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 521,190,055 to approximately 509,228,300.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2718606633880506113, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 25253, "peak_hidden_elements": 83456, "recurrent_macs": 520264215, "recurrent_steps": 22005, "total_inference_macs": 521190055, "training_seconds": 42.96981199993752, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4102224244661858}
prior_hypothesis: A 71-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 533,283,840 to approximately 521,190,055.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3085069725220775893, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27573, "peak_hidden_elements": 87552, "recurrent_macs": 590467500, "recurrent_steps": 22820, "total_inference_macs": 591445500, "training_seconds": 46.008248457917944, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4075592041015625}
prior_hypothesis: A 75-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will retain at least 85% accuracy while reducing total inference MACs from 604,534,400 to approximately 591,445,500.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2594506654065116709, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24129, "peak_hidden_elements": 81408, "recurrent_macs": 496498815, "recurrent_steps": 22005, "total_inference_macs": 497398575, "training_seconds": 49.62836266704835, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.44202573811349694}
prior_hypothesis: A 69-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 509,228,300 to approximately 497,398,575.

## Recent verification evidence

RECENT RESULT
hypothesis: Explicit frame-to-frame spectral motion will let a 76-unit GRU over 28 frames achieve at least 85% accuracy with approximately 604,534,400 total inference MACs, below the qualified 26-step model’s 605,316,800.
change: Replace the absolute-frame-only 88-unit GRU with a 76-unit GRU receiving each normalized frame and its causal first difference; retain mean-plus-final prediction and restore the qualified 28-frame window.
mechanism: Causal delta-augmented compact GRU
evidence_used: The 88-unit 28-frame model reached 85.89%, while the 86-unit version missed by only one example. This tests the load-bearing assumption that dense recurrent width must discover local acoustic motion internally, reallocating capacity from hidden mixing to explicit parameter-free delta features.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3153343419240889388, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28168, "peak_hidden_elements": 88576, "recurrent_macs": 603543360, "recurrent_steps": 22820, "total_inference_macs": 604534400, "training_seconds": 48.35405945801176, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4143990545916411}

RECENT RESULT
hypothesis: A 75-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will retain at least 85% accuracy while reducing total inference MACs from 604,534,400 to approximately 591,445,500.
change: Replace the 88-unit absolute-frame GRU with a 75-unit delta-augmented GRU, resize its state and dual readout, and use the most recent 28 frames.
mechanism: Single-unit refinement of delta-augmented recurrence
evidence_used: The 76-unit delta-augmented model achieved 85.52% accuracy and 0.4144 cross-entropy at 604,534,400 MACs; its qualification and stronger cross-entropy motivate a conservative one-unit width reduction while preserving its proven temporal span and features.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3085069725220775893, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27573, "peak_hidden_elements": 87552, "recurrent_macs": 590467500, "recurrent_steps": 22820, "total_inference_macs": 591445500, "training_seconds": 46.008248457917944, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4075592041015625}

RECENT RESULT
hypothesis: A 74-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will achieve at least 85% validation accuracy while reducing predicted total inference MACs from 591,445,500 to approximately 578,493,520.
change: Replace the current 88-unit absolute-frame GRU with a 74-unit delta-augmented GRU, resize its recurrent state and dual readout, and process the most recent 28 frames.
mechanism: Single-unit refinement of delta-augmented recurrence
evidence_used: The 75-unit delta-augmented model achieved 86.13% accuracy and 0.4076 cross-entropy at 591,445,500 MACs, improving upon the qualified 76-unit version; its 1.13-point margin motivates the next conservative one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3017510226754546524, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 26984, "peak_hidden_elements": 86528, "recurrent_macs": 577528560, "recurrent_steps": 22820, "total_inference_macs": 578493520, "training_seconds": 48.24753158399835, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.44222420534473256}

RECENT RESULT
hypothesis: The qualified 74-unit delta-augmented GRU will retain at least 85% validation accuracy using 27 evenly distributed frames across the latest 28-frame window, while reducing recurrent steps from 28 to 27 and total inference MACs to approximately 557.9 million.
change: Replace the contiguous 28-frame schedule with 27 causally ordered samples spanning the same latest 28-frame window.
mechanism: Span-preserving temporal thinning with causal deltas
evidence_used: The current 74-unit delta model achieved 86.01% accuracy with 28 frames, leaving 1.01 points of margin; separately, span-preserving thinning allowed the 88-unit model to qualify at 26 steps and produced better cross-entropy than its contiguous 27-frame schedule.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2909921767877216489, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 26984, "peak_hidden_elements": 86528, "recurrent_macs": 556902540, "recurrent_steps": 22005, "total_inference_macs": 557867500, "training_seconds": 40.86980366706848, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.43395821975052723}

RECENT RESULT
hypothesis: A 74-unit delta-augmented GRU using 26 evenly distributed frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 557,867,500 to approximately 537,241,480.
change: Install the qualified 74-unit recurrent width and reduce its schedule from 27 to 26 causally ordered, span-preserving frames.
mechanism: Incremental span-preserving temporal thinning with causal deltas
evidence_used: The 74-unit delta GRU achieved 85.89% accuracy at 27 steps and 86.01% at 28 steps; the small loss from thinning and 0.89-point remaining margin motivate testing one fewer step, which saves more MACs than another one-unit width reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2802333308999886454, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 26984, "peak_hidden_elements": 86528, "recurrent_macs": 536276520, "recurrent_steps": 21190, "total_inference_macs": 537241480, "training_seconds": 43.16153083299287, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4356324927207151}

RECENT RESULT
hypothesis: A 73-unit delta-augmented GRU using 27 frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 557,867,500 to approximately 545,509,655.
change: Reduce the GRU width from 76 to 73, resize its state and classifier, and adopt the qualified 27-step span-preserving schedule.
mechanism: Single-unit refinement with span-preserving temporal thinning
evidence_used: The 74-unit, 27-step model achieved 85.89% accuracy, while reducing width from 75 to 74 at 28 steps retained 86.01%; this supports testing the adjacent 73-unit boundary without sacrificing temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2845461367975162861, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 26401, "peak_hidden_elements": 85504, "recurrent_macs": 544557735, "recurrent_steps": 22005, "total_inference_macs": 545509655, "training_seconds": 40.2851898339577, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.41297596188410657}

RECENT RESULT
hypothesis: A 72-unit delta-augmented GRU using 27 span-preserving frames will achieve at least 85% validation accuracy while reducing total inference MACs below 545,509,655 to approximately 533,283,840.
change: Replace the current 88-unit absolute-frame GRU with a 72-unit GRU receiving normalized frames and causal spectral deltas, resize its state and dual readout, and use 27 frames spanning the latest 28-frame window.
mechanism: Single-unit refinement of delta-augmented recurrence
evidence_used: The qualified 73-unit delta GRU achieved 86.87% accuracy and 0.413 cross-entropy at 545,509,655 MACs over the same 27-frame schedule, leaving 1.87 percentage points of margin for an adjacent one-unit width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2781689656642926069, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 25824, "peak_hidden_elements": 84480, "recurrent_macs": 532344960, "recurrent_steps": 22005, "total_inference_macs": 533283840, "training_seconds": 40.71468799980357, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.41659463519699}

RECENT RESULT
hypothesis: A 71-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 533,283,840 to approximately 521,190,055.
change: Reduce the GRU hidden width from 72 to 71 and resize its recurrent state and dual-timescale classifier accordingly.
mechanism: Adjacent-unit recurrent width refinement
evidence_used: The 72-unit model qualified at 85.89% accuracy, while the 73-unit model reached 86.87% on the identical feature representation and schedule; this remaining margin supports testing the adjacent lower-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2718606633880506113, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 25253, "peak_hidden_elements": 83456, "recurrent_macs": 520264215, "recurrent_steps": 22005, "total_inference_macs": 521190055, "training_seconds": 42.96981199993752, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4102224244661858}

RECENT RESULT
hypothesis: A 70-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 521,190,055 to approximately 509,228,300.
change: Reduce the GRU hidden width from 73 to 70 and resize its recurrent state and dual-timescale classifier accordingly.
mechanism: Adjacent-unit recurrent width refinement
evidence_used: The 71-unit model achieved 86.13% validation accuracy and 0.4102 cross-entropy at 521,190,055 MACs, while widths 72 and 73 also qualified on the identical representation and schedule; this supports testing the adjacent lower-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2656212299687902993, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24688, "peak_hidden_elements": 82432, "recurrent_macs": 508315500, "recurrent_steps": 22005, "total_inference_macs": 509228300, "training_seconds": 56.25552050000988, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4362127292375623}

RECENT RESULT
hypothesis: A 69-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 509,228,300 to approximately 497,398,575.
change: Reduce the GRU hidden width from 74 to 69 and resize its recurrent state and dual-timescale classifier accordingly.
mechanism: Adjacent-unit recurrent width refinement
evidence_used: The 70-unit model qualified at 85.28% accuracy and 509,228,300 MACs on the identical representation and schedule, establishing 69 units as the next informative structural boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2594506654065116709, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24129, "peak_hidden_elements": 81408, "recurrent_macs": 496498815, "recurrent_steps": 22005, "total_inference_macs": 497398575, "training_seconds": 49.62836266704835, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.44202573811349694}

RECENT RESULT
hypothesis: An 84-channel gated state whose recurrent control is mediated by a shared 28-dimensional nonlinear bottleneck will achieve at least 85% validation accuracy on the qualified 27-frame schedule while reducing predicted total inference MACs from 497,398,575 to approximately 429,928,800.
change: Replace the full-rank 75-unit GRU with a wider custom gated state update using counted `nn.Linear` projections and a compact shared recurrent controller; retain causal spectral deltas and mean-plus-final prediction, and adopt the qualified 27-frame span-preserving schedule.
mechanism: Shared bottleneck-controlled gated recurrence
evidence_used: The full-rank 69-unit GRU qualified at 85.77% with 497,398,575 MACs, and widths 69–73 all qualified on the same 27-frame representation, while 26-frame thinning failed. This suggests retaining 27 observations but challenges the load-bearing assumption that every gate requires an independent full-rank hidden-to-hidden matrix.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2242574041396680189, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 21384, "peak_hidden_elements": 96768, "recurrent_macs": 428833440, "recurrent_steps": 22005, "total_inference_macs": 429928800, "training_seconds": 94.3087120000273, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4861218317886072}

RECENT RESULT
hypothesis: A 68-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing predicted total inference MACs from 497,398,575 to approximately 485,700,880.
change: Reduce the GRU hidden width from 69 to 68 and resize its recurrent state and dual-timescale classifier accordingly.
mechanism: Adjacent-unit recurrent width refinement
evidence_used: The 69-unit model qualified at 85.77% accuracy and 497,398,575 MACs on the same representation and schedule, while widths 70–73 also qualified; 68 units is therefore the next informative structural boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2533489697012147261, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 23576, "peak_hidden_elements": 80384, "recurrent_macs": 484814160, "recurrent_steps": 22005, "total_inference_macs": 485700880, "training_seconds": 51.98854125011712, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4435367701243769}



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
    """A compact causal GRU supplied with explicit spectral motion."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(140, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        output, hidden = self.gru(
            features.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            count + 1.0,
            normalized,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
            normalized[:, -1, :],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, count, previous = state
        del previous
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
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
    """A compact causal GRU supplied with explicit spectral motion."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 71, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(142, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 71, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 71, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        output, hidden = self.gru(
            features.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            count + 1.0,
            normalized,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
            normalized[:, -1, :],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, count, previous = state
        del previous
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
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
    """A compact causal GRU supplied with explicit spectral motion."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(150, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        output, hidden = self.gru(
            features.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            count + 1.0,
            normalized,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
            normalized[:, -1, :],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, count, previous = state
        del previous
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 28, 0)
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
    """A compact causal GRU supplied with explicit spectral motion."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(138, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        output, hidden = self.gru(
            features.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            count + 1.0,
            normalized,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
            normalized[:, -1, :],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, count, previous = state
        del previous
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
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
