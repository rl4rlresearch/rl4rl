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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2594506654065116709, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24129, "peak_hidden_elements": 81408, "recurrent_macs": 496498815, "recurrent_steps": 22005, "total_inference_macs": 497398575, "training_seconds": 49.62836266704835, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.44202573811349694}
prior_hypothesis: A 69-unit delta-augmented GRU using the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 509,228,300 to approximately 497,398,575.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2391067200308421133, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 22378, "peak_hidden_elements": 82432, "recurrent_macs": 457483950, "recurrent_steps": 22005, "total_inference_macs": 458396750, "training_seconds": 55.99784837500192, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45177418527427626}
prior_hypothesis: A 70-unit GRU preserving the qualified nine-delta representation and 27-frame schedule will reach at least 85% validation accuracy while reducing total inference MACs from 469,632,340 to approximately 458,396,750.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3085069725220775893, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27573, "peak_hidden_elements": 87552, "recurrent_macs": 590467500, "recurrent_steps": 22820, "total_inference_macs": 591445500, "training_seconds": 46.008248457917944, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4075592041015625}
prior_hypothesis: A 75-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will retain at least 85% accuracy while reducing total inference MACs from 604,534,400 to approximately 591,445,500.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2474122191595529333, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 23123, "peak_hidden_elements": 83456, "recurrent_macs": 473393565, "recurrent_steps": 22005, "total_inference_macs": 474319405, "training_seconds": 42.024969290941954, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4445389636455138}
prior_hypothesis: A 71-unit GRU receiving all 20 normalized bands plus 10 adjacent-band pooled deltas will retain at least 85% validation accuracy while reducing total inference MACs to approximately 474,319,405.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 71-unit GRU receiving all 20 normalized bands plus 10 adjacent-band pooled deltas will retain at least 85% validation accuracy while reducing total inference MACs to approximately 474,319,405.
change: Increase the recurrent state to the qualified 71-unit width, compress only the spectrally redundant delta features from 20 to 10 channels, and resize the classifier and state accordingly.
mechanism: Adjacent-band delta pooling with preserved recurrent width
evidence_used: The full-delta 71-unit model achieved 86.13% accuracy, while shrinking the full-input recurrent state to 68 units failed at 84.17%; preserving recurrent capacity while structurally compressing locally correlated spectral motion tests a more favorable allocation of MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2474122191595529333, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 23123, "peak_hidden_elements": 83456, "recurrent_macs": 473393565, "recurrent_steps": 22005, "total_inference_macs": 474319405, "training_seconds": 42.024969290941954, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4445389636455138}

RECENT RESULT
hypothesis: A 71-unit GRU using eight adjacent-band delta pairs plus one pooled upper-band delta will retain at least 85% validation accuracy while reducing total inference MACs from 474,319,405 to approximately 469,632,340.
change: Preserve the qualified 71-unit recurrent state and 27-frame schedule, but reduce delta features from 10 to 9 by merging the two highest-frequency pooled deltas.
mechanism: Selective upper-band delta coarsening
evidence_used: The 71-unit model with 10 pooled deltas achieved 85.52% accuracy at 474,319,405 MACs, while reducing recurrent width to 68 failed; this motivates preserving recurrent capacity and making a conservative reduction only in the spectrally redundant delta representation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2449673747367031655, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 22910, "peak_hidden_elements": 83456, "recurrent_macs": 468706500, "recurrent_steps": 22005, "total_inference_macs": 469632340, "training_seconds": 48.70165237504989, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4457198628618673}

RECENT RESULT
hypothesis: A 71-unit GRU using seven adjacent-band delta pairs plus one pooled upper-band delta over the qualified 27-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 469,632,340 to approximately 464,945,275.
change: Preserve the qualified recurrent width and temporal coverage while reducing delta features from nine to eight by pooling the three highest-frequency band pairs.
mechanism: Selective high-band delta coarsening
evidence_used: The 71-unit model remained qualified at 85.77% accuracy after reducing pooled deltas from ten to nine, whereas reducing recurrent width to 68 failed; this favors another conservative feature reduction while preserving recurrent capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2425225303138533977, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 22697, "peak_hidden_elements": 83456, "recurrent_macs": 464019435, "recurrent_steps": 22005, "total_inference_macs": 464945275, "training_seconds": 50.006185916019604, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46956296581432133}

RECENT RESULT
hypothesis: A 70-unit GRU preserving the qualified nine-delta representation and 27-frame schedule will reach at least 85% validation accuracy while reducing total inference MACs from 469,632,340 to approximately 458,396,750.
change: Reduce the GRU hidden width from 71 to 70 and resize its recurrent state and dual-timescale classifier.
mechanism: Nine-delta recurrent width refinement
evidence_used: The current 71-unit, nine-delta model qualified at 85.77%, and a 70-unit full-delta model qualified at 85.28%; meanwhile, reducing the nine-delta representation to eight channels failed, motivating preservation of the qualified features while testing the adjacent width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2391067200308421133, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 22378, "peak_hidden_elements": 82432, "recurrent_macs": 457483950, "recurrent_steps": 22005, "total_inference_macs": 458396750, "training_seconds": 55.99784837500192, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45177418527427626}

RECENT RESULT
hypothesis: A 69-unit GRU preserving the qualified nine-delta representation and 27-frame schedule will achieve at least 85% validation accuracy while reducing total inference MACs from 458,396,750 to approximately 447,293,190.
change: Reduce recurrent width from 71 to 69, resize the state and classifier, and compress the ten paired deltas into eight individual lower-band deltas plus one pooled upper-band delta.
mechanism: Adjacent-width refinement with selective upper-band delta pooling
evidence_used: The 70-unit nine-delta model qualified at 85.40% and 458,396,750 MACs, while a 69-unit full-delta model qualified at 85.77%; together they motivate testing the adjacent 69-unit boundary with the already-qualified nine-delta representation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2333149341819627447, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 21852, "peak_hidden_elements": 81408, "recurrent_macs": 446393430, "recurrent_steps": 22005, "total_inference_macs": 447293190, "training_seconds": 54.60221683396958, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4779424655656873}



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
        self.gru = nn.GRU(29, 70, num_layers=1, batch_first=True)
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
        paired_delta = 0.5 * (delta[:, 0::2] + delta[:, 1::2])
        pooled_delta = torch.cat(
            (
                paired_delta[:, :8],
                paired_delta[:, 8:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, pooled_delta), dim=1)
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
        paired_deltas = 0.5 * (
            deltas[:, :, 0::2] + deltas[:, :, 1::2]
        )
        pooled_deltas = torch.cat(
            (
                paired_deltas[:, :, :8],
                paired_deltas[:, :, 8:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        features = torch.cat((normalized, pooled_deltas), dim=2)
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
        self.gru = nn.GRU(30, 71, num_layers=1, batch_first=True)
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
        pooled_delta = 0.5 * (delta[:, 0::2] + delta[:, 1::2])
        features = torch.cat((normalized, pooled_delta), dim=1)
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
        pooled_deltas = 0.5 * (deltas[:, :, 0::2] + deltas[:, :, 1::2])
        features = torch.cat((normalized, pooled_deltas), dim=2)
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
