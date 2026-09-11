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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2385719236389813905, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19545, "peak_hidden_elements": 103424, "recurrent_macs": 456060960, "recurrent_steps": 26080, "total_inference_macs": 457371480, "training_seconds": 56.75140625005588, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.42030147248250577}
prior_hypothesis: A 67-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the qualified 68-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2263659815299808373, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19093, "peak_hidden_elements": 133632, "recurrent_macs": 432276000, "recurrent_steps": 26080, "total_inference_macs": 433971200, "training_seconds": 46.8842969161924, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4041227001354007}
prior_hypothesis: A 65-unit GRU augmented with temporal-minimum pooling will reach at least 85% validation accuracy while using approximately 433,971,200 total inference MACs, 2.56% fewer than the qualified 66-unit design.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2644053971023329041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21561, "peak_hidden_elements": 109568, "recurrent_macs": 505508640, "recurrent_steps": 26080, "total_inference_macs": 506897400, "training_seconds": 77.82525529200211, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.41365568008890913}
prior_hypothesis: A 71-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.46% versus the qualified 72-unit design.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2578245952129720328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21048, "peak_hidden_elements": 108032, "recurrent_macs": 492912000, "recurrent_steps": 26080, "total_inference_macs": 494281200, "training_seconds": 76.46355379093438, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40918661480301}
prior_hypothesis: A 70-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.49% versus the qualified 71-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: A 74-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.37% versus the qualified 75-unit design.
change: Reduce the GRU and recurrent-summary width from 75 to 74 and resize the triple-readout classifier from 225 to 222 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 75-unit design achieved 86.75% validation accuracy, while every tested triple-readout width from 75 through 80 qualified; its 1.75-point margin motivates the next one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2846375368645074896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23136, "peak_hidden_elements": 114176, "recurrent_macs": 544237440, "recurrent_steps": 26080, "total_inference_macs": 545684880, "training_seconds": 80.2417831250932, "validation_accuracy": 0.8785276073619632, "validation_cross_entropy": 0.3959634605360909}

RECENT RESULT
hypothesis: A 73-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.4% versus the qualified 74-unit design.
change: Reduce the GRU and recurrent-summary width from 76 to 73 units and resize the triple-readout classifier from 228 to 219 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 74-unit design achieved 87.85% validation accuracy—the strongest result among the tested 74–80-unit triple-readout models—with a 2.85-point margin above the requirement, motivating a one-unit structural reduction to probe the next compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2778118679281006325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22605, "peak_hidden_elements": 112640, "recurrent_macs": 531171360, "recurrent_steps": 26080, "total_inference_macs": 532599240, "training_seconds": 113.48929166584276, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4272858789362059}

RECENT RESULT
hypothesis: A 72-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.43% versus the qualified 73-unit design.
change: Reduce the GRU and recurrent-summary width from 77 to 72 units and resize the triple-readout classifier from 231 to 216 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Boundary-width GRU pruning with triple temporal readout
evidence_used: The 73-unit triple-readout model qualified at 85.89% accuracy and 532,599,240 total MACs; testing the adjacent 72-unit width directly probes the next compute boundary without changing the successful temporal readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 61.63849383289926, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.425748027029213}

RECENT RESULT
hypothesis: A 71-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.46% versus the qualified 72-unit design.
change: Reduce the GRU and recurrent-summary width from 72 to 71 units and resize the triple-readout classifier from 216 to 213 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The current 72-unit design achieved 86.38% validation accuracy at 519,670,080 total MACs, leaving a 1.38-point margin above the requirement; every tested triple-readout width from 72 through 80 qualified, motivating the adjacent structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2644053971023329041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21561, "peak_hidden_elements": 109568, "recurrent_macs": 505508640, "recurrent_steps": 26080, "total_inference_macs": 506897400, "training_seconds": 77.82525529200211, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.41365568008890913}

RECENT RESULT
hypothesis: A 70-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.49% versus the qualified 71-unit design.
change: Reduce the GRU and recurrent-summary width from 73 to 70 units and resize the triple-readout classifier from 219 to 210 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 71-unit triple-readout model achieved 87.24% validation accuracy at 506,897,400 total MACs, leaving a 2.24-point margin after every tested width from 71 through 80 qualified; this motivates probing the adjacent lower width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2578245952129720328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21048, "peak_hidden_elements": 108032, "recurrent_macs": 492912000, "recurrent_steps": 26080, "total_inference_macs": 494281200, "training_seconds": 76.46355379093438, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.40918661480301}

RECENT RESULT
hypothesis: A 69-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.52% versus the qualified 70-unit design.
change: Reduce the GRU and recurrent-summary width from 74 to 69 units and resize the triple-readout classifier from 222 to 207 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 70-unit triple-readout model achieved 86.38% validation accuracy at 494,281,200 total MACs, leaving a 1.38-point margin; every tested width from 70 through 80 qualified, so the adjacent 69-unit width is the most direct next compute-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2513254156726264901, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20541, "peak_hidden_elements": 106496, "recurrent_macs": 480471840, "recurrent_steps": 26080, "total_inference_macs": 481821480, "training_seconds": 60.18430820805952, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.433567388803681}

RECENT RESULT
hypothesis: A 68-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.55% versus the qualified 69-unit design.
change: Replace the current 128-unit mean-only GRU with a 68-unit GRU whose state tracks mean and temporal-maximum summaries and whose classifier consumes mean, final, and maximum outputs across all 32 causal frames.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 69-unit triple-readout model qualified at 85.40% accuracy and 481,821,480 total MACs, while every tested width from 69 through 80 qualified; testing the adjacent 68-unit width directly locates the next structural compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2449078584812962760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20040, "peak_hidden_elements": 104960, "recurrent_macs": 468188160, "recurrent_steps": 26080, "total_inference_macs": 469518240, "training_seconds": 66.43737191706896, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4124882048624425}

RECENT RESULT
hypothesis: A 67-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the qualified 68-unit design.
change: Reduce the GRU and recurrent-summary width from 68 to 67 units and resize the triple-readout classifier from 204 to 201 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 68-unit design achieved 86.38% validation accuracy at 469,518,240 total MACs, and every tested triple-readout width from 68 through 80 qualified; its 1.38-point margin motivates probing the adjacent structural compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2385719236389813905, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19545, "peak_hidden_elements": 103424, "recurrent_macs": 456060960, "recurrent_steps": 26080, "total_inference_macs": 457371480, "training_seconds": 56.75140625005588, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.42030147248250577}

RECENT RESULT
hypothesis: A 66-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.62% versus the qualified 67-unit design.
change: Reduce the GRU and recurrent-summary width from 69 to 66 units and resize the triple-readout classifier from 207 to 198 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 67-unit design achieved 86.63% validation accuracy at 457,371,480 total MACs, while every tested triple-readout width from 67 through 80 qualified; its 1.63-point margin motivates testing the adjacent structural compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2323176111456818336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19056, "peak_hidden_elements": 101888, "recurrent_macs": 444090240, "recurrent_steps": 26080, "total_inference_macs": 445381200, "training_seconds": 63.44164304086007, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.433227473533958}

RECENT RESULT
hypothesis: A 65-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.66% versus the qualified 66-unit design.
change: Reduce the GRU and recurrent-summary width from 70 to 65 units and resize the triple-readout classifier from 210 to 195 inputs, preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with triple temporal readout
evidence_used: The 66-unit design qualified at 85.28% accuracy and 445,381,200 total MACs, while every tested triple-readout width from 66 through 80 qualified; testing the adjacent 65-unit width directly identifies the next structural compute boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2261449210013976053, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 432276000, "recurrent_steps": 26080, "total_inference_macs": 433547400, "training_seconds": 62.24201879207976, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4447986064513037}

RECENT RESULT
hypothesis: A 74-unit GRU evaluated on 24 uniformly distributed causal frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 19% versus the current 71-unit, 32-step design.
change: Increase the GRU and triple-readout width from 71 to 74 units, resize the classifier from 213 to 222 inputs, and skip one frame in every four while retaining both temporal endpoints.
mechanism: Three-of-four causal frame scheduling with compensatory hidden width
evidence_used: The full-resolution 74-unit design achieved 87.85% accuracy, the strongest observed result and 2.85 points above threshold; its margin motivates exchanging modest width for a 25% recurrent-step reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2136669043304791416, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 23136, "peak_hidden_elements": 114176, "recurrent_macs": 408178080, "recurrent_steps": 19560, "total_inference_macs": 409625520, "training_seconds": 47.40071808407083, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4754275269303585}

RECENT RESULT
hypothesis: A 65-unit GRU augmented with temporal-minimum pooling will reach at least 85% validation accuracy while using approximately 433,971,200 total inference MACs, 2.56% fewer than the qualified 66-unit design.
change: Reduce the recurrent width to 65 and expand the classifier from mean/final/maximum to mean/final/maximum/minimum recurrent summaries.
mechanism: Signed temporal-extrema readout at the failed width boundary
evidence_used: The 65-unit triple-readout model missed qualification by only 0.092 percentage points at 433,547,400 MACs; adding the complementary negative-extrema summary costs only 423,800 classifier MACs while preserving the recurrent compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2263659815299808373, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 19093, "peak_hidden_elements": 133632, "recurrent_macs": 432276000, "recurrent_steps": 26080, "total_inference_macs": 433971200, "training_seconds": 46.8842969161924, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4041227001354007}



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
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(201, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 67, device=device, dtype=dtype)
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
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(
            torch.cat((mean_output, final_output, maximum), dim=-1)
        )

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
    """A causal GRU with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(260, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, minimum, count

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
        hidden, summary, maximum, minimum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            torch.minimum(minimum, output),
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
        hidden, summary, maximum, minimum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            torch.minimum(minimum, outputs.amin(dim=1)),
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
        hidden, summary, maximum, minimum, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )

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
        self.gru = nn.GRU(20, 71, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(213, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 71, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 71, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 71, device=device, dtype=dtype)
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
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(
            torch.cat((mean_output, final_output, maximum), dim=-1)
        )

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
        self.gru = nn.GRU(20, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 70, device=device, dtype=dtype)
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
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(
            torch.cat((mean_output, final_output, maximum), dim=-1)
        )

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
