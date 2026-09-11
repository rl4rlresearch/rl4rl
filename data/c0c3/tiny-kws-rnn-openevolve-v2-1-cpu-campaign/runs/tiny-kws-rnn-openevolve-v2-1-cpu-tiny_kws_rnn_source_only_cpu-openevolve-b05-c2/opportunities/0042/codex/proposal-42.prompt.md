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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1585650168745642633, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 302319360, "recurrent_steps": 18745, "total_inference_macs": 303988480, "training_seconds": 38.58691979106516, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41982194397347106}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 23 causal frames while reducing total inference MACs from 317,132,800 to approximately 303,988,480.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1654212942000018968, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 315463680, "recurrent_steps": 19560, "total_inference_macs": 317132800, "training_seconds": 46.19284929195419, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41971319467743484}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 24 causal frames while reducing total inference MACs from 330,277,120 to approximately 317,132,800.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1722775715254395303, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 328608000, "recurrent_steps": 20375, "total_inference_macs": 330277120, "training_seconds": 44.923914291895926, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4193570376905196}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 25 causal frames while reducing total inference MACs from 343,421,440 to approximately 330,277,120.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1517087395491266298, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 289175040, "recurrent_steps": 17930, "total_inference_macs": 290844160, "training_seconds": 39.453580874949694, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223669695707918}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 22 causal frames while reducing total inference MACs from 303,988,480 to approximately 290,844,160.

## Recent verification evidence

RECENT RESULT
hypothesis: A 64-unit full-resolution GRU using mean, final, maximum, and minimum summaries will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.69% versus the qualified 65-unit design.
change: Reduce the recurrent width from 67 to 64, add temporal-minimum tracking, and resize the classifier to consume four 64-unit summaries while preserving all 32 causal steps and the established training procedure.
mechanism: Adjacent-width GRU pruning with signed temporal-extrema readout
evidence_used: The 65-unit signed-extrema design achieved 85.77% accuracy at 433,971,200 MACs, whereas the 65-unit triple-readout design failed; this isolates minimum pooling as useful at the width boundary and motivates testing the adjacent 64-unit signed-extrema model.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2202715128035029648, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 420618240, "recurrent_steps": 26080, "total_inference_macs": 422287360, "training_seconds": 46.185714167077094, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.40811364109530773}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy when one central frame is omitted, while reducing total inference MACs from 422,287,360 to approximately 409,143,040 and recurrent steps from 32 to 31 per example.
change: Replace the 70-unit triple-readout GRU with the qualified 64-unit mean/final/maximum/minimum design and omit one interior frame while retaining both causal endpoints.
mechanism: Single-frame temporal pruning with signed-extrema readout
evidence_used: The 64-unit signed-extrema model achieved 86.63% accuracy at 422,287,360 MACs, providing a 1.63-point margin; the 24-step experiment indicates temporal pruning is computationally valuable but too aggressive, motivating the smallest possible one-frame schedule reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2134152354780653313, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 407473920, "recurrent_steps": 25265, "total_inference_macs": 409143040, "training_seconds": 39.63655120902695, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.40820536233164784}

RECENT RESULT
hypothesis: A 64-unit signed-extrema GRU using 30 causal frames will retain at least 85% validation accuracy while reducing total inference MACs from 409,143,040 to approximately 395,998,720.
change: Replace the 71-unit triple-readout model with the qualified 64-unit mean/final/maximum/minimum model and omit frames at one-quarter and one-half of the sequence, preserving both endpoints.
mechanism: Quarter-and-center temporal pruning with signed-extrema readout
evidence_used: The 64-unit signed-extrema model achieved 86.87% accuracy after one central frame was omitted, exceeding the threshold by 1.87 points; adding one separated interior omission is the smallest next temporal-compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2065589581526276978, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 394329600, "recurrent_steps": 24450, "total_inference_macs": 395998720, "training_seconds": 36.32589954207651, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4145544227646904}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 29 causal frames while reducing total inference MACs from 395,998,720 to approximately 382,854,400.
change: Add a third, evenly separated interior omission at three-quarters of the sequence, preserving both endpoints and reducing execution from 30 to 29 recurrent steps for 32-frame inputs.
mechanism: Three-point temporal pruning with signed-extrema readout
evidence_used: The current 30-step model achieved 86.63% accuracy, 1.63 points above threshold, after the preceding 31-step model achieved 86.87%; this supports testing the smallest next temporal reduction with omissions distributed across the sequence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1997026808271900643, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 381185280, "recurrent_steps": 23635, "total_inference_macs": 382854400, "training_seconds": 32.3983185000252, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.41096569599549465}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 28 causal frames while reducing total inference MACs from 382,854,400 to approximately 369,710,080.
change: Preserve the proven quarter, center, and three-quarter omissions and add one omission at one-eighth of the sequence, reducing 32-frame execution from 29 to 28 recurrent steps.
mechanism: Incremental early-frame temporal pruning with signed-extrema readout
evidence_used: The 29-step design achieved 86.63% accuracy, matching the 30-step result and retaining a 1.63-point margin; this supports the smallest next temporal reduction while preserving the three already-qualified omissions.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1928464035017524308, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 368040960, "recurrent_steps": 22820, "total_inference_macs": 369710080, "training_seconds": 37.94612995814532, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4123815056736484}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 27 causal frames while reducing total inference MACs from 369,710,080 to approximately 356,565,760.
change: Preserve the four proven omissions and additionally omit the frame at seven-eighths of the sequence, reducing 32-frame execution from 28 to 27 recurrent steps while retaining both endpoints.
mechanism: Symmetric late-frame temporal pruning
evidence_used: The 28-step design achieved 86.87% accuracy, 1.87 points above threshold, after every incremental reduction from 32 through 28 steps qualified; adding the late counterpart to the proven one-eighth omission is the smallest next compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1859901261763147973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 354896640, "recurrent_steps": 22005, "total_inference_macs": 356565760, "training_seconds": 28.181677292101085, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4157459376048457}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 26 causal frames while reducing total inference MACs from 356,565,760 to approximately 343,421,440.
change: Replace the 65-unit full-resolution model with the qualified 64-unit signed-extrema architecture and omit six distributed interior frames, adding a three-eighths omission to the proven 27-step schedule.
mechanism: Incremental mid-sequence temporal pruning with signed-extrema readout
evidence_used: The 27-step 64-unit design achieved 86.38% accuracy, 1.38 points above threshold, after every incremental reduction from 32 through 27 steps qualified; removing one additional interior frame is the smallest next structural compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1791338488508771638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 341752320, "recurrent_steps": 21190, "total_inference_macs": 343421440, "training_seconds": 46.085821042070165, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.41763563097620304}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 25 causal frames while reducing total inference MACs from 343,421,440 to approximately 330,277,120.
change: Add the missing five-eighths-frame omission, producing evenly spaced omissions at every fourth interior frame while retaining both endpoints.
mechanism: Uniform one-in-four temporal pruning with signed-extrema readout
evidence_used: The 64-unit design qualified at every incremental reduction from 32 through 26 steps; the current 26-step result achieved 85.77% accuracy, so the adjacent 25-step schedule is the most informative next temporal-compute boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1722775715254395303, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 328608000, "recurrent_steps": 20375, "total_inference_macs": 330277120, "training_seconds": 44.923914291895926, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4193570376905196}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 24 causal frames while reducing total inference MACs from 330,277,120 to approximately 317,132,800.
change: Restore the proven seven uniformly spaced omissions and additionally omit the frame at fifteen-sixteenths of the sequence, retaining both endpoints and reducing execution to 24 steps.
mechanism: Incremental late-edge temporal pruning
evidence_used: The 25-step design achieved 86.13% accuracy, and every incremental reduction from 32 through 25 steps qualified; its 1.13-point margin motivates testing the adjacent step boundary with a low-information edge-adjacent omission.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1654212942000018968, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 315463680, "recurrent_steps": 19560, "total_inference_macs": 317132800, "training_seconds": 46.19284929195419, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41971319467743484}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 23 causal frames while reducing total inference MACs from 317,132,800 to approximately 303,988,480.
change: Apply the qualified 24-step schedule and additionally omit the frame at one-sixteenth of the sequence, symmetrically complementing the proven fifteen-sixteenths omission while retaining both endpoints.
mechanism: Symmetric early-edge temporal pruning
evidence_used: The 24-step design achieved 86.26% accuracy with a 1.26-point margin after every adjacent reduction from 32 through 24 steps qualified; removing one edge-adjacent frame is the smallest and most informative next compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1585650168745642633, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 302319360, "recurrent_steps": 18745, "total_inference_macs": 303988480, "training_seconds": 38.58691979106516, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41982194397347106}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 22 causal frames while reducing total inference MACs from 303,988,480 to approximately 290,844,160.
change: Apply the qualified 23-step schedule and additionally omit the earliest interior frame, preserving both endpoints and reducing 32-frame execution to 22 recurrent steps.
mechanism: Incremental boundary-frame temporal pruning
evidence_used: The 23-step design achieved 86.26% accuracy with a 1.26-point margin after every adjacent reduction from 32 through 23 steps qualified; pruning one likely low-information boundary frame is the smallest next structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1517087395491266298, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 289175040, "recurrent_steps": 17930, "total_inference_macs": 290844160, "training_seconds": 39.453580874949694, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223669695707918}

RECENT RESULT
hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 21 causal frames while reducing total inference MACs from 290,844,160 to approximately 277,699,840.
change: Omit frame 29, the latest remaining near-boundary interior frame, while preserving both causal endpoints and reducing execution from 22 to 21 recurrent steps.
mechanism: Symmetric boundary-frame temporal pruning
evidence_used: The 22-step design achieved 86.01% accuracy after every adjacent reduction from 32 through 22 steps qualified; complementing the omitted early frames 1 and 2 with late frame 29 is the smallest next structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448524622236889963, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 276030720, "recurrent_steps": 17115, "total_inference_macs": 277699840, "training_seconds": 38.7681831670925, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4345594441232506}



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
    """A causal GRU with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
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
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
        return schedule


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
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
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
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            if available_frames > 16:
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
        return schedule


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
    """A causal GRU with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
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
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            schedule = [index for index in schedule if index not in skipped]
        return schedule


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
    """A causal GRU with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
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
        schedule = list(range(available_frames))
        if available_frames > 8:
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
        return schedule


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
