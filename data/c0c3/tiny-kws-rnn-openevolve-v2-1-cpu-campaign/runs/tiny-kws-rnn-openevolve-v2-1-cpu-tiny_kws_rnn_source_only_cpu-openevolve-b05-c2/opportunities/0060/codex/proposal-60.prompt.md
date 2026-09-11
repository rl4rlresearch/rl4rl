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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448524622236889963, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 276030720, "recurrent_steps": 17115, "total_inference_macs": 277699840, "training_seconds": 34.16089425003156, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4290467431940184}
prior_hypothesis: Omitting frame 6 instead of accuracy-critical frame 29 will retain at least 85% validation accuracy while executing 21 recurrent steps and approximately 277,699,840 total inference MACs.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1409018555080668489, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18129, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 270126045, "training_seconds": 34.332134458003566, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45630040081000767}
prior_hypothesis: A 63-unit signed-extrema GRU on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 277,699,840 to approximately 270,126,045.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1517087395491266298, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 289175040, "recurrent_steps": 17930, "total_inference_macs": 290844160, "training_seconds": 39.453580874949694, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223669695707918}
prior_hypothesis: The qualified 64-unit signed-extrema GRU will retain at least 85% validation accuracy on 22 causal frames while reducing total inference MACs from 303,988,480 to approximately 290,844,160.

## Recent verification evidence

RECENT RESULT
hypothesis: A 63-unit signed-extrema GRU on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 277,699,840 to approximately 270,126,045.
change: Restore the qualified 21-step schedule and structurally reduce the GRU width from 64 to 63 units, including its state summaries and classifier input.
mechanism: One-unit recurrent-width contraction at the qualified temporal frontier
evidence_used: The 64-unit 21-step design achieved 85.40% accuracy, while several 20-step variants failed; this identifies temporal execution as the current boundary and motivates the smaller, orthogonal 1/64 recurrent-width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1409018555080668489, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18129, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 270126045, "training_seconds": 34.332134458003566, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45630040081000767}

RECENT RESULT
hypothesis: A 62-unit signed-extrema GRU on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 262,654,940.
change: Reduce the GRU width from 63 to 62 units and resize its recurrent states and classifier input accordingly.
mechanism: One-unit recurrent-width contraction at the qualified temporal frontier
evidence_used: The 63-unit, 21-step model qualified at 85.03%, while several 20-step schedules failed; the adjacent width reduction is therefore the most informative orthogonal test of the structural-compute boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1370048134589860111, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 17656, "peak_hidden_elements": 127488, "recurrent_macs": 261037980, "recurrent_steps": 17115, "total_inference_macs": 262654940, "training_seconds": 33.75018354202621, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.46306695294526456}

RECENT RESULT
hypothesis: Projecting each 20-band frame to 16 learned features while retaining the qualified 64-unit GRU will maintain at least 85% validation accuracy and reduce total inference MACs below the qualified 63-unit design, to approximately 270,032,320.
change: Add a 20-to-16 linear input projection and change the GRU input width to 16, leaving the 64-unit recurrent state, summaries, schedule, and training procedure unchanged.
mechanism: Learned spectral bottleneck with recurrent-width preservation
evidence_used: The 63-unit GRU barely qualified at 85.03%, while 62 units failed at 84.17%; this indicates recurrent-state contraction has reached its accuracy boundary and motivates compressing the spectral input while restoring the better-qualified 64-unit state.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1408529671219378811, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18176, "peak_hidden_elements": 131584, "recurrent_macs": 268363200, "recurrent_steps": 17115, "total_inference_macs": 270032320, "training_seconds": 36.68211720907129, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.5009793240599837}

RECENT RESULT
hypothesis: Omitting frame 27 instead of frame 26 from the qualified 21-step schedule will preserve at least 85% accuracy while reducing execution to 20 recurrent steps and total inference MACs to approximately 264,555,520.
change: Restore the qualified frame-6 omission and additionally omit frame 27 for 32-frame inputs, retaining frame 26, accuracy-critical frame 29, and both endpoints.
mechanism: Late-cluster edge pruning
evidence_used: The frame-26 20-step design reached 84.91%, the closest failed temporal reduction; pruning adjacent cluster-edge frame 27 is the smallest informative schedule substitution while preserving the same compute reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1379961848982513628, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 262886400, "recurrent_steps": 16300, "total_inference_macs": 264555520, "training_seconds": 38.889631791971624, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.43291577298217027}

RECENT RESULT
hypothesis: The qualified 63-unit, 21-step GRU will retain at least 85% accuracy when its correlated mean and terminal summaries share classifier weights, reducing total inference MACs from 270,126,045 to approximately 269,715,285.
change: Restore the qualified 21-frame, 63-unit design and replace the four-way readout with a three-way readout over mean-plus-final, maximum, and minimum summaries.
mechanism: Fixed mean-terminal summary fusion
evidence_used: The 63-unit, 21-step model qualified at 85.03%, whereas 62 units and every tested 20-step schedule failed; preserving its recurrent path while structurally reducing only the redundant summary readout is the smallest orthogonal compute reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1406875968419015625, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 17625, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 269715285, "training_seconds": 41.92766183288768, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.46411665465934143}

RECENT RESULT
hypothesis: Fusing only one of 63 mean/final feature pairs will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 270,119,525.
change: Reduce the classifier input from 252 to 251 features by averaging one corresponding mean and terminal channel while preserving every other summary feature and the qualified 63-unit, 21-step recurrent path.
mechanism: Single-channel mean-terminal readout fusion
evidence_used: The 63-unit four-summary model qualified at 85.03%, while fusing all 63 mean/final pairs failed at 83.44%; contracting only one pair is the smallest structural test of whether the broad fusion was too aggressive.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1408984545768578761, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18121, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 270119525, "training_seconds": 37.291890625143424, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46148365231379407}

RECENT RESULT
hypothesis: A 112-unit tanh RNN using the qualified 22-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 267,998,080.
change: Replace the 64-unit GRU with a wider, single-affine 112-unit tanh RNN, resize its summaries and classifier, and restore frame 6 to use the stronger 22-frame schedule.
mechanism: Gate-to-width recurrent reallocation
evidence_used: The 22-step GRU achieved 86.01% accuracy, while the 63-unit 21-step GRU barely qualified at 85.03% and further width/readout contractions failed; reallocating gated-computation MACs into a wider recurrent state tests a structurally different efficiency frontier while recovering the stronger schedule.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1397918765928887450, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 18640, "peak_hidden_elements": 229888, "recurrent_macs": 265077120, "recurrent_steps": 17930, "total_inference_macs": 267998080, "training_seconds": 34.154743334045634, "validation_accuracy": 0.8269938650306748, "validation_cross_entropy": 0.544946026948332}

RECENT RESULT
hypothesis: Averaging the two highest mel bands into one feature before the qualified 63-unit, 21-step GRU will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 266,891,310.
change: Restore the qualified 63-unit and 21-frame design, then reduce its recurrent input width from 20 to 19 using fixed pooling of the final two normalized mel bands.
mechanism: Fixed edge-band pooling with qualified recurrent width
evidence_used: The 63-unit, 21-step GRU qualified at 85.03%, whereas the more aggressive learned 20-to-16 bottleneck failed at 83.68%; a one-feature fixed reduction preserves information from every mel band while testing the smallest spectral contraction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1392145685120155965, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 17940, "peak_hidden_elements": 129536, "recurrent_macs": 265248270, "recurrent_steps": 17115, "total_inference_macs": 266891310, "training_seconds": 32.92324254196137, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.44561610309624233}

RECENT RESULT
hypothesis: A 79-unit minimal gated recurrent unit on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 269,773,150.
change: Replace the three-affine 64-unit GRU with a two-affine 79-unit minimal gated recurrent update, preserve all four temporal summaries, and adopt the qualified 21-frame schedule.
mechanism: Single-gate recurrent width reallocation
evidence_used: The 63-unit GRU qualified at 85.03%, while the 112-unit ungated tanh RNN failed at 82.70%; this indicates gating remains important and motivates reallocating three-affine GRU computation into a wider single-gate state rather than removing gating entirely.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1407177801063812641, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 18376, "peak_hidden_elements": 162304, "recurrent_macs": 267712830, "recurrent_steps": 17115, "total_inference_macs": 269773150, "training_seconds": 68.74762225011364, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.4882515468480397}

RECENT RESULT
hypothesis: Reducing the batch size to 96 will provide enough additional optimizer updates for the single-channel mean/final fusion to recover at least 85% validation accuracy, while retaining 21 recurrent steps and reducing total inference MACs to approximately 270,119,525.
change: Fuse one corresponding mean/final summary channel, reduce the classifier input from 252 to 251 features, and lower training batch size from 128 to 96 without changing the qualified recurrent path or inference schedule.
mechanism: Increased update density for minimally compressed readout
evidence_used: The full 63-unit model qualified at 85.03%, while the otherwise identical one-channel fusion reached 84.79% and was only two validation predictions short; this was a much smaller accuracy deficit than the 62-unit or spectral contractions, motivating a training-only recovery attempt for the smallest known structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1408984545768578761, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 18121, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 270119525, "training_seconds": 41.40846679196693, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4255784953298744}

RECENT RESULT
hypothesis: The qualified 63-unit, 21-step GRU will retain at least 85% validation accuracy when one redundant softmax logit is fixed at zero, reducing total inference MACs from 270,126,045 to approximately 269,920,665.
change: Restore the qualified 63-unit recurrent path and replace the eight-output classifier with seven learned logits plus one fixed reference logit.
mechanism: Reference-class softmax parameterization
evidence_used: The 63-unit, 21-step model qualified at 85.03%; unlike the failed summary fusion, fixing one softmax logit removes only the common-offset redundancy and preserves the full space of eight-class probability distributions.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1407947261749842056, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 17876, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 269920665, "training_seconds": 35.7661010408774, "validation_accuracy": 0.8269938650306748, "validation_cross_entropy": 0.4789292317957966}

RECENT RESULT
hypothesis: A symmetric seven-contrast classifier on the qualified 63-unit, 21-step recurrent path will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 269,920,665.
change: Restore the qualified 63-unit GRU and 21-frame schedule, then replace eight directly learned logits with seven learned orthonormal contrasts expanded into eight centered logits using fixed arithmetic.
mechanism: Orthonormal centered-logit parameterization
evidence_used: The 63-unit, 21-step model qualified at 85.03%. Fixing one class logit at zero reduced the target MACs but failed at 82.70%; an orthonormal centered basis preserves the same seven-dimensional softmax space without privileging a reference class or introducing its asymmetric conditioning.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1407947261749842056, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 21, "parameters": 17876, "peak_hidden_elements": 129536, "recurrent_macs": 268483005, "recurrent_steps": 17115, "total_inference_macs": 269920665, "training_seconds": 40.27297379216179, "validation_accuracy": 0.8257668711656442, "validation_cross_entropy": 0.47149989590323044}



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
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
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
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 63, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
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
                skipped.add(3 * available_frames // 16)
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
