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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1764539150337571809, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20584, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338283680, "training_seconds": 35.3120228338521, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.438861083984375}
prior_hypothesis: A 68-unit GRU using the qualifying 24-frame schedule with only the earliest boundary frame removed will reach at least 85% accuracy in 23 steps and use fewer MACs than the verified 67-unit, 24-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 68-unit GRU processing 23 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.6% versus the verified 67-unit, 24-step model.
change: Increase recurrent and summary width from 67 to 68 units, resize the four-statistic classifier input from 268 to 272 features, and reduce the uniform schedule from 24 to 23 steps.
mechanism: Deviation-assisted width–time rebalancing
evidence_used: Successive deviation-readout exchanges qualified at 65 units/26 steps, 66/25, and 67/24; the latest reached 85.03%, motivating one more incremental exchange while preserving a net MAC reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1764539150337571809, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20584, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338283680, "training_seconds": 37.355420750100166, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4483058250754889}

RECENT RESULT
hypothesis: Adding a zero-MAC temporal trend statistic to the 68-unit, 23-step GRU will raise validation accuracy from 83.93% to at least 85% while keeping total dense inference MACs below the verified 67-unit, 24-step model.
change: Use 68 hidden units over 23 uniformly spaced frames and augment the classifier with a fifth summary encoding the normalized linear temporal trend of each recurrent unit.
mechanism: Temporal-trend-augmented recurrent readout
evidence_used: The 68-unit, 23-step model reduced MACs to 338,283,680 but missed the threshold by 1.07 points; its 0.4483 cross-entropy was better than the qualifying 67-unit, 24-step model’s 0.4548, suggesting a compact timing-sensitive readout may correct decisions without restoring a recurrent step.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1766851783559673313, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 21128, "peak_hidden_elements": 174592, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338727040, "training_seconds": 41.42003950010985, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.44180428908646474}

RECENT RESULT
hypothesis: Four contiguous temporal-bin means will capture nonlinear phoneme timing omitted by the trend statistic, raising the 68-unit, 23-step GRU from 84.29% to at least 85% while remaining below the verified 67-unit, 24-step model’s dense MAC cost.
change: Use 68 hidden units over 23 frames and replace the global mean readout with four causal temporal-bin means, retaining final, maximum, and deviation summaries.
mechanism: Coarse temporal-pyramid recurrent readout
evidence_used: The 68-unit, 23-step model had lower cross-entropy than the qualifying 67-unit, 24-step model, and adding a linear trend improved both accuracy and cross-entropy but remained short of 85%; a four-bin summary provides richer timing information while preserving the one-step MAC saving.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1771477050003876321, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 22216, "peak_hidden_elements": 281088, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 339613760, "training_seconds": 50.24004374979995, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.44104288487346627}

RECENT RESULT
hypothesis: A 68-unit, 23-step GRU augmented with four temporal-bin deviations will achieve at least 85% validation accuracy while using fewer dense inference MACs than the verified 67-unit, 24-step model.
change: Use 23 frames and summarize recurrent outputs with final, maximum, global deviation, four temporal-bin means, and four temporal-bin deviations.
mechanism: Temporal-bin deviation readout
evidence_used: The 68-unit, 23-step four-bin-mean model reached 84.66% accuracy at 339,613,760 MACs; adding within-bin deviations supplies localized activation-duration information while keeping the estimated total near 341.4M MACs, below the qualifying 343,793,080-MAC design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Four temporal-bin means and deviations will raise the 68-unit, 23-step GRU from 84.66% to at least 85% accuracy while keeping total inference below 343,793,080 MACs.
change: Use a 68-unit GRU over 23 frames and accumulate causal first and second moments in four contiguous temporal bins for an 11-summary classifier.
mechanism: Localized temporal-deviation readout
evidence_used: The 68-unit, 23-step four-bin-mean model reached 84.66%; localized deviations add complementary duration information with an estimated total cost of 341,387,200 MACs.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Four temporal-bin means and deviations will raise the 68-unit, 23-step GRU from 84.66% to at least 85% accuracy while keeping total inference below the verified 343,793,080-MAC model.
change: Use a 68-unit GRU over 23 frames and accumulate first and second moments in four temporal bins, classifying from final, maximum, global deviation, bin-mean, and bin-deviation summaries.
mechanism: Localized temporal-deviation readout
evidence_used: The 68-unit, 23-step four-bin-mean model reached 84.66% at 339,613,760 MACs; four localized deviation summaries add activation-duration information while keeping estimated cost near 341,387,200 MACs.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding four temporal-bin deviation summaries to the 68-unit, 23-step GRU will raise validation accuracy from 84.66% to at least 85% while keeping total inference below 343,793,080 MACs.
change: Use a 68-unit GRU over 23 frames, accumulate first and second moments in four contiguous temporal bins, and classify from final, maximum, global deviation, bin-mean, and bin-deviation summaries.
mechanism: Localized temporal-deviation readout
evidence_used: The verified 68-unit, 23-step four-bin-mean model reached 84.66% at 339,613,760 MACs; localized deviations add complementary activation-duration information with an estimated total of 341,387,200 MACs.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 68-unit GRU using the qualifying 24-frame schedule with only the earliest boundary frame removed will reach at least 85% accuracy in 23 steps and use fewer MACs than the verified 67-unit, 24-step model.
change: Increase hidden width from 67 to 68 and replace uniform 23-frame resampling with the verified 24-frame sampling pattern minus frame zero.
mechanism: Boundary-aware temporal subsampling
evidence_used: Uniform 68-unit/23-step sampling achieved 83.93% accuracy but better cross-entropy than the qualifying 67-unit/24-step model; preserving 23 of that qualifying model’s exact sample locations tests whether schedule displacement, rather than insufficient capacity, caused the regression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1764539150337571809, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20584, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338283680, "training_seconds": 35.3120228338521, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.438861083984375}

RECENT RESULT
hypothesis: Reducing the qualifying 68-unit, 23-step GRU to 67 units will retain at least 85% validation accuracy while lowering total dense inference MACs by approximately 2.6%.
change: Keep the successful 23-frame boundary-aware schedule and four-statistic readout, but reduce recurrent width from 68 to 67 and resize the classifier and states accordingly.
mechanism: Redundant-width pruning on the boundary-aware schedule
evidence_used: The 67-unit GRU already achieved 85.03% accuracy with the full qualifying 24-frame schedule, while removing its earliest boundary frame and using 68 units improved accuracy to 85.52%; this tests whether that added unit was unnecessary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1718936913989268001, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20081, "peak_hidden_elements": 137728, "recurrent_macs": 327793815, "recurrent_steps": 18745, "total_inference_macs": 329541175, "training_seconds": 35.89221137482673, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4525481616061158}

RECENT RESULT
hypothesis: Anchoring one of eight logits at zero will preserve at least 85% accuracy because eight-class softmax has only seven independent logit differences, while reducing classifier MACs by 12.5% and total validation MACs by 221,680.
change: Change the classifier to produce seven learned logits and append a zero reference logit, preserving the successful 68-unit, 23-step recurrent model and four-statistic readout.
mechanism: Reference-class softmax parameterization
evidence_used: The current 68-unit boundary-aware model qualifies at 85.52% accuracy; the failed 67-unit pruning shows recurrent capacity should be retained, motivating an expressivity-preserving reduction confined to the classifier.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1763382833726521056, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20311, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338062000, "training_seconds": 37.75031812489033, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.4591554981067868}

RECENT RESULT
hypothesis: A 69-unit GRU retaining the latest 22 indices of the successful boundary-aware schedule will preserve at least 85% accuracy while reducing total dense inference MACs by approximately 1.8% versus the qualifying 68-unit, 23-step model.
change: Increase recurrent width from 68 to 69, resize the four-statistic classifier from 272 to 276 inputs, and remove the earliest remaining scheduled frame to execute 22 recurrent steps.
mechanism: Boundary-preserving width–time exchange
evidence_used: Removing frame zero from the verified 24-frame pattern allowed the 68-unit, 23-step model to reach 85.52%, whereas uniform 23-frame resampling reached only 83.93%; preserving the successful sample locations while dropping the next earliest boundary frame tests another favorable width–time exchange.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1732408852659308533, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 21093, "peak_hidden_elements": 141824, "recurrent_macs": 330324390, "recurrent_steps": 17930, "total_inference_macs": 332123910, "training_seconds": 44.80489349993877, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.45988281811673215}

RECENT RESULT
hypothesis: A 69-unit GRU executing 22 steps will retain at least 85% validation accuracy when frame 1 is preserved and frame 2 is removed, while reducing total inference MACs by approximately 1.8% versus the qualifying 68-unit, 23-step model.
change: Increase recurrent and summary width from 68 to 69, resize the classifier from 272 to 276 inputs, and reduce the successful boundary-aware schedule to 22 frames by removing its second-earliest frame instead of its earliest frame.
mechanism: Earliest-boundary-preserving width–time exchange
evidence_used: The 68-unit schedule containing frames 1 and 2 achieved 85.52%, whereas the 69-unit variant that removed frame 1 fell to 84.42%; this suggests the earliest boundary frame is disproportionately informative and motivates preserving frame 1 while removing adjacent frame 2.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1732408852659308533, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 21093, "peak_hidden_elements": 141824, "recurrent_macs": 330324390, "recurrent_steps": 17930, "total_inference_macs": 332123910, "training_seconds": 43.765470375074074, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.46720665305670056}



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
    """A compact causal GRU with mean, final, max, and deviation readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(272, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, square_summary, running_max, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden, summary, square_summary, running_max, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            square_summary + output.square(),
            torch.maximum(running_max, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden, summary, square_summary, running_max, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            square_summary + outputs.square().sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, square_summary, running_max, count = state
        safe_count = count.clamp_min(1.0)
        mean = summary / safe_count
        deviation = (
            square_summary / safe_count - mean.square()
        ).clamp_min(0.0).sqrt()
        features = torch.cat(
            (
                mean,
                hidden[:, 0, :],
                running_max,
                deviation,
            ),
            dim=1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 23:
            return list(range(available_frames))
        qualifying_schedule = [
            step * (available_frames - 1) // 23
            for step in range(24)
        ]
        return qualifying_schedule[1:]


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


