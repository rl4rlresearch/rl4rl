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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1793277019134885521, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 20081, "peak_hidden_elements": 137728, "recurrent_macs": 342045720, "recurrent_steps": 19560, "total_inference_macs": 343793080, "training_seconds": 35.397358583053574, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4548308109213238}
prior_hypothesis: A 67-unit GRU processing 24 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.4% versus the verified 66-unit, 25-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 65-unit GRU using 27 frames obtained by removing only the second frame from the verified 28-frame schedule will achieve at least 85% accuracy while matching the 65-unit/27-step model’s 366,004,275 MAC cost.
change: Restore the 65-unit GRU and replace the rephased 27-frame schedule with a 27-frame subset of the successful 28-frame schedule that retains both utterance endpoints.
mechanism: Nested temporal pruning with boundary preservation
evidence_used: The 65-unit model achieved 85.77% at 28 steps but narrowly missed at 27 steps with 84.79%; the prior 27-step formula moved many sampling locations, so pruning one early, densely sampled frame from the successful schedule isolates frame count from schedule rephasing.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1909133992177028853, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18573, "peak_hidden_elements": 100352, "recurrent_macs": 364732875, "recurrent_steps": 22005, "total_inference_macs": 366004275, "training_seconds": 40.939223709050566, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46081740490497985}

RECENT RESULT
hypothesis: A 65-unit, 27-step GRU augmented with per-unit temporal standard deviation will achieve at least 85% validation accuracy while using fewer dense inference MACs than the verified 66-unit, 27-step model.
change: Reduce the GRU width to 65 and extend the mean/final/maximum readout with a running second-moment statistic, producing a 260-feature classifier input without adding recurrent matrix cost.
mechanism: Variance-augmented temporal readout
evidence_used: The plain 65-unit, 27-step model reached 84.79%, only two validation examples short of qualification, while 65 units qualified at 28 steps. Temporal standard deviation supplies complementary activation-duration information at only one additional 65-by-8 classifier slice; unlike the unsuccessful minimum-statistic experiment, it preserves the stronger 65-unit recurrent core.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1911344597462861173, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 19093, "peak_hidden_elements": 133632, "recurrent_macs": 364732875, "recurrent_steps": 22005, "total_inference_macs": 366428075, "training_seconds": 44.28432758315466, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.41471749522203316}

RECENT RESULT
hypothesis: A 64-unit, 27-step GRU retaining the temporal-deviation readout will achieve at least 85% validation accuracy while reducing dense inference MACs versus the verified 65-unit model.
change: Reduce the GRU and all recurrent summary widths from 65 to 64 units, and resize the four-statistic classifier input from 260 to 256 features.
mechanism: Variance-assisted recurrent-width pruning
evidence_used: Temporal deviation raised the 65-unit, 27-step design from 84.79% to 86.50%; testing it at 64 units directly probes whether that accuracy gain permits the next structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1859901261763147973, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 354896640, "recurrent_steps": 22005, "total_inference_macs": 356565760, "training_seconds": 38.6987130001653, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.43020456700237253}

RECENT RESULT
hypothesis: A 64-unit GRU with deviation readout processing 26 uniformly spaced frames will achieve at least 85% validation accuracy while reducing dense inference MACs below the verified 64-unit, 27-step model.
change: Reduce the uniform full-utterance frame schedule from 27 to 26 recurrent steps while preserving recurrent width, four-statistic readout, and training procedure.
mechanism: Variance-assisted temporal pruning
evidence_used: The deviation readout raised the 65-unit, 27-step design from 84.79% to 86.50% and enabled 64 units at 27 steps to qualify at 85.40%; this motivates testing whether its temporal information allows one additional step reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1791338488508771638, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18608, "peak_hidden_elements": 131584, "recurrent_macs": 341752320, "recurrent_steps": 21190, "total_inference_macs": 343421440, "training_seconds": 40.89927391707897, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.45492850461620493}

RECENT RESULT
hypothesis: A 65-unit GRU with deviation readout processing 26 uniformly spaced frames will retain at least 85% validation accuracy while using approximately 1.0% fewer dense inference MACs than the verified 64-unit, 27-step model.
change: Increase the recurrent and summary width from 64 to 65 units, resize the classifier input from 256 to 260 features, and reduce the schedule from 27 to 26 steps.
mechanism: Width–time rebalancing with variance-assisted readout
evidence_used: The deviation readout gave the 65-unit, 27-step model 86.50% accuracy, while reducing both width and steps to 64 units and 26 frames reached 84.05%; restoring one unit preserves an estimated MAC advantage over the qualifying 64-unit, 27-step design while testing whether added recurrent capacity recovers the threshold.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1840881553895471733, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 19093, "peak_hidden_elements": 133632, "recurrent_macs": 351224250, "recurrent_steps": 21190, "total_inference_macs": 352919450, "training_seconds": 45.87364699994214, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.43728420515001915}

RECENT RESULT
hypothesis: Projecting each 20-band frame to 16 learned features while retaining the verified 65-unit, 26-step GRU will preserve at least 85% validation accuracy and reduce total dense inference MACs by approximately 2.8%.
change: Add a 20-to-16 linear input projection and change the GRU input width from 20 to 16, preserving recurrent capacity, temporal summaries, schedule, and training procedure.
mechanism: Learned spectral bottleneck before recurrence
evidence_used: The 65-unit, 26-step deviation-readout model achieved 86.13% accuracy, whereas reducing recurrent width to 64 at 26 steps failed with 84.05%; this motivates preserving the successful hidden width while reducing the more redundant spectral input dimension.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1790037632321339889, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18649, "peak_hidden_elements": 133632, "recurrent_macs": 341476850, "recurrent_steps": 21190, "total_inference_macs": 343172050, "training_seconds": 36.306638916023076, "validation_accuracy": 0.8196319018404908, "validation_cross_entropy": 0.4820497290488401}

RECENT RESULT
hypothesis: A 66-unit GRU with deviation readout processing 25 uniformly spaced frames will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 1.2% versus the verified 65-unit, 26-step model.
change: Increase recurrent and summary width from 65 to 66 units, resize the classifier input from 260 to 264 features, and reduce the uniform schedule from 26 to 25 steps.
mechanism: Width–time rebalancing with deviation-assisted readout
evidence_used: The 65-unit deviation-readout model achieved 86.13% at 26 steps, providing accuracy margin, while the same width–time exchange previously allowed 66 units at 27 steps to outperform the qualifying 65-unit, 28-step design in MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1818698980003451539, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 19584, "peak_hidden_elements": 135680, "recurrent_macs": 346945500, "recurrent_steps": 20375, "total_inference_macs": 348666780, "training_seconds": 42.63805266702548, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4410978539589724}

RECENT RESULT
hypothesis: A 67-unit GRU processing 24 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.4% versus the verified 66-unit, 25-step model.
change: Increase recurrent and summary width from 66 to 67 units, resize the classifier input from 264 to 268 features, and reduce the uniform schedule from 25 to 24 steps.
mechanism: Width–time rebalancing with deviation-assisted readout
evidence_used: Exchanging one step for one hidden unit previously produced qualifying 65-unit/26-step and 66-unit/25-step deviation-readout models; the latest achieved 85.64% accuracy, motivating the next incremental exchange.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1793277019134885521, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 20081, "peak_hidden_elements": 137728, "recurrent_macs": 342045720, "recurrent_steps": 19560, "total_inference_macs": 343793080, "training_seconds": 35.397358583053574, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4548308109213238}

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
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(268, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
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
        target_steps = min(24, available_frames)
        return [
            step * (available_frames - 1) // (target_steps - 1)
            for step in range(target_steps)
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


