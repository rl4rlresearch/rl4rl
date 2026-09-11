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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1198556178784957731, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24301, "peak_hidden_elements": 182784, "recurrent_macs": 227866665, "recurrent_steps": 21190, "total_inference_macs": 229777840, "training_seconds": 41.95087758405134, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43152330082618384}
prior_hypothesis: A 67-unit paired-frame GRU with the successful six-transition early pool split into two three-transition summaries will retain at least 85% accuracy while reducing total inference MACs from 234,980,800 to approximately 229,777,840.

## Recent verification evidence

RECENT RESULT
hypothesis: Representing each non-affine layer-normalized frame with its first 19 coordinates will retain at least 85% accuracy while reducing total inference MACs by approximately 3.55M, because the omitted coordinate is determined by the normalization’s zero-sum constraint.
change: Disable redundant LayerNorm affine parameters, reduce the GRU input width from 20 to 19, and slice one normalized coordinate in both recurrent execution paths.
mechanism: Lossless zero-mean mel coordinate elimination
evidence_used: Directly fusing two high-frequency bands at the same 19-input GRU width reached only 84.54%, indicating that its information loss was harmful; this patch achieves the same recurrent MAC reduction without discarding spectral information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1429845007897611051, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15146, "peak_hidden_elements": 89600, "recurrent_macs": 272984250, "recurrent_steps": 20375, "total_inference_macs": 274118730, "training_seconds": 44.831994000123814, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4811973618583445}

RECENT RESULT
hypothesis: A 56-unit GRU processing the verified 26-frame window at indices 3–28 will retain at least 85% validation accuracy while reducing total dense inference MACs below the successful 58-unit, 25-step model.
change: Reduce GRU width from 58 to 56 units, resize its recurrent state and classifier, and restore the 26-frame central schedule.
mechanism: Capacity-for-time recurrent tradeoff
evidence_used: The 58-unit 26-step model achieved 86.01% accuracy, while increasing width to 59 could not compensate for reducing the schedule to 24 steps. This indicates temporal context is more valuable than marginal width and motivates exchanging two hidden units for one additional frame; 56 units is the largest integer width that makes this 26-step configuration cheaper than the current 25-step model.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1416963981025129766, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 14496, "peak_hidden_elements": 86528, "recurrent_macs": 270553920, "recurrent_steps": 21190, "total_inference_macs": 271649280, "training_seconds": 41.23041675006971, "validation_accuracy": 0.838036809815951, "validation_cross_entropy": 0.4688794375928633}

RECENT RESULT
hypothesis: Mapping normalized 20-band frames into a well-conditioned 19-dimensional Helmert contrast basis will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.55M versus the verified 25-step model.
change: Preserve affine input normalization, replace the raw 20 normalized coordinates with 19 orthonormal spectral contrasts, and structurally reduce the GRU input width from 20 to 19.
mechanism: Orthonormal lossless mel-coordinate elimination
evidence_used: Direct coordinate omission with a 19-input GRU narrowly missed at 84.91%; because normalized frames lie in a 19-dimensional zero-mean subspace, an orthonormal contrast basis preserves that information while avoiding the poor conditioning of reconstructing the omitted coordinate as the sum of the other 19.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1429845007897611091, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15186, "peak_hidden_elements": 89600, "recurrent_macs": 272984250, "recurrent_steps": 20375, "total_inference_macs": 274118730, "training_seconds": 40.68544124998152, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.46309915554304065}

RECENT RESULT
hypothesis: A 70-unit GRU that jointly consumes adjacent frames from the verified 26-frame window will retain at least 85% accuracy while reducing learned GRU transitions from 25 to 13 and estimated total validation MACs from 277.66M to 246.11M.
change: Cache the first frame of each causal pair without a dense operation, concatenate it with the next frame, and execute one wider GRU transition per pair; vectorize the identical paired computation during sequence training.
mechanism: Lossless causal frame-pair transitions
evidence_used: The 58-unit model reached 86.01% on frames 3–28, while deleting frames and compensating with width failed. This challenges the load-bearing assumption that every retained frame needs its own full recurrent transition: all 26 proven-useful frames remain available, but temporal updates occur at a learned two-frame clock.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1283766510225752138, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25248, "peak_hidden_elements": 118784, "recurrent_macs": 244744500, "recurrent_steps": 21190, "total_inference_macs": 246113700, "training_seconds": 25.08536475012079, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42380269992570935}

RECENT RESULT
hypothesis: A 69-unit paired-frame GRU will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 2.32%, from 246.11M to 240.40M.
change: Reduce the paired GRU hidden width from 70 to 69 and resize its recurrent state and three-way pooled classifier accordingly.
mechanism: One-unit paired-GRU width pruning
evidence_used: The 70-unit paired-frame model achieved 85.77% accuracy with 13 learned transitions, providing a 0.77-point margin and motivating a conservative width-boundary probe that preserves all 26 input frames and the successful temporal mechanism.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1253987106327190496, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24681, "peak_hidden_elements": 117248, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240404625, "training_seconds": 33.33605133299716, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.448084123295509}

RECENT RESULT
hypothesis: A 69-unit paired-frame GRU with separate early/late means will reach at least 85% accuracy while reducing estimated total inference MACs from 246.11M to 240.85M.
change: Reduce the paired GRU from 70 to 69 units and replace its global mean with separate means for the first six and remaining seven learned transitions, expanding the classifier from three to four pooled views.
mechanism: Recurrent-to-readout MAC reallocation with temporal-pyramid pooling
evidence_used: The 69-unit model narrowly missed the threshold at 84.79%, only 0.21 points short, while the otherwise identical 70-unit model reached 85.77%; spending 552 additional classifier MACs per example on explicit temporal structure tests whether cheap readout capacity can recover that narrow loss.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1256333748861381728, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25233, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240854505, "training_seconds": 32.31677362485789, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4130140246057803}

RECENT RESULT
hypothesis: Early/late mean pooling plus max pooling will retain at least 85% validation accuracy while removing 552 classifier MACs per example, reducing total validation MACs from 240.85M to approximately 240.40M.
change: Remove the final hidden-state view from the classifier while preserving the temporal-pyramid summaries, max pooling, 69-unit paired GRU, and 26-frame schedule.
mechanism: Redundant final-state readout removal
evidence_used: The 69-unit model passed at 85.89% only after adding separate early/late summaries; unlike the failed global-mean three-view model, this patch preserves that successful temporal structure and removes the final state, which is already represented within the late summary and recurrent maximum.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1253987106327190496, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24681, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240404625, "training_seconds": 33.265574417077005, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4450192410521712}

RECENT RESULT
hypothesis: Replacing eight learned logits with seven learned relative logits will preserve at least 85% accuracy while reducing total validation MACs by 168,705, from 240,404,625 to 240,235,920.
change: Produce seven classifier outputs, append a fixed reference logit, and mean-center all eight logits without changing the recurrent model or the representable class probabilities.
mechanism: Softmax gauge-dimension elimination
evidence_used: The current 69-unit paired-GRU design passes at 85.28%. Softmax probabilities have only seven independent logit differences, so removing the unidentifiable common-offset degree reduces classifier cost without reducing predictive expressiveness.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1253107115376868783, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24473, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240235920, "training_seconds": 39.14977308316156, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46183706646316625}

RECENT RESULT
hypothesis: Restoring the informative final-state view while reducing the paired GRU from 69 to 68 units will retain at least 85% validation accuracy and reduce total inference MACs from 240,235,920 to approximately 234,980,800.
change: Resize the paired GRU to 68 units and classify four 68-dimensional views: early mean, late mean, temporal maximum, and final recurrent state.
mechanism: Recurrent-to-readout capacity reallocation
evidence_used: The 69-unit four-view model achieved 85.89% accuracy, substantially above the threshold; removing its final-state view reduced accuracy to 85.28%, showing that cheap readout capacity can support recurrent-width pruning.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1225695609832554381, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24391, "peak_hidden_elements": 150528, "recurrent_macs": 233429040, "recurrent_steps": 21190, "total_inference_macs": 234980800, "training_seconds": 37.61118116695434, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43288474053692966}

RECENT RESULT
hypothesis: A 67-unit paired-frame GRU with the successful six-transition early pool split into two three-transition summaries will retain at least 85% accuracy while reducing total inference MACs from 234,980,800 to approximately 229,777,840.
change: Reduce GRU width from 68 to 67 and replace the early/late/max/final readout with first-early/second-early/late/max/final pooling, preserving the established transition-six boundary while adding inexpensive temporal resolution.
mechanism: Temporal-pyramid readout for recurrent-width pruning
evidence_used: The 68-unit four-view model achieved 85.77%, and at 69 units the informative final-state view improved accuracy from 85.28% to 85.89%. This shows that cheap temporally structured readout capacity can support recurrent-width pruning.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1198556178784957731, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24301, "peak_hidden_elements": 182784, "recurrent_macs": 227866665, "recurrent_steps": 21190, "total_inference_macs": 229777840, "training_seconds": 41.95087758405134, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43152330082618384}

RECENT RESULT
hypothesis: A 66-unit paired-frame GRU with the seven-transition late pool split into three- and four-transition summaries will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 224,627,040.
change: Reduce GRU width from 67 to 66 and add a sixth pooled classifier view by splitting the late summary at transition nine; preserve all 26 frames, 13 learned transitions, max pooling, final-state pooling, and seven relative logits.
mechanism: Progressive temporal-pyramid readout for recurrent-width pruning
evidence_used: Reducing the successful 68-unit model to 67 units while adding one inexpensive temporal summary retained 85.52% accuracy and lowered MACs; applying the same recurrent-to-readout trade reallocates capacity toward temporal structure while pruning another hidden unit.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1171688822234078833, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24203, "peak_hidden_elements": 214016, "recurrent_macs": 222367860, "recurrent_steps": 21190, "total_inference_macs": 224627040, "training_seconds": 32.4138487088494, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4517080295305311}

RECENT RESULT
hypothesis: A 76-unit GRU consuming 27 central frames in nine causal three-frame transitions will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 229,611,580.
change: Cache two frames, concatenate each causal frame triplet into a 60-feature GRU input, widen the hidden state to 76 units, and use three equal transition summaries plus maximum and final-state views.
mechanism: Causal three-frame transition packing
evidence_used: Pairing preserved all useful frames and reached 85.77% with only 13 learned transitions, whereas deleting frames failed; triplet packing tests whether further transition reduction can preserve accuracy while a 76-unit width keeps recurrent MACs slightly below the successful 67-unit paired model.
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
    """A causal GRU that performs one learned transition per adjacent frame pair."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(335, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        frame = self.input_norm(frame)

        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )

        paired = torch.cat((pending, frame), dim=1)
        output, hidden = self.gru(
            paired.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        early_weight = (count < 3.0).to(dtype=output.dtype)
        middle_weight = (
            (count >= 3.0) & (count < 6.0)
        ).to(dtype=output.dtype)
        late_weight = (count >= 6.0).to(dtype=output.dtype)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            middle_summary + middle_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        if frames.shape[1] == 0:
            return state

        frames = self.input_norm(frames)
        paired_parts: list[torch.Tensor] = []
        position = 0

        if phase[0, 0].item() >= 0.5:
            paired_parts.append(
                torch.cat((pending.unsqueeze(1), frames[:, :1, :]), dim=2)
            )
            position = 1

        pair_count = (frames.shape[1] - position) // 2
        if pair_count > 0:
            end = position + 2 * pair_count
            paired_parts.append(
                torch.cat(
                    (
                        frames[:, position:end:2, :],
                        frames[:, position + 1:end:2, :],
                    ),
                    dim=2,
                )
            )
            position = end

        if paired_parts:
            paired = torch.cat(paired_parts, dim=1)
            outputs, hidden = self.gru(
                paired,
                hidden.transpose(0, 1).contiguous(),
            )
            sequence_maximum = outputs.amax(dim=1)
            maximum = torch.where(
                count > 0,
                torch.maximum(maximum, sequence_maximum),
                sequence_maximum,
            )
            positions = count.unsqueeze(1) + torch.arange(
                paired.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
            early_weights = (positions < 3.0).to(dtype=outputs.dtype)
            middle_weights = (
                (positions >= 3.0) & (positions < 6.0)
            ).to(dtype=outputs.dtype)
            late_weights = (positions >= 6.0).to(dtype=outputs.dtype)
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            middle_summary = middle_summary + (
                outputs * middle_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
            count = count + paired.shape[1]
            hidden = hidden.transpose(0, 1)
            pending = torch.zeros_like(pending)
            phase = torch.zeros_like(phase)

        if position < frames.shape[1]:
            pending = frames[:, position, :]
            phase = torch.ones_like(phase)

        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=3.0).clamp_min(1.0)
        middle_count = (count - 3.0).clamp(max=3.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                middle_summary / middle_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        relative_logits = self.classifier(pooled)
        logits = torch.cat(
            (
                relative_logits,
                torch.zeros_like(relative_logits[:, :1]),
            ),
            dim=1,
        )
        return logits - logits.mean(dim=1, keepdim=True)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(26, available_frames)
        steps -= steps % 2
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))


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


