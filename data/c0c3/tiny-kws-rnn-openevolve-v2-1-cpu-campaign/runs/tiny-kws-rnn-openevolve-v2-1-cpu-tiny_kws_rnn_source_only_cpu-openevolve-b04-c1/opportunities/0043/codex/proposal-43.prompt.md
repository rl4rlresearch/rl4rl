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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1256333748861381728, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25233, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240854505, "training_seconds": 32.31677362485789, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4130140246057803}
prior_hypothesis: A 69-unit paired-frame GRU with separate early/late means will reach at least 85% accuracy while reducing estimated total inference MACs from 246.11M to 240.85M.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 27 contiguous near-central frames (indices 3–29) will retain at least 85% validation accuracy while reducing recurrent steps by 3.6% and total dense inference MACs to approximately 299.8M.
change: Reduce the causal schedule from 28 to 27 steps by removing the earliest remaining boundary frame, preserving the verified 58-unit GRU, pooling, and training procedure.
mechanism: Continued boundary-frame temporal pruning
evidence_used: The 28-step model achieved 86.99% accuracy, a 1.99-point margin above threshold, after every contiguous boundary-pruning result from 31 through 28 steps passed; this supports probing one additional boundary frame without weakening recurrent capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1563731167429818105, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 298651860, "recurrent_steps": 22005, "total_inference_macs": 299786340, "training_seconds": 40.470391666051, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4336063033964005}

RECENT RESULT
hypothesis: Processing the 26 contiguous central frames (indices 3–28) will retain at least 85% validation accuracy while reducing recurrent steps by 3.7% and total dense inference MACs to approximately 288.7M.
change: Reduce the causal schedule from 27 to 26 steps, removing the latest remaining boundary frame while preserving the verified 58-unit GRU, temporal pooling, and training procedure.
mechanism: Continued symmetric boundary-frame pruning
evidence_used: The contiguous 28-step and 27-step models achieved 86.99% and 86.38% accuracy respectively, and every boundary-pruning design from 31 through 27 steps passed; the current 1.38-point margin supports testing the complementary boundary removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1506034369388107310, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 287590680, "recurrent_steps": 21190, "total_inference_macs": 288725160, "training_seconds": 34.169467833125964, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4451760321307036}

RECENT RESULT
hypothesis: Processing the 25 contiguous near-central frames at indices 4–28 will retain at least 85% validation accuracy while reducing recurrent steps by 3.8% and total dense inference MACs to approximately 277.7M.
change: Reduce the causal schedule from 26 to 25 steps by removing the earliest remaining boundary frame while preserving the verified 58-unit GRU, temporal pooling, and training procedure.
mechanism: Continued boundary-frame temporal pruning
evidence_used: The 26-step central-window model achieved 86.01% accuracy, and every contiguous boundary-pruning design from 31 through 26 steps passed; its 1.01-point margin supports probing one additional boundary removal.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448337571346396515, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277663980, "training_seconds": 39.21840350003913, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4698159902373706}

RECENT RESULT
hypothesis: Processing 25 contiguous frames at indices 3–27 will achieve at least 85% validation accuracy while retaining the failed 25-step design’s approximately 277.7M total MACs.
change: Shift the 25-frame window one frame earlier, dropping the latest boundary frame from the verified 26-frame schedule instead of the earliest.
mechanism: Complementary boundary-frame temporal pruning
evidence_used: The 26-frame window at indices 3–28 achieved 86.01% accuracy, while indices 4–28 narrowly missed at 84.91%; testing the complementary 3–27 window isolates which boundary frame is more informative at identical inference cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448337571346396515, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277663980, "training_seconds": 31.00489095808007, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45725814725723735}

RECENT RESULT
hypothesis: A 59-unit GRU processing frames 3–26 will retain at least 85% validation accuracy while executing 24 recurrent steps and fewer dense MACs than the verified 58-unit, 25-step model.
change: Increase GRU width from 58 to 59 units, resize its state and classifier, and remove the latest frame from the successful 3–27 window.
mechanism: Capacity-for-time recurrent tradeoff
evidence_used: The 58-unit model passed with frames 3–27 but failed with frames 4–28, indicating earlier frames are more valuable at this boundary; one added recurrent unit provides a modest capacity buffer while the 24-step design still reduces recurrent MACs by approximately 1.1%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1432676283047580081, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 15801, "peak_hidden_elements": 91136, "recurrent_macs": 273507480, "recurrent_steps": 19560, "total_inference_macs": 274661520, "training_seconds": 27.843760833144188, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.4776791531615462}

RECENT RESULT
hypothesis: Fusing the two highest adjacent mel bands will preserve at least 85% validation accuracy while reducing the successful 25-step model by exactly 174 GRU MACs per recurrent step, approximately 3.55M validation MACs overall.
change: Keep the verified 58-unit GRU, 25-frame schedule, and temporal readout, but average the two highest normalized mel bands so the GRU structurally consumes 19 inputs instead of 20.
mechanism: Adjacent high-frequency mel-band fusion
evidence_used: Removing a recurrent step failed even after increasing width to 59, indicating temporal context is currently more valuable than added capacity; the successful 58-unit, 25-step design motivates preserving its dynamics while testing redundancy between adjacent spectral bands.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1429845007897611091, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15186, "peak_hidden_elements": 89600, "recurrent_macs": 272984250, "recurrent_steps": 20375, "total_inference_macs": 274118730, "training_seconds": 45.39927483280189, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4697868300361867}

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
        self.gru = nn.GRU(40, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(276, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
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
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            early_summary,
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
        early_weight = (count < 6.0).to(dtype=output.dtype)
        late_weight = 1.0 - early_weight
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
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
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            early_summary,
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
            early_weights = (positions < 6.0).to(dtype=outputs.dtype)
            late_weights = 1.0 - early_weights
            early_summary = early_summary + (
                outputs * early_weights
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
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            hidden,
            early_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        del pending, phase
        early_count = count.clamp(max=6.0).clamp_min(1.0)
        late_count = (count - 6.0).clamp_min(1.0)
        pooled = torch.cat(
            (
                early_summary / early_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(pooled)

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


