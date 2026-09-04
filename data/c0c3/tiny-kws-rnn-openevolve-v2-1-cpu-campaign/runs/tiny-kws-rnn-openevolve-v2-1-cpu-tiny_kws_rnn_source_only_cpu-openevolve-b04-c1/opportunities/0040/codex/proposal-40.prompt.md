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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1448337571346396515, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277663980, "training_seconds": 31.00489095808007, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45725814725723735}
prior_hypothesis: Processing 25 contiguous frames at indices 3–27 will achieve at least 85% validation accuracy while retaining the failed 25-step design’s approximately 277.7M total MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 29 contiguous near-central frames (indices 2–30) will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs by roughly 3.3% versus the verified 30-step model.
change: Reduce the schedule from 30 to 29 steps and bias the odd-length window toward later frames, removing the earliest remaining boundary frame.
mechanism: Boundary-frame temporal pruning
evidence_used: The 58-unit GRU improved from 85.15% at 31 steps to 85.77% at 30 central steps, while dropping the original earliest boundary frame already passed; this supports testing one further boundary removal without reducing recurrent capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1679124763513239695, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 320774220, "recurrent_steps": 23635, "total_inference_macs": 321908700, "training_seconds": 50.2952406660188, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4395944794262845}

RECENT RESULT
hypothesis: Processing the 28 contiguous central frames (indices 2–29) will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs by approximately 3.4% versus the verified 29-step model.
change: Reduce the causal schedule from 29 to 28 steps, removing the latest remaining boundary frame while preserving the 58-unit GRU, temporal pooling, and training procedure.
mechanism: Symmetric boundary-frame temporal pruning
evidence_used: The central 30-step model achieved 85.77% accuracy and the 29-step model achieved 85.64%; both passed, indicating that boundary-frame pruning has preserved accuracy so far and motivating the complementary boundary removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1621427965471528900, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 15360, "peak_hidden_elements": 89600, "recurrent_macs": 309713040, "recurrent_steps": 22820, "total_inference_macs": 310847520, "training_seconds": 42.28234770800918, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.42989435488460986}

RECENT RESULT
hypothesis: A 42-unit fast GRU processing all 28 scheduled frames plus a 52-unit context GRU updated every fourth fast output will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 15.4% versus the verified 58-unit single-rate GRU.
change: Replace the homogeneous recurrent state with full-rank fast and slow GRUs, maintain separate mean/maximum/final summaries for both timescales, and update the command-scale state only seven times per recording.
mechanism: Hierarchical multi-rate fast/slow GRU
evidence_used: The 28-step 58-unit GRU reached 86.99% accuracy, leaving a 1.99-point margin. The failed MGU showed that weakening the gated update itself was harmful; this alternative retains standard GRUs while challenging the assumption that every unit must perform an equally expensive update at every frame.
result: the implementation could not be verified

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

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the recurrent model represents time, updates state, controls computation, or forms command predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
    """A compact causal GRU with mean, maximum, and final temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
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
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return hidden.transpose(0, 1), summary + output, maximum, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        sequence_maximum = outputs.amax(dim=1)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, sequence_maximum),
            sequence_maximum,
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            maximum,
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(25, available_frames)
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


