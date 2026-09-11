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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4211441137302266104, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807384640, "training_seconds": 132.73040741705336, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40266498027403663}
prior_hypothesis: A 92-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the successful 96-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 28 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 12.5% versus the 32-step baseline.
change: Select 28 unique, approximately uniform causal frame indices spanning the first through final input frames.
mechanism: Seven-of-eight temporal sampling with endpoint preservation
evidence_used: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 83.93%; 28 steps is the most informative midpoint for locating the minimum schedule that clears 85%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 109.27272541704588, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4741032419029189}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will preserve at least 85% accuracy while reducing recurrent MACs by approximately 22% versus the successful 128-unit baseline.
change: Reduce the GRU hidden state and classifier input from 128 to 112 units while retaining the complete causal frame schedule.
mechanism: Full-rate recurrent width reduction
evidence_used: Temporal reduction was costly—28 frames achieved only 83.93%—while the 32-frame baseline achieved 87.24%; preserving temporal evidence and reducing recurrent width targets compute without repeating the demonstrated failure mode.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 146.85255179204978, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42283919632800515}

RECENT RESULT
hypothesis: A 108-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 6.5% versus the successful 112-unit model.
change: Reduce the GRU hidden state and classifier input from 112 to 108 units while preserving the full causal frame schedule and training procedure.
mechanism: Conservative full-rate recurrent width reduction
evidence_used: The 112-unit full-rate GRU achieved 85.77% accuracy, whereas reducing temporal coverage to 28 frames failed at 83.93%; this supports preserving all frames and making a smaller structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5645409772253230232, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 1081589760, "recurrent_steps": 26080, "total_inference_macs": 1082293920, "training_seconds": 80.48066208302043, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4493955928123802}

RECENT RESULT
hypothesis: A 104-unit full-rate GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.7% versus the successful 108-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input from 108 to 104 units while preserving all 32 causal frames and the established training procedure.
mechanism: Conservative recurrent width reduction
evidence_used: The 108-unit full-rate model achieved 86.01% accuracy, while temporal subsampling repeatedly failed; the remaining accuracy margin supports another small structural width reduction without discarding temporal evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 69.56130520813167, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}

RECENT RESULT
hypothesis: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the successful 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input from 104 to 100 units while preserving the full causal schedule and established training procedure.
mechanism: Conservative full-rate recurrent width reduction
evidence_used: The 104-unit full-rate GRU achieved 85.52% accuracy, and prior 112- and 108-unit models also cleared 85%; temporal subsampling repeatedly failed, so another small width reduction is the best-supported route to lower compute without discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 90.91042829095386, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}

RECENT RESULT
hypothesis: Replacing the eight learned logits with seven learned relative logits and one fixed reference logit will retain at least 85% accuracy while eliminating 100 classifier MACs per example and 101 parameters, because softmax is invariant to a shared logit offset.
change: Change the classifier to produce seven logits and append a fixed zero as the eighth class logit.
mechanism: Reference-class softmax head
evidence_used: The 100-unit full-rate GRU barely cleared the requirement at 85.03%, making further recurrent reductions risky; preserving its recurrent computation while removing the mathematically redundant eighth affine score is the lowest-risk structural cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900316758335593927, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37347, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939450500, "training_seconds": 96.30393720907159, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4956174323895226}

RECENT RESULT
hypothesis: Removing one algebraically redundant affine-free LayerNorm coordinate will retain at least 85% validation accuracy while reducing every GRU step by 300 MACs.
change: Disable the absorbable LayerNorm affine parameters, pass 19 normalized bands to the 100-unit GRU, and preserve all 32 causal steps.
mechanism: LayerNorm zero-sum input-coordinate elimination
evidence_used: The full-rate 100-unit model achieved 85.03%, while temporal and output-head reductions failed; affine-free LayerNorm outputs sum to zero, so one coordinate is recoverable from the other 19 and its GRU contribution can be represented by the remaining input weights and biases.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4859930700229051188, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37108, "peak_hidden_elements": 102912, "recurrent_macs": 931056000, "recurrent_steps": 26080, "total_inference_macs": 931708000, "training_seconds": 152.10125641687773, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4655018344247268}

RECENT RESULT
hypothesis: A 160-unit recurrent cell using one dense hidden transform and an input-conditioned retention gate will reach at least 85% accuracy while cutting per-step recurrent MACs from 36,000 to 32,000 versus the 100-unit GRU.
change: Replace the GRU with a wider custom gated recurrence that shares one learned hidden transform, preserves information through an input-driven retention path, and classifies from both the final state and its temporal mean.
mechanism: Shared-transform input-gated recurrent cell with endpoint-and-trajectory readout
evidence_used: Full temporal coverage is load-bearing: 24- and 28-frame schedules failed, while full-rate GRUs reached 85.03–87.24%. The prior designs assumed three dense GRU transforms were necessary; this patch instead spends fewer MACs on a wider state with one recurrent transform and a richer temporal readout.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 160-unit orthogonally initialized tanh RNN processing all 32 frames will reach at least 85% validation accuracy while reducing recurrent MACs by 20% versus the 100-unit GRU; jointly classifying the final state and temporal mean will offset the loss of GRU gating.
change: Replace the 100-unit GRU with a fused 160-unit standard RNN, preserve full-rate causal processing, and classify from the concatenated final hidden state and mean recurrent output.
mechanism: Fused single-transform Elman recurrence with endpoint-and-trajectory readout
evidence_used: Full-rate GRUs consistently cleared 85% while every 16–28-frame schedule failed, so all frames should be retained. The prior 160-unit shared-transform recurrence targeted lower MACs but timed out; using the evaluator-supported fused nn.RNN sequence path tests the same efficient single-transform idea without its training-speed failure.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3928755735212512208, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31728, "peak_hidden_elements": 164352, "recurrent_macs": 751104000, "recurrent_steps": 26080, "total_inference_macs": 753190400, "training_seconds": 86.33514370792545, "validation_accuracy": 0.7374233128834355, "validation_cross_entropy": 0.7670786290081001}

RECENT RESULT
hypothesis: Evaluating the learned classifier only from step 28 onward and exiting examples whose top-two logit margin exceeds `3 + remaining_steps` will preserve at least 85% accuracy while reducing exact MACs through selective 28–31-step inference.
change: Preserve the verified 100-unit GRU and training path exactly, use a zero-MAC recurrent-state proxy before step 28, and add conservative confidence-based exits during the final four possible recurrent steps.
mechanism: Confidence-gated late recurrent exit with deferred learned-head evaluation
evidence_used: The full-rate 100-unit GRU achieved 85.03%, while forcing every example to use 28 steps achieved only 83.93%; conditional late exit retains all 32 steps for uncertain examples while shortening only predictions already separated by a large margin.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 96-unit full-rate GRU classifying from both its final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by 7.2% versus the successful 100-unit model.
change: Reduce the GRU from 100 to 96 units, preserve all 32 causal frames, and concatenate the final hidden state with the mean recurrent output for classification.
mechanism: Narrow gated recurrence with endpoint-and-trajectory readout
evidence_used: The 100-unit full-rate GRU achieved 85.03%, while every tested temporal reduction failed; retaining full temporal coverage and enriching the readout is the most direct way to compensate for another structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 185.94730154192075, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.41307916933773486}

RECENT RESULT
hypothesis: A 92-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.5% versus the successful 96-unit model.
change: Reduce the GRU hidden width and both classifier input branches from 96 to 92 units, preserving all 32 causal frames and the established training procedure.
mechanism: Conservative full-rate recurrent width reduction with trajectory readout
evidence_used: The 96-unit endpoint-and-trajectory model achieved 86.26% accuracy with a 1.26-point margin, while temporal subsampling repeatedly failed; a modest width reduction preserves complete temporal evidence and directly targets the dominant recurrent MAC cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4211441137302266104, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807384640, "training_seconds": 132.73040741705336, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40266498027403663}



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
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((hidden[:, 0, :], mean_output), dim=-1))

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


