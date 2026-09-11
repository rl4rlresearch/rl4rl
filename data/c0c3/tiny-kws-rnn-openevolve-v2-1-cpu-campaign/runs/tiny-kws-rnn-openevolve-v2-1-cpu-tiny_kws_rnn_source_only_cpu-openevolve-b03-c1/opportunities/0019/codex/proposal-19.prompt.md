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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 61.01974808308296, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.422041311439561}
prior_hypothesis: An 80-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the successful 84-unit model.

## Recent verification evidence

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

RECENT RESULT
hypothesis: An 88-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.8% versus the successful 92-unit model.
change: Reduce the GRU hidden width, temporal summary, and classifier input from 92 to 88 units while preserving all 32 causal frames and the established training procedure.
mechanism: Conservative recurrent width reduction with trajectory readout
evidence_used: The 92-unit endpoint-and-trajectory model achieved 86.50% accuracy, slightly exceeding the 96-unit model’s 86.26%; this suggests the enriched readout retains enough capacity for another modest width reduction without discarding temporal evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 124.77520741592161, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3970399072565184}

RECENT RESULT
hypothesis: An 84-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.1% versus the successful 88-unit model.
change: Reduce the GRU hidden width and both classifier input branches from 88 to 84 units while preserving all 32 causal frames and the established training procedure.
mechanism: Conservative recurrent width reduction with trajectory readout
evidence_used: The endpoint-and-trajectory design remained above the requirement through successive reductions from 96 to 92 to 88 units, with the 88-unit model achieving 85.77%; another four-unit reduction directly tests the remaining capacity margin without repeating the failed temporal-subsampling strategy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3570977772028654184, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 684600000, "training_seconds": 62.599689666880295, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.44686067733296586}

RECENT RESULT
hypothesis: An 80-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.4% versus the successful 84-unit model.
change: Reduce the GRU hidden width and both classifier input branches from 84 to 80 units while preserving all 32 causal frames and the established training procedure.
mechanism: Conservative recurrent width reduction with trajectory readout
evidence_used: Successive endpoint-and-trajectory GRU reductions from 96 to 92 to 88 to 84 units all exceeded 85% accuracy; the 84-unit model achieved 85.64%, supporting another four-unit structural reduction without discarding temporal evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 61.01974808308296, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.422041311439561}

RECENT RESULT
hypothesis: A 76-unit full-rate GRU using final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by 8.8% versus the successful 80-unit model.
change: Reduce the GRU hidden width and both classifier input branches from 80 to 76 units while preserving all 32 causal frames and the established training procedure.
mechanism: Conservative recurrent width reduction with trajectory readout
evidence_used: Successive four-unit reductions from 96 through 80 units all met the accuracy requirement; the 80-unit model achieved 85.64%, supporting one more structural width reduction without repeating failed temporal subsampling.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2982752710124852568, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23608, "peak_hidden_elements": 78336, "recurrent_macs": 570839040, "recurrent_steps": 26080, "total_inference_macs": 571830080, "training_seconds": 61.12833937490359, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4455052288032017}



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
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
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


