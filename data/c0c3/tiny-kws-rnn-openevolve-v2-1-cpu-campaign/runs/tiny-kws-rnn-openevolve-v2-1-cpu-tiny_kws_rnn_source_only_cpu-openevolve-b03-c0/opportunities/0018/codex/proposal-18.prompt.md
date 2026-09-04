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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722703125947029984, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36104, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905399800, "training_seconds": 63.04487991705537, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45813509701219807}
prior_hypothesis: Retaining the verified 98-unit recurrent dynamics while excluding one summary coordinate from the classifier will preserve at least 85% validation accuracy and reduce exact classifier MACs by 6,520 over the validation set.

## Recent verification evidence

RECENT RESULT
hypothesis: A 104-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.7% versus the verified 108-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 108 to 104 while preserving all 32 causal steps and the verified training procedure.
mechanism: Recurrent-width boundary search with full temporal coverage
evidence_used: The 108-unit full-frame model achieved 86.01% accuracy, slightly exceeding the 112-unit model’s 85.77%, so another modest four-unit reduction is the most direct test of the feasible width boundary without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 105.70535412500612, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}

RECENT RESULT
hypothesis: A 100-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the verified 104-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 104 to 100 while preserving all 32 causal steps and the training procedure.
mechanism: Recurrent-width boundary search with full temporal coverage
evidence_used: The 104-unit full-frame model achieved 85.52% accuracy, and each prior width reduction from 120 through 108 to 104 remained feasible; another four-unit reduction directly probes the remaining width boundary without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 93.20833162497729, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}

RECENT RESULT
hypothesis: A 99-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.8% versus the verified 100-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 100 to 99 while preserving all 32 causal steps and the verified training procedure.
mechanism: Fine-grained recurrent-width boundary search
evidence_used: The 100-unit full-frame GRU achieved 85.03% accuracy, while all tested widths from 104 through 120 also passed; because the remaining accuracy margin is narrow, a one-unit reduction is the most informative low-risk refinement of the feasible width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 100.78154466697015, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4628224940387749}

RECENT RESULT
hypothesis: A 98-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.84% versus the verified 99-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 99 to 98 while preserving all 32 causal steps and the verified training procedure.
mechanism: Fine-grained recurrent-width boundary search
evidence_used: The 99-unit full-frame GRU achieved 85.03% accuracy, identical to the 100-unit result, so a one-unit reduction is the most informative next probe of the feasible width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 108.62356691597961, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4702767307772958}

RECENT RESULT
hypothesis: A 97-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.9% versus the verified 98-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 98 to 97 while preserving all 32 causal steps and the verified training procedure.
mechanism: One-unit recurrent-width boundary probe
evidence_used: The 98-unit model achieved 85.03% accuracy, matching the verified 99- and 100-unit models; this stable accuracy across consecutive widths motivates testing the next one-unit structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634959100755551733, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888578200, "training_seconds": 100.91902008303441, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.45978345812464055}

RECENT RESULT
hypothesis: Removing the eight classifier bias parameters will retain at least 85% validation accuracy while preserving exact MACs and recurrent steps, improving the parameter tiebreaker.
change: Disable the bias in the 98-to-8 classifier while leaving the verified recurrent architecture and training procedure unchanged.
mechanism: Bias-free classification head
evidence_used: The 98-unit model achieved 85.03% accuracy, whereas reducing recurrent width to 97 failed; a smaller head-only structural reduction is the lowest-risk next optimization.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119704, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36104, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 146.27364645898342, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4729783602287433}

RECENT RESULT
hypothesis: A 99-unit GRU processing 31 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 1.3% versus the verified 98-unit, 32-step model.
change: Increase GRU width from 98 to 99 and use 31 unique increasing frame indices spanning the complete recording.
mechanism: Capacity-for-time exchange with mild uniform subsampling
evidence_used: The 98- and 99-unit full-frame models both achieved 85.03%, while 97 units failed; restoring one unit provides the smallest evidenced capacity buffer for a much milder temporal reduction than the unsuccessful 24- and 28-frame trials.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4661082503522968417, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893586375, "training_seconds": 213.3947905418463, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4816822824302627}

RECENT RESULT
hypothesis: Training the 97-unit GRU with batch size 64 will approximately double optimizer updates and recover validation accuracy to at least 85%, while retaining the previously measured 97-unit inference cost.
change: Reduce the GRU, state, summary, and classifier width from 98 to 97, and reduce training batch size from 128 to 64.
mechanism: Smaller-batch training–compensated recurrent-width compression
evidence_used: The 97-unit model narrowly missed at 84.42% while achieving lower cross-entropy (0.4598) than the passing 98-unit model (0.4703), suggesting its representation remains viable and may benefit from more optimization updates per fixed 50,000-example exposure.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Removing label smoothing will sharpen class boundaries enough for the 97-unit GRU to recover the 0.58-point accuracy shortfall and reach at least 85%, while retaining its lower measured inference cost and avoiding the batch-64 timeout.
change: Reduce the GRU and classifier width from 98 to 97, keep batch size 128, and replace label-smoothed cross-entropy with standard cross-entropy.
mechanism: Hard-target training-compensated recurrent-width compression
evidence_used: The 97-unit model achieved 84.42% accuracy but lower validation cross-entropy than the passing 98-unit model (0.4598 versus 0.4703), indicating viable representations; the attempted smaller-batch remedy timed out, motivating a zero-runtime-cost objective adjustment.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634959100755551733, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888578200, "training_seconds": 163.8366866670549, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.45162806598686733}

RECENT RESULT
hypothesis: Merging the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.85% versus the verified 98-unit model.
change: Preserve the 98-unit GRU and all 32 causal steps, but average the final two mel bands into one feature and reduce the GRU input width from 20 to 19.
mechanism: Fixed adjacent-band spectral pooling
evidence_used: The 98-unit full-frame model passed at 85.03%, while reducing hidden width to 97 and reducing temporal coverage both failed; this motivates an orthogonal, minimal reduction in redundant adjacent spectral resolution.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4682742184241608696, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35816, "peak_hidden_elements": 100864, "recurrent_macs": 897099840, "recurrent_steps": 26080, "total_inference_macs": 897738800, "training_seconds": 180.0237707500346, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4613856496986436}

RECENT RESULT
hypothesis: Alternating the final GRU input between mel bands 18 and 19 will preserve their distinct spectral cues and achieve at least 85% accuracy while retaining the 19-input model’s lower 897,738,800-MAC inference cost.
change: Keep the passing 98-unit, 32-step architecture, reduce its GRU input to 19 features, and alternate which of the two highest mel bands occupies the final input channel at each causal step.
mechanism: Temporal multiplexing of adjacent high-frequency mel bands
evidence_used: Averaging the two highest bands reduced inference MACs by 0.85% and narrowly missed accuracy at 84.91% with better cross-entropy than the passing baseline; temporal multiplexing tests whether retaining their distinct values avoids the information loss from averaging.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4682742184241608696, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35816, "peak_hidden_elements": 100864, "recurrent_macs": 897099840, "recurrent_steps": 26080, "total_inference_macs": 897738800, "training_seconds": 126.76342004095204, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46272703767554163}

RECENT RESULT
hypothesis: Retaining the verified 98-unit recurrent dynamics while excluding one summary coordinate from the classifier will preserve at least 85% validation accuracy and reduce exact classifier MACs by 6,520 over the validation set.
change: Reduce only the classification head input from 98 to 97 and classify from the first 97 coordinates of the averaged recurrent summary.
mechanism: One-coordinate recurrent readout pruning
evidence_used: The 98-unit GRU passed at 85.03%, while shrinking the recurrent state to 97 failed; pruning one coordinate only at readout preserves the full recurrent capacity and is a lower-risk structural reduction than altering recurrence, temporal coverage, or spectral inputs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722703125947029984, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36104, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905399800, "training_seconds": 63.04487991705537, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45813509701219807}



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
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
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
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        return self.classifier(averaged[:, :97])

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


