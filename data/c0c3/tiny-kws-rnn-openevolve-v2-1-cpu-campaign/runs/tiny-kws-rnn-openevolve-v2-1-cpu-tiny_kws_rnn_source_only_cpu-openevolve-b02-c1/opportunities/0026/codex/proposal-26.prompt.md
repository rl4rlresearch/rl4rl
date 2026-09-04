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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 54.18845779192634, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45900973454574867}
prior_hypothesis: An orthonormal seven-coordinate parameterization of the eight centered logits will retain at least 85% accuracy while matching the prior seven-logit MAC and parameter reductions.

## Recent verification evidence

RECENT RESULT
hypothesis: A 60-unit GRU paired with a 59-unit GRU will retain at least 85% validation accuracy while reducing exact recurrent and classifier MACs below the verified two-by-60 model.
change: Reduce one parallel GRU by one unit and resize its state, temporal summary, and classifier input from 120 aggregate features to 119.
mechanism: Asymmetric two-block gated recurrence
evidence_used: The two-by-60 model achieved 85.28% accuracy at 751,886,400 total MACs after successfully narrowing the two-by-64 design; reducing only one block is the smallest untested structural cost reduction at this accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3865192330916823897, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 740228640, "recurrent_steps": 26080, "total_inference_macs": 741004520, "training_seconds": 91.05980095919222, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.44478286087878643}

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when skipping only the first frame, while reducing recurrent steps from 32 to 31 and total inference MACs below 741,004,520.
change: Preserve the successful asymmetric parallel-GRU model and training procedure, but process frames 1–31 instead of all 32 frames.
mechanism: Single-frame boundary trimming on asymmetric block-diagonal recurrence
evidence_used: The current 60+59 model achieved 86.50% accuracy, providing a 1.50-point margin, and prior first-frame trimming preserved 85.03% accuracy in the 99-unit GRU while reducing its recurrent execution to 31 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3744531542705007737, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 717096495, "recurrent_steps": 25265, "total_inference_macs": 717872375, "training_seconds": 98.94699670816772, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.44520491149527897}

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and total inference MACs below 717,872,375.
change: Skip the first two input frames instead of only the first frame, preserving the model width and training procedure.
mechanism: Second boundary-frame trimming
evidence_used: Skipping one initial frame reduced the 60+59 model by 23,132,145 recurrent MACs while retaining 86.13% accuracy, leaving a 1.13-point margin above the requirement; removing one additional boundary frame is the largest conservative cost reduction directly supported by that result.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3623870754493191577, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 693964350, "recurrent_steps": 24450, "total_inference_macs": 694740230, "training_seconds": 115.44413508311845, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.45253378277176}

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 3–31, while reducing recurrent execution from 30 to 29 steps and lowering total inference MACs below 694,740,230.
change: Skip the first three input frames instead of the first two, preserving the model architecture and training procedure.
mechanism: Third boundary-frame trimming
evidence_used: Processing frames 2–31 achieved 85.64% accuracy at 694,740,230 MACs; the preceding one-frame trim also remained feasible, making one additional boundary trim the most direct test of the minimum viable temporal budget.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3503209966281375417, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671608085, "training_seconds": 93.30611424986273, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45579669226898006}

RECENT RESULT
hypothesis: Representing eight-class softmax probabilities with seven learned logits and one fixed-zero reference logit will retain at least 85% validation accuracy while reducing classifier MACs by 119 per example and learned parameters by 120.
change: Change the classifier from eight outputs to seven and append a fixed-zero eighth logit, preserving the full softmax function class and the verified 29-step recurrent architecture.
mechanism: Reference-class logit gauge fixing
evidence_used: The 60+59 GRU over frames 3–31 met the target at 85.03% accuracy; softmax is invariant to subtracting one class logit from every logit, so fixing that reference logit to zero removes redundant learned computation without reducing representational capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 69.34918704093434, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.481317831402176}

RECENT RESULT
hypothesis: Two independent 59-unit GRUs over frames 3–31 will retain at least 85% validation accuracy while reducing total inference MACs from 671,608,085 to approximately 661,745,770.
change: Reduce the 60-unit GRU to 59 units, producing two balanced 59-unit blocks and resizing the recurrent state, temporal summary, and classifier input to 118 features.
mechanism: Balanced two-block recurrent width reduction
evidence_used: Reducing the successful 60+60 model to 60+59 improved observed accuracy from 85.28% to 86.50% while lowering cost; removing one unit from the larger remaining block is the smallest recurrent-capacity reduction and is slightly cheaper than an unequal 60+58 split of the same aggregate width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3451766630581662271, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29666, "peak_hidden_elements": 121344, "recurrent_macs": 660976410, "recurrent_steps": 23635, "total_inference_macs": 661745770, "training_seconds": 77.23555104108527, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.47705680051464244}

RECENT RESULT
hypothesis: Two independent 100-unit tanh RNNs over the verified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs below 570 million.
change: Replace the gated 60+59 recurrence with wider 100+100 ungated recurrent blocks, resizing the state, temporal summary, and classifier while preserving causal mean pooling and frames 3–31.
mechanism: Wider ungated block-diagonal tanh recurrence
evidence_used: Parallel recurrent blocks already achieved 86.50% accuracy before trimming; replacing each GRU’s three gated matrices with one RNN transition permits 200 aggregate hidden features at 24,000 recurrent MACs per step versus 28,383 currently. This challenges the load-bearing assumption that learned gates are necessary for these short normalized sequences.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2965612016587133683, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 26048, "peak_hidden_elements": 205312, "recurrent_macs": 567240000, "recurrent_steps": 23635, "total_inference_macs": 568544000, "training_seconds": 57.03650337504223, "validation_accuracy": 0.7177914110429447, "validation_cross_entropy": 0.8712244513576016}

RECENT RESULT
hypothesis: Balanced 61+60 GRUs processing frames 4–31 will retain at least 85% validation accuracy while reducing exact total inference MACs from 671,608,085 to approximately 667,657,780.
change: Remove one additional leading frame, widen the recurrent blocks from 60+59 to 61+60 units, and resize the state, summary, and classifier to 121 features.
mechanism: Reallocate one recurrent step’s MAC budget into balanced hidden width
evidence_used: The 60+59 model remained feasible through three consecutive frame trims, reaching 85.03% at 29 steps, while 59+59 failed; this suggests recurrent width is at its capacity boundary. At 28 steps, 61+60 is the widest near-balanced two-block configuration that still costs less than the current design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482604574237516365, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30965, "peak_hidden_elements": 124416, "recurrent_macs": 666868860, "recurrent_steps": 22820, "total_inference_macs": 667657780, "training_seconds": 81.21272766706534, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.47602081298828125}

RECENT RESULT
hypothesis: Compressing each normalized 20-band frame to 16 learned features while preserving the verified 60+59 hidden capacity and 29-step schedule will retain at least 85% validation accuracy and reduce total inference MACs below 646 million.
change: Add a shared bias-free 20-to-16 linear projection and change both GRUs to consume its output, leaving recurrent width, pooling, classifier, training, and frame schedule unchanged.
mechanism: Learned low-rank mel input projection
evidence_used: The 60+59 model over frames 3–31 achieved 85.03%, while reducing hidden width to 59+59 failed at 84.29%; this motivates reducing input-side matrix rank without removing recurrent state capacity or additional temporal evidence.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3366611564273013929, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28989, "peak_hidden_elements": 122368, "recurrent_macs": 644644625, "recurrent_steps": 23635, "total_inference_macs": 645420505, "training_seconds": 78.76660037506372, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4927171110375527}

RECENT RESULT
hypothesis: Merging the two highest adjacent normalized mel bands into one feature will retain at least 85% accuracy while reducing total inference MACs from 671,608,085 to approximately 663,170,390.
change: Preserve the verified 60+59 hidden state and 29-frame schedule, but reduce each GRU’s input width from 20 to 19 by averaging the final two normalized mel bands.
mechanism: Adjacent high-mel band coalescing
evidence_used: The 20-to-16 projection reached 84.17%, while reducing recurrent width to 59+59 reached 84.29%; this suggests hidden capacity is more load-bearing and motivates a conservative one-feature input reduction that preserves information from every band.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3459197665273266165, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29740, "peak_hidden_elements": 122368, "recurrent_macs": 662394510, "recurrent_steps": 23635, "total_inference_macs": 663170390, "training_seconds": 49.59928729198873, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.4746463892650019}

RECENT RESULT
hypothesis: An orthonormal seven-coordinate parameterization of the eight centered logits will retain at least 85% accuracy while matching the prior seven-logit MAC and parameter reductions.
change: Replace the eight-output classifier with seven learned contrast coordinates and expand them through a fixed orthonormal basis into eight zero-sum logits.
mechanism: Orthonormal softmax gauge fixing
evidence_used: The fixed-zero seven-logit model reached 84.42% despite preserving softmax expressivity, suggesting its asymmetric parameterization hurt optimization; balanced orthonormal contrasts preserve the same full function class with better-conditioned, nonconstant logits for every class.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 54.18845779192634, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45900973454574867}

RECENT RESULT
hypothesis: A 61+60 GRU processing frames 4–31 with the verified orthonormal seven-coordinate classifier will recover the 0.46-point accuracy shortfall of the prior 28-step model, reaching at least 85% while reducing total inference MACs below the current 671,511,100.
change: Skip one additional leading frame, widen the parallel GRUs from 60+59 to 61+60 units, resize the recurrent state and summary to 121 features, and retain the seven-coordinate orthonormal classifier.
mechanism: Step-to-width reallocation with orthonormal logit contrasts
evidence_used: The prior 61+60, 28-step model reached 84.54%, narrowly missing the threshold, while the orthonormal seven-coordinate classifier subsequently produced 85.40% on the verified 29-step architecture and reduced classifier MACs; this directly tests whether its balanced parameterization can close the small 28-step accuracy gap.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482090183392159228, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30843, "peak_hidden_elements": 124416, "recurrent_macs": 666868860, "recurrent_steps": 22820, "total_inference_macs": 667559165, "training_seconds": 54.79871074995026, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.4892535437835506}



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
    """Asymmetric parallel causal GRUs with a shared online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a, hidden_b, summary, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output = torch.cat((output_a[:, 0, :], output_b[:, 0, :]), dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            summary + output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a, hidden_b, summary, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b), dim=2)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, summary, count = state
        coordinates = self.classifier(summary / count.clamp_min(1.0))
        coordinate_sum = coordinates.sum(dim=1, keepdim=True)
        scale = 1.0 / math.sqrt(8.0)
        correction = (scale * scale) / (1.0 - scale)
        return torch.cat(
            (
                coordinates - correction * coordinate_sum,
                scale * coordinate_sum,
            ),
            dim=1,
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))


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


