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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 96.61382112489082, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}
prior_hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the qualified 112-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4493518622856917429, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36229, "peak_hidden_elements": 99840, "recurrent_macs": 860197455, "recurrent_steps": 25265, "total_inference_macs": 861462335, "training_seconds": 138.08664666698314, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4214622076303681}
prior_hypothesis: A 97-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the qualified 98-unit, 31-step design.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4578588665885341461, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 876493380, "recurrent_steps": 25265, "total_inference_macs": 877771300, "training_seconds": 133.824828249868, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40440371460709834}
prior_hypothesis: A 98-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs to approximately 877,771,300 and recurrent steps from 26,080 to 25,265.

## Recent verification evidence

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the qualified 112-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 104 while preserving the proven full-frame schedule and training procedure.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 112-unit full-resolution GRU achieved 86.13% accuracy and improved cost over the qualified 120-unit model, while reduced-frame designs failed; another eight-unit width reduction is the most direct test of remaining capacity margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 96.61382112489082, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 13% versus the qualified 104-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 96 while preserving the full-frame schedule and established training procedure.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 104-unit full-resolution GRU achieved 85.89% accuracy, while prior 112- and 120-unit reductions also remained qualified; reduced-frame designs failed, so testing the next eight-unit width reduction is the most informative cost-boundary experiment.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 100.25730083300732, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.44714848805058954}

RECENT RESULT
hypothesis: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the qualified 104-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 112 to 100 while preserving the proven full-frame schedule and training procedure.
mechanism: Full-resolution boundary-width reduction
evidence_used: The 104-unit model qualified at 85.89% accuracy, while the 96-unit model reached only 84.29%; testing the 100-unit midpoint is the most informative next probe of the width–accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 70.43287016591057, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4471566978407784}

RECENT RESULT
hypothesis: A 98-unit GRU using all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 3.6% versus the qualified 100-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 120 to 98 while preserving the full-frame schedule and established training procedure.
mechanism: Full-resolution midpoint width reduction
evidence_used: The 100-unit model qualified at 85.40% accuracy while the 96-unit model reached 84.29%; 98 units is the most informative untested midpoint at the observed width–accuracy boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 65.70172241679393, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4501637090203221}

RECENT RESULT
hypothesis: A 99-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.8% versus the qualified 100-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 100 to 99 while preserving the full-frame schedule and established training procedure.
mechanism: Full-resolution one-unit boundary refinement
evidence_used: The 100-unit model qualified at 85.40% accuracy while the 98-unit model reached 84.54%; 99 units is the only untested integer midpoint at the observed width–accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 118.49872245802544, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45291225690783166}

RECENT RESULT
hypothesis: A learned 20-to-16 feature projection feeding a 100-unit GRU will retain at least 85% validation accuracy while reducing total dense MACs below the qualified 99-unit full-input GRU.
change: Add a 16-dimensional linear frontend bottleneck and use the previously qualified 100-unit recurrent width throughout the state and classifier.
mechanism: Learned input bottleneck with recurrent-capacity tradeoff
evidence_used: The full-input 100-unit GRU achieved 85.40% accuracy; replacing four input dimensions with a learned projection reduces estimated per-step MACs from 35,343 for the qualified 99-unit design to 35,120 while restoring one unit of recurrent capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4781029096180900264, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36584, "peak_hidden_elements": 102912, "recurrent_macs": 915929600, "recurrent_steps": 26080, "total_inference_macs": 916581600, "training_seconds": 196.61224879091606, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.5152046063194977}

RECENT RESULT
hypothesis: A 98-unit GRU classifying the concatenated mean and final recurrent outputs will reach at least 85% accuracy while using fewer total MACs than the qualified 99-unit mean-pooled GRU.
change: Reduce the GRU width from 120 to 98 and give the classifier separate learned weights for the sequence-wide mean and terminal hidden state, retaining all 32 frames.
mechanism: Dual-view recurrent pooling
evidence_used: The mean-pooled 98-unit model narrowly missed at 84.54%, while 99 units qualified at 85.40%; adding the terminal recurrent representation costs only 784 classifier MACs per example and directly tests whether pooling—not recurrent capacity—caused the boundary failure.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4726070047843913056, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 906045280, "training_seconds": 185.1498672501184, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.39721636625886697}

RECENT RESULT
hypothesis: A 96-unit GRU classifying concatenated mean and terminal outputs will retain at least 85% validation accuracy while reducing total dense MACs by about 3.7% versus the qualified 98-unit dual-view model.
change: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 98 to 96 while retaining all 32 frames and the established training procedure.
mechanism: Dual-view recurrent pooling with narrower state
evidence_used: The 98-unit dual-view model achieved 87.12% accuracy versus 84.54% for 98-unit mean-only pooling, showing that the terminal representation recovered substantial accuracy and provides margin for the next two-unit width reduction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 97-unit GRU classifying concatenated mean and terminal outputs will achieve at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the qualified 98-unit dual-view model.
change: Reduce recurrent width from 99 to 97 and replace mean-only classification with a linear classifier over concatenated sequence-mean and terminal recurrent states.
mechanism: Dual-view recurrent pooling with one-unit width refinement
evidence_used: The 98-unit dual-view model achieved 87.12% accuracy and 906,045,280 total MACs, substantially outperforming the same-width mean-only model’s 84.54%; the 96-unit dual-view attempt produced no accuracy evidence because training timed out, making 97 units the closest informative capacity reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4638258004028255349, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36229, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 889210640, "training_seconds": 193.08655041712336, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.41533637485621167}

RECENT RESULT
hypothesis: A 96-unit GRU using concatenated sequence-mean and terminal states will achieve at least 85% validation accuracy with approximately 872,532,480 total inference MACs, 1.88% fewer than the qualified 97-unit model.
change: Reduce recurrent width from 104 to 96 and expand the classifier input to combine mean-pooled and terminal recurrent representations.
mechanism: Dual-view recurrent pooling at the unresolved width boundary
evidence_used: The 97-unit dual-view model achieved 86.01% accuracy, and the 98-unit version achieved 87.12%; the prior 96-unit attempt timed out without producing accuracy evidence, leaving it as the closest unresolved cost boundary.
result: the edit reproduced a previously verified implementation

RECENT RESULT
hypothesis: A 98-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs to approximately 877,771,300 and recurrent steps from 26,080 to 25,265.
change: Increase recurrent width from 97 to 98, retain mean-plus-terminal classification, and omit only the earliest input frame.
mechanism: Single-edge-frame omission with capacity insurance
evidence_used: The 98-unit dual-view model achieved 87.12% accuracy across all 32 frames, providing 2.12 percentage points of margin; because larger reductions to 24 or 28 steps failed, skipping a single likely edge-context frame is the most conservative unresolved temporal-cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4578588665885341461, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 876493380, "recurrent_steps": 25265, "total_inference_macs": 877771300, "training_seconds": 133.824828249868, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40440371460709834}

RECENT RESULT
hypothesis: A 97-unit dual-view GRU processing frames 1–31 will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the qualified 98-unit, 31-step design.
change: Reduce the GRU and dual-view pooling width from 98 to 97 and omit the earliest input frame, yielding 31 recurrent steps.
mechanism: Joint one-unit width refinement and single-edge-frame omission
evidence_used: The 97-unit dual-view model qualified at 86.01% using all frames, while the 98-unit 31-step model qualified at 86.50%; combining these individually successful, minimal reductions is the closest unresolved cost refinement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4493518622856917429, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36229, "peak_hidden_elements": 99840, "recurrent_macs": 860197455, "recurrent_steps": 25265, "total_inference_macs": 861462335, "training_seconds": 138.08664666698314, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4214622076303681}



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
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
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
        return self.classifier(summary / count.clamp_min(1.0))

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
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
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
        return self.classifier(summary / count.clamp_min(1.0))

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
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
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
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))


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
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)

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
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(1, available_frames))


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
