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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5014660316337748473, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 959924115, "recurrent_steps": 22005, "total_inference_macs": 961371555, "training_seconds": 136.92549666692503, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.4030273999173217}
prior_hypothesis: Removing the penultimate input frame from the qualified dual-readout model will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.57%.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019696634925428970, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45360, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770624880, "training_seconds": 47.418838291894644, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4611427821996022}
prior_hypothesis: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when one of its 110 final-state readout coordinates is removed, while reducing dense classifier MACs by 6,520 and learned parameters by 8.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019730644237518698, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770631400, "training_seconds": 52.34599075000733, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46311264506146954}
prior_hypothesis: Omitting frame 0 from the qualified 110-unit, 23-step schedule while retaining frame 3 will achieve at least 85% validation accuracy with 22 recurrent steps and approximately 4.35% fewer recurrent MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the penultimate input frame from the qualified dual-readout model will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.57%.
change: Reduce the causal schedule from 28 to 27 steps by omitting frame 30 in addition to frame 1, while preserving both endpoints, every interior selected frame, and the 111-unit dual-view readout.
mechanism: Nested trailing-edge frame omission
evidence_used: The current 28-step dual-readout design achieved 87.48% accuracy, a 2.48-point margin; moreover, the earlier single-readout transition from 29 steps to the nested 28-step schedule lost only 0.245 points, supporting an isolated additional edge-adjacent omission.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5014660316337748473, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 959924115, "recurrent_steps": 22005, "total_inference_macs": 961371555, "training_seconds": 136.92549666692503, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.4030273999173217}

RECENT RESULT
hypothesis: A 111-unit dual-readout GRU omitting frame 29 from the qualified 27-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.7%.
change: Classify from concatenated mean and final recurrent states, and use 26 frames by excluding indices 1, 29, and 30 from the qualified 29-frame base schedule.
mechanism: Dual-view readout with nested trailing-edge subsampling
evidence_used: The 27-step dual-readout design achieved 87.48% accuracy, and removing frame 30 from its 28-step predecessor caused no accuracy loss, supporting a further isolated trailing-edge omission with a 2.48-point margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4829211788595015713, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 46113, "peak_hidden_elements": 114176, "recurrent_macs": 924371370, "recurrent_steps": 21190, "total_inference_macs": 925818810, "training_seconds": 178.81841133302078, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.418622584430718}

RECENT RESULT
hypothesis: A 110-unit dual-readout GRU on the qualified 26-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.7% versus the 111-unit design.
change: Use the qualified 26-step nested schedule and concatenated mean/final-state readout, while reducing the GRU and summary width from 111 to 110.
mechanism: Dual-view temporal readout with conservative recurrent-width reduction
evidence_used: The 111-unit dual-readout 26-step model achieved 86.13% accuracy; the observed 111-to-110 width reduction cost 0.86 percentage points at 30 steps, implying approximately 85.28% if that effect transfers.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4749230388888015958, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 909051000, "recurrent_steps": 21190, "total_inference_macs": 910485400, "training_seconds": 138.12378745805472, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.4115896681335075}

RECENT RESULT
hypothesis: Removing frame 28 from the current 32-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.85%.
change: Extend the qualified nested schedule to 25 steps by additionally omitting `available_frames - 4`, preserving the 110-unit dual-view GRU and training procedure.
mechanism: Nested trailing-frame causal subsampling
evidence_used: The current 26-step design achieved 86.63% accuracy; the preceding isolated 27-to-26-step reduction lost 1.35 points, leaving enough observed margin to test one further trailing-edge omission.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4566855452725391643, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 874087500, "recurrent_steps": 20375, "total_inference_macs": 875521900, "training_seconds": 109.8094190841075, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4312623661719948}

RECENT RESULT
hypothesis: A 110-unit dual-readout GRU using 24 nested frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 4% versus the qualified 25-step design.
change: Adopt the qualified 110-unit width and additionally omit frame 27 from the 25-step schedule, excluding frames 1 and 27–30 for 32-frame inputs.
mechanism: Nested trailing-frame causal subsampling
evidence_used: The qualified 110-unit, 25-step design achieved 86.13% accuracy; its preceding 26-to-25-step reduction lost only 0.49 percentage points, leaving a 1.13-point margin for one further isolated trailing-frame removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4384480516562767328, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 839124000, "recurrent_steps": 19560, "total_inference_macs": 840558400, "training_seconds": 102.825183958048, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4569458569485717}

RECENT RESULT
hypothesis: A 110-unit GRU using 23 frames and separate early/late mean readouts plus the final state will retain at least 85% accuracy while reducing total inference MACs by about 4.1% versus the qualified 24-step design.
change: Use the qualified 110-unit width, omit frame 25 in addition to frames 1 and 27–30, and replace the global mean with separately accumulated first-12-step and remaining-step means.
mechanism: Phase-split temporal pooling with nested causal subsampling
evidence_used: The 110-unit 24-step model qualified at 85.40%, while adding a complementary temporal view previously raised the nested 28-step model from 84.79% to 87.48%; phase-split pooling preserves both coarse temporal order and the final-state view while one fewer recurrent step saves substantially more MACs than its wider classifier adds.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4205846604730013093, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 46248, "peak_hidden_elements": 169472, "recurrent_macs": 804160500, "recurrent_steps": 18745, "total_inference_macs": 806312100, "training_seconds": 104.6492619998753, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.4539595247046348}

RECENT RESULT
hypothesis: Omitting frame 2 from the qualified 110-unit, 24-step schedule while retaining the global-mean/final-state readout will preserve at least 85% accuracy and reduce total inference MACs by approximately 4.16%.
change: Reduce execution to 23 recurrent steps by additionally excluding frame index 2, without changing the qualified architecture, readout, or training procedure.
mechanism: Nested early-edge frame subsampling
evidence_used: The qualified 24-step model achieved 85.40%; previously removing adjacent early frame 1 cost only 0.245 points, while the failed 23-step experiment also changed the readout and removed late frame 25, so an isolated early-frame removal is the most informative lower-cost test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4202105580400143013, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 804160500, "recurrent_steps": 18745, "total_inference_macs": 805594900, "training_seconds": 60.93729812512174, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45483384395669574}

RECENT RESULT
hypothesis: Omitting frame 3 from the qualified 110-unit, 23-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 4.35%.
change: Use 22 recurrent steps by excluding frames 1–3 and 27–30 for 32-frame inputs, while preserving the qualified dual-readout GRU and training procedure.
mechanism: Nested leading-edge frame subsampling
evidence_used: The qualified 23-step design achieved 85.52% accuracy after removing frame 2 from the 24-step schedule, slightly improving accuracy; this supports testing one further adjacent early-frame omission while retaining the more informative late frames implicated by the failed frame-25 removal.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019730644237518698, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770631400, "training_seconds": 78.29272337513976, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4751080752881758}

RECENT RESULT
hypothesis: A 109-unit dual-readout GRU on the qualified 23-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.67%.
change: Adopt the qualified 23-step schedule and reduce the GRU, summary, and dual-view classifier widths from 110 to 109 units.
mechanism: Conservative recurrent-width reduction on the qualified sparse schedule
evidence_used: The 110-unit 23-step design achieved 85.52% accuracy, while the prior 111-to-110 reduction on the qualified 26-step schedule retained accuracy and improved it from 86.13% to 86.63%, supporting another one-unit reduction without changing temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4131931616067016269, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 44629, "peak_hidden_elements": 112128, "recurrent_macs": 790720335, "recurrent_steps": 18745, "total_inference_macs": 792141695, "training_seconds": 71.14847308304161, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4664676876887222}

RECENT RESULT
hypothesis: The qualified 110-unit, 23-step model will retain at least 85% validation accuracy while eliminating 179,300 dense validation MACs, because eight-way softmax needs only seven independent logits after removing its common-offset degree of freedom.
change: Restore the qualified 23-frame dual-readout GRU, but learn seven classifier outputs and derive the eighth as their negative sum, reducing total inference MACs and parameters without restricting softmax decision expressivity.
mechanism: Seven-coordinate centered logit basis
evidence_used: The qualified 110-unit, 23-step design reached 85.52% accuracy, while reducing it to 109 units or 22 steps missed the threshold; this motivates preserving its recurrent path and removing redundancy only from the eight-way output parameterization.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4201170324317675492, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 45147, "peak_hidden_elements": 113152, "recurrent_macs": 804160500, "recurrent_steps": 18745, "total_inference_macs": 805415600, "training_seconds": 67.66855654190294, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4553224528494057}

RECENT RESULT
hypothesis: Omitting frame 0 from the qualified 110-unit, 23-step schedule while retaining frame 3 will achieve at least 85% validation accuracy with 22 recurrent steps and approximately 4.35% fewer recurrent MACs.
change: Exclude frame 0 in addition to frames 1, 2, and 27–30, preserving the qualified dual-readout GRU and training procedure.
mechanism: Leading-silence frame omission
evidence_used: The 23-step model qualified at 85.52% after frames 1 and 2 were removed, whereas removing frame 3 reduced accuracy to 84.54%; this isolates whether the remaining earliest frame is less informative than frame 3 at the same 22-step cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019730644237518698, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770631400, "training_seconds": 52.34599075000733, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46311264506146954}

RECENT RESULT
hypothesis: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when one of its 110 final-state readout coordinates is removed, while reducing dense classifier MACs by 6,520 and learned parameters by 8.
change: Adopt the qualified 22-frame schedule and classify from the complete 110-coordinate mean output plus 109 coordinates of the final hidden state.
mechanism: Single-coordinate complementary-readout pruning
evidence_used: The 110-unit, 22-step design achieved 85.03% accuracy, while reducing recurrent width to 109 units and reducing outputs to seven logits both failed; pruning one of 220 readout coordinates preserves the qualified recurrent path and all eight logits.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019696634925428970, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45360, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770624880, "training_seconds": 47.418838291894644, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4611427821996022}



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
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 4:
            return [
                frame
                for frame in schedule
                if frame not in (1, available_frames - 2)
            ]
        return schedule


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
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(219, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
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
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 6:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    0,
                    1,
                    2,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
        return schedule


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
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(220, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if available_frames >= 6:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    0,
                    1,
                    2,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
        return schedule


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
