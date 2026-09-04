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
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2837201355893873265, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 31625, "peak_hidden_elements": 93696, "recurrent_macs": 543332790, "recurrent_steps": 17930, "total_inference_macs": 543926110, "training_seconds": 111.14460374996997, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4392411776115558}
prior_hypothesis: A 91-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs to approximately 543,926,110.

REFERENCE DESIGN 1
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3042825907870834613, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 582714405, "recurrent_steps": 17115, "total_inference_macs": 583346845, "training_seconds": 114.14717325009406, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.44280826123945555}
prior_hypothesis: The 97-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing standard inference from 22 to 21 recurrent steps and total MACs to approximately 583.3 million.

REFERENCE DESIGN 2
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2444402303504125855, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 468061020, "recurrent_steps": 17115, "total_inference_macs": 468621740, "training_seconds": 136.71907783392817, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.45798510217959165}
prior_hypothesis: The 86-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 468,621,740.

REFERENCE DESIGN 3
verified_results: {"batch_size": 32, "examples_processed": 50000, "inference_cost": 3396556762916016377, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 1564, "p95_recurrent_steps": 21, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 650489805, "recurrent_steps": 17115, "total_inference_macs": 651161365, "training_seconds": 101.10433149989694, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.47942851247963}
prior_hypothesis: A 103-unit GRU trained at batch size 32 on frames 4–24 will retain at least 85% validation accuracy while reducing inference from 22 to 21 recurrent steps and total MACs to approximately 651.2 million.

## Recent verification evidence

RECENT RESULT
hypothesis: A 103-unit GRU trained at batch size 16 on 19 frames distributed across the qualified frames 4–24 window will retain at least 85% validation accuracy while reducing total inference MACs to approximately 589.2 million.
change: Use batch size 16 and cap the schedule at 19 evenly distributed frames, preserving both endpoints of the qualified 21-frame temporal window.
mechanism: Full-span interior frame thinning
evidence_used: The contiguous 20-frame batch-size-16 design qualified at 85.03%, while the 21-frame frames 4–24 design qualified at 85.15%; thinning two redundant interior frames tests a lower-cost frontier without discarding the later frames omitted by the prior trailing-trim strategy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3073408781604517737, "maximum_recurrent_steps": 19, "mean_recurrent_steps": 19.0, "median_recurrent_steps": 19, "optimizer_steps": 3127, "p95_recurrent_steps": 19, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 588538395, "recurrent_steps": 15485, "total_inference_macs": 589209955, "training_seconds": 130.31128349993378, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.48320044769099885}

RECENT RESULT
hypothesis: Allowing predictions above 90% confidence to exit after 16 recurrent steps will preserve at least 85% validation accuracy while lowering mean execution below the qualified fixed 20-step model and therefore reducing total inference MACs below 620,185,660.
change: Add a parameter-free early-exit rule to the qualified 22-step, batch-size-32 model; uncertain examples retain all 22 steps while confident examples stop after step 16 or later.
mechanism: Confidence-gated adaptive recurrent depth
evidence_used: The current 22-step model reaches 85.89% accuracy, providing more margin than the barely qualifying 20-step model at 85.03%; the failed fixed 19-step thinning result shows that removing context uniformly is risky, motivating conditional computation that preserves full context for difficult clips.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 97-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference cost to approximately 611.1 million MACs.
change: Replace the 103-unit, 20-step frontier with a narrower 97-unit GRU processing 22 contiguous frames, using batch size 16 for additional optimizer updates.
mechanism: Width-for-temporal-context exchange
evidence_used: The 103-unit 22-step design reached 85.89% accuracy, providing substantially more margin than the 20-step design’s 85.03%; reducing width to 97 lowers estimated recurrent MACs despite restoring two informative frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3187565289042172533, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 610462710, "recurrent_steps": 17930, "total_inference_macs": 611095150, "training_seconds": 129.0912150000222, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.42572902375203703}

RECENT RESULT
hypothesis: The 97-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing standard inference from 22 to 21 recurrent steps and total MACs to approximately 583.3 million.
change: Remove frame 25 from the current schedule, processing 21 contiguous frames for standard recordings while preserving valid schedules for shorter inputs.
mechanism: Incremental trailing-frame trim at reduced recurrent width
evidence_used: The current 97-unit, 22-step model achieved 86.50% accuracy, a 1.50-point margin; the analogous 103-unit batch-size-32 trim from 22 to 21 steps lost only 0.74 points and still qualified.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3042825907870834613, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 582714405, "recurrent_steps": 17115, "total_inference_macs": 583346845, "training_seconds": 114.14717325009406, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.44280826123945555}

RECENT RESULT
hypothesis: A 91-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs to approximately 543,926,110.
change: Narrow the GRU and classifier from 103 to 91 units while expanding the standard schedule from 20 to 22 contiguous frames.
mechanism: Recurrent width-for-context exchange
evidence_used: The 97-unit, 22-step design achieved 86.50% accuracy and 611,095,150 MACs; another six-unit width reduction preserves its temporal context and tests a structurally cheaper model against a 1.50-point accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2837201355893873265, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 31625, "peak_hidden_elements": 93696, "recurrent_macs": 543332790, "recurrent_steps": 17930, "total_inference_macs": 543926110, "training_seconds": 111.14460374996997, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4392411776115558}

RECENT RESULT
hypothesis: An 85-unit GRU trained at batch size 16 on frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs from 543,926,110 to approximately 480,629,950.
change: Replace the current 103-unit, 21-step model with an 85-unit GRU processing 22 contiguous frames, and use batch size 16.
mechanism: Incremental recurrent-width reduction
evidence_used: Reducing the 22-step GRU from 97 to 91 units lowered accuracy by only 0.37 points, from 86.50% to 86.13%, leaving 1.13 points of margin; repeating that six-unit reduction is the most direct test of a substantially cheaper frontier.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2507038954126867893, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 28013, "peak_hidden_elements": 87552, "recurrent_macs": 480075750, "recurrent_steps": 17930, "total_inference_macs": 480629950, "training_seconds": 114.5374007918872, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4446986379798936}

RECENT RESULT
hypothesis: An 86-unit GRU trained at batch size 16 on frames 4–25 will achieve at least 85% validation accuracy while reducing total inference MACs to approximately 490.9 million.
change: Reduce the GRU and classifier from 103 to 86 units, retain the qualified 22-frame schedule, and use batch size 16.
mechanism: Minimal recurrent-width rebound
evidence_used: The 85-unit design missed qualification by only 0.092 percentage points at 480.6 million MACs, while the 91-unit design achieved 86.13%; testing the adjacent 86-unit width is the smallest capacity increase and cheapest unresolved frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2560663136964334490, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 490349640, "recurrent_steps": 17930, "total_inference_macs": 490910360, "training_seconds": 148.6919713751413, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4445762213022431}

RECENT RESULT
hypothesis: The 86-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 468,621,740.
change: Remove frame 25 from the standard schedule while preserving valid schedules for shorter inputs.
mechanism: Incremental trailing-frame trim at the 86-unit frontier
evidence_used: The current 86-unit, 22-step model achieved 86.75% accuracy, leaving 1.75 points of margin; the analogous 97-unit trim from 22 to 21 steps retained 86.01% accuracy and reduced MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2444402303504125855, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 468061020, "recurrent_steps": 17115, "total_inference_macs": 468621740, "training_seconds": 136.71907783392817, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.45798510217959165}

RECENT RESULT
hypothesis: The qualified 86-unit GRU will retain at least 85% validation accuracy on frames 4–23 while reducing execution from 21 to 20 recurrent steps and total inference MACs to approximately 446 million.
change: Change the current model to an 86-unit GRU and classifier, and shorten the standard schedule to 20 contiguous frames while retaining batch size 16 and the established training procedure.
mechanism: Incremental trailing-frame trim at fixed recurrent width
evidence_used: The 86-unit, 21-step design achieved 85.15% accuracy at 468,621,740 MACs; analogously, trimming the 103-unit model from 21 to 20 steps reduced accuracy by only 0.12 percentage points and still qualified.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 131.9196069170721, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4744566279686302}

RECENT RESULT
hypothesis: An 87-unit GRU trained on frames 4–23 will recover at least one validation example over the 86-unit, 20-step design, reaching at least 85% accuracy while reducing total inference MACs from the qualified 468,621,740 to approximately 455,777,340.
change: Replace the 97-unit GRU and classifier with 87 units and shorten the standard schedule from 21 to 20 contiguous frames, retaining batch size 16 and the established training procedure.
mechanism: Minimal width rebound at the 20-step frontier
evidence_used: The 86-unit, 20-step design missed qualification by only one validation example at 84.91%, while the adjacent 86-unit, 21-step design reached 85.15%; adding one hidden unit is the smallest capacity increase available while preserving the cheaper 20-step execution.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2377403958605877233, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 29193, "peak_hidden_elements": 89600, "recurrent_macs": 455210100, "recurrent_steps": 16300, "total_inference_macs": 455777340, "training_seconds": 116.49121791590005, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.5012824590952119}

RECENT RESULT
hypothesis: An 86-unit GRU processing frames 5–24 will reach at least 85% validation accuracy while retaining the 20-step cost of approximately 446,333,120 MACs.
change: Use batch size 16, reduce the GRU to 86 units, and replace the unsuccessful frames 4–23 schedule with 20 contiguous frames shifted one step later.
mechanism: Late-context-preserving leading-frame trim
evidence_used: The 86-unit frames 4–24 model qualified at 85.15%, while removing frame 24 missed qualification by only one example; the failed 87-unit rebound suggests preserving the final frame is more informative than adding capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 109.89118841709569, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4928817468186829}

RECENT RESULT
hypothesis: Processing frames 4–22 and 24 with the qualified 86-unit GRU will preserve at least 85% accuracy while reducing inference to 20 recurrent steps and approximately 446,333,120 total MACs.
change: Keep both endpoints of the qualified frames 4–24 window while omitting the redundant penultimate frame, replacing the unsuccessful contiguous 20-frame schedules.
mechanism: Full-span late-frame substitution
evidence_used: Removing frame 24 missed qualification by only one validation example, while removing frame 4 performed worse; this suggests retaining the final frame and dropping an adjacent interior frame is the most targeted unresolved 20-step schedule.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 104.73653479106724, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.47286453715131327}



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

BATCH_SIZE = 16
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(91, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
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
        if available_frames <= 11:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))


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

BATCH_SIZE = 16
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)

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
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))


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

BATCH_SIZE = 16
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(86, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
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
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))


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

BATCH_SIZE = 32
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 103, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(103, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 103, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 103, device=device, dtype=dtype)
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
        if available_frames <= 12:
            return list(range(available_frames))
        return list(range(4, available_frames - 7))


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
