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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710729226804725177, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 518584500, "recurrent_steps": 20375, "total_inference_macs": 519679860, "training_seconds": 41.29506133310497, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4631549016098303}
prior_hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy on frames 4–28, reducing execution from 26 to 25 recurrent steps and total inference MACs from 540,423,240 to approximately 519,679,860.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2927130479794615167, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 560071260, "recurrent_steps": 22005, "total_inference_macs": 561166620, "training_seconds": 49.65840066596866, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4219981094079515}
prior_hypothesis: Retaining frame 28 while dropping frame 1 will recover at least 85% validation accuracy with 27 recurrent steps and approximately 561,166,620 total inference MACs.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2818929853299670172, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 539327880, "recurrent_steps": 21190, "total_inference_macs": 540423240, "training_seconds": 47.78405483299866, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.43240570817256996}
prior_hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy on frames 3–28, reducing execution from 27 to 26 recurrent steps and total inference MACs from 561,166,620 to approximately 540,423,240.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3306704161026971853, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 632827125, "recurrent_steps": 23635, "total_inference_macs": 633935525, "training_seconds": 58.149112458806485, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4017870850358273}
prior_hypothesis: An 85-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 647,491,420 to approximately 633,935,525.

## Recent verification evidence

RECENT RESULT
hypothesis: Averaging the mean recurrent output with the final hidden state before classification will retain at least 85% validation accuracy while halving classifier MACs, reducing total inference MACs from 608,609,400 to approximately 608,061,720.
change: Replace the 168-to-8 classifier over concatenated recurrent views with an 84-to-8 classifier over their elementwise average.
mechanism: Parameter-free fusion of temporal and final-state views
evidence_used: The current 84-unit, 18-band model already meets the threshold at 85.03%; unlike another spectral or recurrent-width reduction, this preserves both qualified recurrent representations and all 29 recurrent steps while removing only redundant classifier-side dimensionality.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3171742457162927479, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 26924, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608061720, "training_seconds": 43.2407687921077, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.45419027878462903}

RECENT RESULT
hypothesis: The qualified 84-unit, 18-band GRU with a linear 168→7→8 classifier will retain at least 85% validation accuracy while reducing total inference MACs from 608,609,400 to approximately 608,518,120.
change: Use the qualified lowest 18 mel bands and replace the 168→8 classifier with two linear layers having a seven-dimensional bottleneck and no intervening nonlinearity.
mechanism: Softmax-equivalent rank-7 classifier factorization
evidence_used: The 18-band trunk achieved 85.03% accuracy at 608,609,400 MACs. For eight-class softmax, seven independent logit contrasts suffice, so a rank-7 linear head preserves the necessary classifier output dimension while structurally saving 112 MACs per example; unlike the failed averaging head, it retains both recurrent views independently.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3174123109009208446, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27491, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608518120, "training_seconds": 47.02099200012162, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.47702409241097105}

RECENT RESULT
hypothesis: The qualified 84-unit, 18-band GRU with seven learned class-contrast logits and one fixed reference logit will retain at least 85% accuracy while reducing total inference MACs from 608,609,400 to approximately 608,472,480.
change: Adopt the qualified 18-band recurrent trunk and replace the 168→8 classifier with one 168→7 linear layer whose outputs are concatenated with a zero reference-class logit.
mechanism: Single-layer reference-class logit parameterization
evidence_used: The 84-unit, 18-band trunk achieved 85.03% accuracy. The failed 168→7→8 head used a non-convex two-matrix factorization; directly learning seven logits preserves the full eight-class softmax function class without that factorization and saves 168 MACs per example.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3173885043824580342, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27427, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608472480, "training_seconds": 42.87987033301033, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4565411245896041}

RECENT RESULT
hypothesis: An 84-unit GRU using the lowest 17 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 608,609,400 to approximately 602,653,380.
change: Adopt the qualified 84-unit architecture and structurally discard the three highest-frequency mel bands in both recurrent execution paths.
mechanism: Adjacent high-frequency edge-band pruning
evidence_used: The 84-unit, 19-band model achieved 86.75% accuracy and the 18-band model remained qualified at 85.03%; preserving recurrent width while removing the next edge band is the closest unresolved structural reduction and saves exactly 5,956,020 recurrent MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3143531732784505157, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 601558020, "recurrent_steps": 23635, "total_inference_macs": 602653380, "training_seconds": 37.495506417006254, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46444571091353526}

RECENT RESULT
hypothesis: An 84-unit GRU using mel bands 1–17 over the qualified 29-frame schedule will recover at least 85% validation accuracy while matching the 17-band model’s approximately 602,653,380 total inference MACs.
change: Reduce the recurrent input to 17 bands, but preserve band 17 by discarding the lowest-frequency band instead of the third-highest band.
mechanism: Opposite-edge spectral pruning
evidence_used: Bands 0–17 achieved 85.03%, while bands 0–16 narrowly missed at 84.79%; testing bands 1–17 distinguishes a dimensional-capacity limit from the loss of informative band 17.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3143531732784505157, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 601558020, "recurrent_steps": 23635, "total_inference_macs": 602653380, "training_seconds": 37.70312804193236, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4105138579760592}

RECENT RESULT
hypothesis: An 84-unit GRU using mel bands 2–17 over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 602,653,380 to approximately 596,697,360.
change: Structurally reduce the recurrent input from 19 to 16 bands, selecting bands 2–17 in both recurrent execution paths while preserving the qualified recurrent width, dual-view classifier, schedule, and training procedure.
mechanism: Opposite-edge spectral pruning preserving the informative upper band
evidence_used: The 17-band selection 1–17 achieved 86.26% accuracy, whereas 0–16 missed at 84.79%; this indicates retaining upper band 17 and pruning another low-frequency edge band is the best-supported next spectral reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3112464226190545683, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27088, "peak_hidden_elements": 86528, "recurrent_macs": 595602000, "recurrent_steps": 23635, "total_inference_macs": 596697360, "training_seconds": 39.87663216609508, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.44875549129181846}

RECENT RESULT
hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 29 to 28 causal steps, lowering total inference MACs from 602,653,380 to approximately 581,910,000.
change: Adopt the qualified 17-band input selection and omit one additional trailing frame while preserving recurrent width, dual-view classification, and training procedure.
mechanism: Adjacent temporal edge pruning with the qualified spectral trunk
evidence_used: The 17-band bands-1–17 design achieved 86.26% accuracy at 602,653,380 MACs, while reducing to 16 bands failed; preserving its spectral and recurrent capacity while testing the nearest temporal reduction isolates a new cost axis with substantially larger potential savings.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3035331106289560162, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 580814640, "recurrent_steps": 22820, "total_inference_macs": 581910000, "training_seconds": 40.22126704105176, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4087623455773102}

RECENT RESULT
hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 28 to 27 causal steps, lowering total inference MACs from 581,910,000 to approximately 561,166,620.
change: Adopt the qualified 84-unit, 17-band architecture and omit one additional trailing frame while preserving its dual-view classifier and training procedure.
mechanism: Adjacent trailing-frame pruning on the qualified spectral trunk
evidence_used: The 28-step, 84-unit, bands-1–17 design achieved 86.87% validation accuracy at 581,910,000 MACs, leaving a 1.87-point margin; the closest unresolved reduction is one trailing recurrent step, saving approximately 20,743,380 MACs without reducing spectral or hidden capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2927130479794615167, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 560071260, "recurrent_steps": 22005, "total_inference_macs": 561166620, "training_seconds": 49.81973470794037, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4304316140391344}

RECENT RESULT
hypothesis: Retaining frame 28 while dropping frame 1 will recover at least 85% validation accuracy with 27 recurrent steps and approximately 561,166,620 total inference MACs.
change: Change the schedule from frames 1–28 to frames 2–28, preserving the qualified recurrent architecture, spectral selection, and training procedure.
mechanism: Opposite-edge temporal pruning
evidence_used: The 28-step frames-1–28 model achieved 86.87% accuracy, while frames 1–27 narrowly missed at 84.91%; this tests whether losing the latest frame, rather than using only 27 steps, caused the regression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2927130479794615167, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 560071260, "recurrent_steps": 22005, "total_inference_macs": 561166620, "training_seconds": 49.65840066596866, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4219981094079515}

RECENT RESULT
hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy on frames 3–28, reducing execution from 27 to 26 recurrent steps and total inference MACs from 561,166,620 to approximately 540,423,240.
change: Preserve the qualified recurrent architecture and training procedure while removing the earliest frame from the successful 27-step frames-2–28 schedule.
mechanism: Adjacent leading-frame pruning
evidence_used: Frames 2–28 achieved 85.89% accuracy, whereas frames 1–27 missed at 84.91%, showing that the latest retained frame is more valuable than the earliest; pruning frame 2 is therefore the best-supported next temporal reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2818929853299670172, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 539327880, "recurrent_steps": 21190, "total_inference_macs": 540423240, "training_seconds": 47.78405483299866, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.43240570817256996}

RECENT RESULT
hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy on frames 4–28, reducing execution from 26 to 25 recurrent steps and total inference MACs from 540,423,240 to approximately 519,679,860.
change: Adopt the qualified 17-band recurrent input and remove the earliest frame from the successful 26-step frames-3–28 schedule, while preserving recurrent width, dual-view classification, and training procedure.
mechanism: Adjacent leading-frame pruning on the qualified spectral trunk
evidence_used: Frames 3–28 with bands 1–17 achieved 85.89% accuracy at 540,423,240 MACs; the preceding successful transition from frames 2–28 to frames 3–28 preserved identical accuracy, making another leading-frame removal the closest supported reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710729226804725177, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 518584500, "recurrent_steps": 20375, "total_inference_macs": 519679860, "training_seconds": 41.29506133310497, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4631549016098303}

RECENT RESULT
hypothesis: An 85-unit GRU using mel bands 1–17 on frames 5–28 will retain at least 85% validation accuracy while reducing total inference MACs from 519,679,860 to approximately 509,864,000.
change: Adopt the qualified spectral trunk, remove the earliest frame from its 25-step schedule, and increase hidden width from 84 to 85 to compensate for the lost temporal evidence.
mechanism: One-step-to-one-unit recurrent budget reallocation
evidence_used: The 84-unit, 17-band model met the threshold on frames 4–28, and prior temporal comparisons showed retaining later frames was preferable to retaining earlier ones. Adding one hidden unit while removing one step still saves approximately 9.8 million MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2659528207372151482, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 27922, "peak_hidden_elements": 87552, "recurrent_macs": 508755600, "recurrent_steps": 19560, "total_inference_macs": 509864000, "training_seconds": 34.687475374899805, "validation_accuracy": 0.8233128834355828, "validation_cross_entropy": 0.5030568877612155}



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
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
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
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
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
        return list(range(4, available_frames - 3))


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
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
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
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
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
        return list(range(2, available_frames - 3))


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
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
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
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
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
        return list(range(3, available_frames - 3))


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
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
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
        return list(range(1, available_frames - 2))


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
