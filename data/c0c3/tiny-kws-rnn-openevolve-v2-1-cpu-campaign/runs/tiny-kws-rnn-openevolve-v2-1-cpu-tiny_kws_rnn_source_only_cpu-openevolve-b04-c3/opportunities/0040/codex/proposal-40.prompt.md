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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5578887557316745014, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 55024, "peak_hidden_elements": 123392, "recurrent_macs": 1067976000, "recurrent_steps": 21190, "total_inference_macs": 1069540800, "training_seconds": 78.03866162500344, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.41509296440639376}
prior_hypothesis: The qualified 120-unit GRU will retain at least 85% validation accuracy after dropping the next-earliest scheduled frame, while reducing total inference MACs from 1.111B to approximately 1.070B.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4477257919862028873, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 51568, "peak_hidden_elements": 139776, "recurrent_macs": 856571520, "recurrent_steps": 18745, "total_inference_macs": 858344960, "training_seconds": 41.5718735831324, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4289872620003355}
prior_hypothesis: A 48/88 hierarchical GRU using all 23 qualified frames will retain at least 85% validation accuracy while reducing total inference MACs from 871,010,875 to approximately 858,344,960.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4411777240597283517, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 50857, "peak_hidden_elements": 138752, "recurrent_macs": 844031115, "recurrent_steps": 18745, "total_inference_macs": 845791515, "training_seconds": 31.589560042135417, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4211810901852473}
prior_hypothesis: A 48/87 hierarchical GRU using all 23 qualified frames will retain at least 85% validation accuracy while reducing total inference MACs from 858,344,960 to approximately 845,791,515.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4543325259760321905, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 52285, "peak_hidden_elements": 140800, "recurrent_macs": 869224395, "recurrent_steps": 18745, "total_inference_macs": 871010875, "training_seconds": 33.55239391699433, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.44666392320504217}
prior_hypothesis: Reducing the contextual GRU from 90 to 89 units while preserving all 23 qualified frames will retain at least 85% validation accuracy and lower total inference MACs from 883,789,260 to approximately 871,010,875.

## Recent verification evidence

RECENT RESULT
hypothesis: A 119-unit GRU using the qualified 26-frame schedule and unchanged dual-view supervision will retain at least 85% validation accuracy while reducing total inference MACs from 1,069,540,800 to approximately 1,053,063,130.
change: Reconstruct the best 26-step design while reducing the GRU state and all associated readout dimensions from 120 to 119 units.
mechanism: Deep-supervised adjacent-width reduction at 26 recurrent steps
evidence_used: The 120-unit 26-step design qualified at 85.28%; at 27 steps, every adjacent reduction from 123 through 120 units qualified, with accuracy improving from 85.77% to 86.13%, supporting one adjacent width test on the lower-step design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5492937523337999329, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 54209, "peak_hidden_elements": 122368, "recurrent_macs": 1051511370, "recurrent_steps": 21190, "total_inference_macs": 1053063130, "training_seconds": 83.15202737506479, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4237176719618721}

RECENT RESULT
hypothesis: A 119-unit GRU whose independently supervised mean and terminal heads are also used for inference will recover the failed 119-unit design’s 0.34-point accuracy deficit, reaching at least 85% while retaining approximately 1.053B inference MACs and reducing learned parameters.
change: Reduce the recurrent width to 119, remove the redundant concatenated classifier, and average the separately supervised mean-state and terminal-state logits for the final prediction.
mechanism: Inference-head-tied dual-view supervision
evidence_used: The prior 119-unit design missed qualification narrowly at 84.66%, while training-only dual-view supervision raised the 124-unit design from 84.79% to 86.38%; tying those supervised views directly to inference targets the optimization gap without increasing inference MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5492937523337997417, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 52297, "peak_hidden_elements": 122368, "recurrent_macs": 1051511370, "recurrent_steps": 21190, "total_inference_macs": 1053063130, "training_seconds": 53.00572799984366, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.4435740933096482}

RECENT RESULT
hypothesis: A stacked 48-unit acoustic GRU and 90-unit contextual GRU, using all 27 qualified frames and both levels’ mean and terminal states, will retain at least 85% accuracy while reducing total inference MACs below the qualified 120-unit/26-step model’s 1.070B.
change: Replace the monolithic 120-unit recurrence with two narrower recurrent stages, normalize their interface and readouts, and classify from fine- and context-scale temporal summaries while retaining training-only dual-view supervision.
mechanism: Hierarchical recurrent factorization with multilevel temporal readout
evidence_used: The 120-unit model qualified at 86.13% with 27 steps, but reducing either temporal coverage or width approached the accuracy boundary. This suggests monolithic width and frame deletion are load-bearing assumptions. The factorized recurrence preserves 27-frame coverage and increases total recurrent state from 120 to 138 units while reducing structural recurrent cost from 50,400 to 47,052 MACs per frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5410082336840918593, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 1035379260, "recurrent_steps": 22005, "total_inference_macs": 1037178780, "training_seconds": 31.742179125081748, "validation_accuracy": 0.8834355828220859, "validation_cross_entropy": 0.3520360934953748}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% accuracy after dropping the next-earliest scheduled frame, while reducing total inference MACs from 1.037B to approximately 999M.
change: Replace the monolithic 123-unit GRU with the qualified acoustic/contextual hierarchy and reduce its schedule from 27 to 26 recurrent steps.
mechanism: Hierarchical recurrent factorization with late-biased temporal thinning
evidence_used: Reference Design 2 achieved 88.34% accuracy at 27 steps, leaving 3.34 points of margin; the monolithic 120-unit model retained 85.28% under the same adjacent 27-to-26-step early-frame thinning.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5210056567703729598, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 997031880, "recurrent_steps": 21190, "total_inference_macs": 998831400, "training_seconds": 47.61131420801394, "validation_accuracy": 0.8822085889570552, "validation_cross_entropy": 0.3578270192526601}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional early frame, while reducing total inference MACs from 998,831,400 to approximately 960,484,020.
change: Replace the monolithic GRU with the qualified hierarchical dual-readout model, retain auxiliary supervision, and reduce its schedule from 26 to 25 recurrent steps.
mechanism: Hierarchical recurrent factorization with 25-step late-biased temporal thinning
evidence_used: The identical 48/90 hierarchy achieved 88.22% accuracy at 26 steps—3.22 percentage points above the threshold—and previously lost only 0.12 points when reduced from 27 to 26 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5010030798566540603, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 958684500, "recurrent_steps": 20375, "total_inference_macs": 960484020, "training_seconds": 44.086129875155166, "validation_accuracy": 0.8846625766871166, "validation_cross_entropy": 0.35723108396939707}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 960,484,020 to approximately 922,136,640.
change: Reduce the current schedule from 25 to 24 recurrent steps by dropping the next-earliest frame from the original 28-frame grid.
mechanism: Hierarchical recurrence with 24-step late-biased temporal thinning
evidence_used: The identical hierarchy achieved 88.22% at 26 steps and 88.47% at 25 steps, so the latest thinning preserved accuracy and leaves 3.47 percentage points of qualification margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4810005029429351608, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 920337120, "recurrent_steps": 19560, "total_inference_macs": 922136640, "training_seconds": 43.2259170000907, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.39977568643956096}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 922,136,640 to approximately 883,789,260.
change: Reduce the schedule to 23 recurrent steps by dropping the five earliest frames from the original 28-frame grid, preserving architecture and training procedure.
mechanism: Hierarchical recurrence with 23-step late-biased temporal thinning
evidence_used: The identical hierarchy qualified at 86.26% with 24 steps and 922,136,640 MACs; testing the adjacent 23-step schedule directly locates the temporal-compression boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4609979260292162613, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 881989740, "recurrent_steps": 18745, "total_inference_macs": 883789260, "training_seconds": 40.58797562494874, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43369420525486485}

RECENT RESULT
hypothesis: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional earliest frame, while reducing total inference MACs from 883,789,260 to approximately 845,441,880.
change: Reduce the schedule from 27 to 22 recurrent steps by dropping the six earliest frames from the original 28-frame grid, preserving the architecture and training procedure.
mechanism: Hierarchical recurrence with 22-step late-biased temporal thinning
evidence_used: The identical hierarchy qualified at 85.77% with 23 steps and 883,789,260 MACs; testing the adjacent 22-step schedule directly locates the remaining temporal-compression boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4409953491154973618, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 843642360, "recurrent_steps": 17930, "total_inference_macs": 845441880, "training_seconds": 29.59954662481323, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.46877828960769746}

RECENT RESULT
hypothesis: Retaining frame 6 while removing adjacent frame 7 will recover the failed 22-step schedule to at least 85% accuracy, while preserving its 845,441,880 total inference MACs.
change: Replace the monolithic GRU with the qualified 48/90 hierarchical model and use 22 frames by retaining the earliest qualified frame and deleting its immediate successor.
mechanism: Onset-anchored 22-step hierarchical recurrence
evidence_used: The hierarchy achieved 85.77% with frames 6–31 over 23 steps, but fell to 83.68% when frame 6 was removed for 22 steps; this directly motivates preserving frame 6 and thinning adjacent temporal redundancy instead.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4409953491154973618, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 53008, "peak_hidden_elements": 141824, "recurrent_macs": 843642360, "recurrent_steps": 17930, "total_inference_macs": 845441880, "training_seconds": 35.30572137515992, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.45439057145382}

RECENT RESULT
hypothesis: Reducing the contextual GRU from 90 to 89 units while preserving all 23 qualified frames will retain at least 85% validation accuracy and lower total inference MACs from 883,789,260 to approximately 871,010,875.
change: Keep the qualified hierarchical architecture, schedule, and training procedure unchanged while shrinking the contextual state and corresponding normalization, readout, and auxiliary-head dimensions by one unit.
mechanism: Adjacent contextual-state width reduction
evidence_used: The current 48/90 hierarchy qualifies at 85.77% with 23 steps, whereas both tested 22-step schedules failed; prior adjacent-width experiments frequently qualified when sufficient margin remained, motivating structural width reduction without further temporal information loss.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4543325259760321905, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 52285, "peak_hidden_elements": 140800, "recurrent_macs": 869224395, "recurrent_steps": 18745, "total_inference_macs": 871010875, "training_seconds": 33.55239391699433, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.44666392320504217}

RECENT RESULT
hypothesis: A 48/88 hierarchical GRU using all 23 qualified frames will retain at least 85% validation accuracy while reducing total inference MACs from 871,010,875 to approximately 858,344,960.
change: Reconstruct the qualified 23-step hierarchy and shrink its contextual GRU, combined normalizations, classifier, and auxiliary heads from 89 to 88 contextual units.
mechanism: Adjacent contextual-state width reduction
evidence_used: The otherwise identical 48/89 model qualified at 85.28% and 871,010,875 MACs, while both 22-step variants failed; this makes an adjacent width reduction the most direct remaining structural test without sacrificing temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4477257919862028873, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 51568, "peak_hidden_elements": 139776, "recurrent_macs": 856571520, "recurrent_steps": 18745, "total_inference_macs": 858344960, "training_seconds": 41.5718735831324, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4289872620003355}

RECENT RESULT
hypothesis: A 48/87 hierarchical GRU using all 23 qualified frames will retain at least 85% validation accuracy while reducing total inference MACs from 858,344,960 to approximately 845,791,515.
change: Reconstruct the qualified 23-step hierarchy and shrink its contextual GRU, combined normalizations, classifier, auxiliary heads, and state tensors from 88 to 87 contextual units.
mechanism: Adjacent contextual-state width reduction
evidence_used: The otherwise identical 48/88 model qualified at 85.28% with 858,344,960 MACs, while both 22-step variants failed; an adjacent contextual-width reduction is the most direct remaining structural test without sacrificing temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4411777240597283517, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 50857, "peak_hidden_elements": 138752, "recurrent_macs": 844031115, "recurrent_steps": 18745, "total_inference_macs": 845791515, "training_seconds": 31.589560042135417, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4211810901852473}

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
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
        self.mean_aux = nn.Linear(120, 8)
        self.terminal_aux = nn.Linear(120, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
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
        mean = summary / count.clamp_min(1.0)
        terminal = hidden[:, 0, :]
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
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
    del step, total_steps
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2


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
    """A hierarchical causal GRU with fine and contextual temporal state."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.acoustic_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.context_norm = nn.LayerNorm(48)
        self.context_gru = nn.GRU(48, 88, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(136)
        self.terminal_norm = nn.LayerNorm(136)
        self.classifier = nn.Linear(272, 8)
        self.mean_aux = nn.Linear(136, 8)
        self.terminal_aux = nn.Linear(136, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        acoustic_hidden = torch.zeros(
            batch_size, 1, 48, device=device, dtype=dtype
        )
        context_hidden = torch.zeros(
            batch_size, 1, 88, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 88, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_output, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frame).unsqueeze(1),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_output, context_hidden = self.context_gru(
            self.context_norm(acoustic_output),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_output[:, 0, :],
            context_summary + context_output[:, 0, :],
            count + 1.0,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_outputs, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frames),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_outputs, context_hidden = self.context_gru(
            self.context_norm(acoustic_outputs),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_outputs.sum(dim=1),
            context_summary + context_outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        divisor = count.clamp_min(1.0)
        mean = self.mean_norm(
            torch.cat(
                (
                    acoustic_summary / divisor,
                    context_summary / divisor,
                ),
                dim=-1,
            )
        )
        terminal = self.terminal_norm(
            torch.cat(
                (
                    acoustic_hidden[:, 0, :],
                    context_hidden[:, 0, :],
                ),
                dim=-1,
            )
        )
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
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
    del step, total_steps
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2


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
    """A hierarchical causal GRU with fine and contextual temporal state."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.acoustic_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.context_norm = nn.LayerNorm(48)
        self.context_gru = nn.GRU(48, 87, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(135)
        self.terminal_norm = nn.LayerNorm(135)
        self.classifier = nn.Linear(270, 8)
        self.mean_aux = nn.Linear(135, 8)
        self.terminal_aux = nn.Linear(135, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        acoustic_hidden = torch.zeros(
            batch_size, 1, 48, device=device, dtype=dtype
        )
        context_hidden = torch.zeros(
            batch_size, 1, 87, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 87, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_output, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frame).unsqueeze(1),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_output, context_hidden = self.context_gru(
            self.context_norm(acoustic_output),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_output[:, 0, :],
            context_summary + context_output[:, 0, :],
            count + 1.0,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_outputs, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frames),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_outputs, context_hidden = self.context_gru(
            self.context_norm(acoustic_outputs),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_outputs.sum(dim=1),
            context_summary + context_outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        divisor = count.clamp_min(1.0)
        mean = self.mean_norm(
            torch.cat(
                (
                    acoustic_summary / divisor,
                    context_summary / divisor,
                ),
                dim=-1,
            )
        )
        terminal = self.terminal_norm(
            torch.cat(
                (
                    acoustic_hidden[:, 0, :],
                    context_hidden[:, 0, :],
                ),
                dim=-1,
            )
        )
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
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
    del step, total_steps
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2


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
    """A hierarchical causal GRU with fine and contextual temporal state."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.acoustic_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.context_norm = nn.LayerNorm(48)
        self.context_gru = nn.GRU(48, 89, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(137)
        self.terminal_norm = nn.LayerNorm(137)
        self.classifier = nn.Linear(274, 8)
        self.mean_aux = nn.Linear(137, 8)
        self.terminal_aux = nn.Linear(137, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        acoustic_hidden = torch.zeros(
            batch_size, 1, 48, device=device, dtype=dtype
        )
        context_hidden = torch.zeros(
            batch_size, 1, 89, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 89, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_output, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frame).unsqueeze(1),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_output, context_hidden = self.context_gru(
            self.context_norm(acoustic_output),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_output[:, 0, :],
            context_summary + context_output[:, 0, :],
            count + 1.0,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_outputs, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frames),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_outputs, context_hidden = self.context_gru(
            self.context_norm(acoustic_outputs),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_outputs.sum(dim=1),
            context_summary + context_outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        divisor = count.clamp_min(1.0)
        mean = self.mean_norm(
            torch.cat(
                (
                    acoustic_summary / divisor,
                    context_summary / divisor,
                ),
                dim=-1,
            )
        )
        terminal = self.terminal_norm(
            torch.cat(
                (
                    acoustic_hidden[:, 0, :],
                    context_hidden[:, 0, :],
                ),
                dim=-1,
            )
        )
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
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
    del step, total_steps
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2


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
