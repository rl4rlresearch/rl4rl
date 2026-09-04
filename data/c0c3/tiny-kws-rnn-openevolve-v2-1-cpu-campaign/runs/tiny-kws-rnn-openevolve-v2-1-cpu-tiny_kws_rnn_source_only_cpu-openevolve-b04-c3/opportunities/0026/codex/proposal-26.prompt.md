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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5883088100548758405, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 55845, "peak_hidden_elements": 124416, "recurrent_macs": 1126281915, "recurrent_steps": 22005, "total_inference_macs": 1127859755, "training_seconds": 89.98986204201356, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.40845211473710696}
prior_hypothesis: A 121-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 122-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6065037920228762065, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57505, "peak_hidden_elements": 126464, "recurrent_macs": 1161137835, "recurrent_steps": 22005, "total_inference_macs": 1162741755, "training_seconds": 102.38246341701597, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3980303430849789}
prior_hypothesis: A 123-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 124-unit design.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5973718666103851817, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 56672, "peak_hidden_elements": 125440, "recurrent_macs": 1143643860, "recurrent_steps": 22005, "total_inference_macs": 1145234740, "training_seconds": 102.06052891700529, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4081423332354774}
prior_hypothesis: A 122-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 123-unit design’s 1.163B.

## Recent verification evidence

RECENT RESULT
hypothesis: A 127-unit GRU using a directly generated 27-frame uniform grid will achieve at least 85% validation accuracy while reducing recurrent steps and exact inference MACs below every qualified design.
change: Reduce the recurrent and classifier width from 128 to 127, and replace the 28-frame grid with a uniformly distributed 27-frame schedule spanning both endpoints.
mechanism: Re-optimized 27-frame uniform striding with boundary-width GRU
evidence_used: The 127-unit/28-step model qualified at 85.28%, while 127-unit/27-step schedules formed by deleting frames from the 28-frame grid missed by as little as one validation example; regenerating the grid for exactly 27 steps tests whether balanced temporal allocation recovers that narrow margin.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6432882639791173913, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1232434035, "recurrent_steps": 22005, "total_inference_macs": 1233262075, "training_seconds": 109.30428883293644, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4708150758333733}

RECENT RESULT
hypothesis: A 127-unit GRU dropping the earliest frame and reducing label smoothing from 0.03 to 0.01 will recover the single validation example needed for at least 85% accuracy while lowering inference to approximately 1.233B MACs and 27 steps.
change: Remove frame 0 from the qualified 28-frame schedule and use less-smoothed cross-entropy to strengthen narrow decision margins without changing inference cost.
mechanism: Sharpened-loss early-boundary frame thinning
evidence_used: The prior 127-unit variant dropping frame 0 achieved 84.91%, only one validation example below qualification and better than the other tested 27-frame schedules; the qualified 127-unit/28-step model shows sufficient capacity, motivating a training-only margin adjustment at the lower-cost boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6432882639791173913, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1232434035, "recurrent_steps": 22005, "total_inference_macs": 1233262075, "training_seconds": 103.82524874992669, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.44297885075668614}

RECENT RESULT
hypothesis: A 127-unit GRU using the best-performing 27-frame schedule will exceed 85% validation accuracy when its classifier jointly uses the mean recurrent output and terminal hidden state, while remaining below the qualified 128-unit, 27-step design in exact MACs.
change: Reduce recurrent width to 127, remove the earliest frame from the 28-frame grid, and concatenate mean-pooled and terminal recurrent features for a 254-to-8 classifier.
mechanism: Learned mean-and-terminal dual recurrent readout
evidence_used: The 127-unit model dropping frame 0 reached 84.91%, only one validation example below qualification; the added terminal-state readout restores sequence-order information discarded by mean pooling at a classifier cost much smaller than the recurrent MAC savings versus 128 units.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6437201822426569369, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58849, "peak_hidden_elements": 130560, "recurrent_macs": 1232434035, "recurrent_steps": 22005, "total_inference_macs": 1234090115, "training_seconds": 108.26358229201287, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4089824840335027}

RECENT RESULT
hypothesis: A 126-unit GRU retaining the qualified 27-frame schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing total inference MACs from 1.234B to approximately 1.216B.
change: Reduce the GRU state, temporal summary, and concatenated classifier input by one unit while preserving training and temporal coverage.
mechanism: Dual-readout recurrent-width boundary search
evidence_used: Adding terminal-state features raised the same 127-unit, 27-step schedule from 84.91% to 85.40%; the observed 128-to-127 width reduction at 28 steps cost only about 0.25 percentage points, suggesting the dual readout has enough margin for the next width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6343127814022391793, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58008, "peak_hidden_elements": 129536, "recurrent_macs": 1214411940, "recurrent_steps": 22005, "total_inference_macs": 1216054980, "training_seconds": 107.13583445805125, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.38781575396017065}

RECENT RESULT
hypothesis: A 125-unit GRU retaining the qualified 27-frame early-drop schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing exact inference MACs below the qualified 126-unit design.
change: Reduce recurrent width from 128 to 125, classify from concatenated mean and terminal states, and switch to the qualified schedule that removes frame 0.
mechanism: Dual-readout recurrent-width boundary search
evidence_used: The 126-unit version of this exact dual-readout, 27-step design achieved 86.50% accuracy at 1.216B MACs, leaving 1.50 percentage points of qualification margin and motivating the adjacent lower-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6249742494188031053, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57173, "peak_hidden_elements": 128512, "recurrent_macs": 1196521875, "recurrent_steps": 22005, "total_inference_macs": 1198151875, "training_seconds": 93.77462554210797, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.38509455955833016}

RECENT RESULT
hypothesis: A 124-unit GRU retaining the qualified 27-frame early-drop schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing total inference MACs from 1.198B to approximately 1.180B.
change: Reduce recurrent width to 124, classify from concatenated mean and terminal states, and use the qualified schedule that removes frame 0.
mechanism: Adjacent dual-readout recurrent-width reduction
evidence_used: The otherwise identical 125-unit design achieved 85.52% accuracy at 1.198B MACs; testing the adjacent width is the most direct structural boundary search below the best qualified design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6157045862923487149, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 56344, "peak_hidden_elements": 127488, "recurrent_macs": 1178763840, "recurrent_steps": 22005, "total_inference_macs": 1180380800, "training_seconds": 83.91581020806916, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.38839361272706574}

RECENT RESULT
hypothesis: Independently supervised mean-state and terminal-state predictors can safely exit highly confident, agreeing examples after 23 steps, retaining at least 85% accuracy while reducing exact inference MACs below 1,198,151,875.
change: Replace the inseparable concatenated readout with independently supervised temporal-mean and terminal heads, add a cheap early probe, and stop examples up to four frames early only when both full heads confidently agree.
mechanism: Dual-view consensus-gated anytime inference
evidence_used: The 125-unit dual-readout model qualified at 85.52%, and adding terminal information previously lifted the 127-unit/27-step design from 84.91% to 85.40%. This challenges the shared fixed-compute assumption while preserving the qualified recurrent path and making its two predictive views usable as a conservative exit consensus.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 124-unit GRU augmented with mean-times-terminal features will recover at least two validation examples, reaching 85% accuracy while keeping total inference below the qualified 125-unit model’s 1.198B MACs.
change: Reduce recurrent width from 126 to 124 and extend the classifier with a parameter-free multiplicative interaction between mean and terminal recurrent states.
mechanism: Elementwise mean–terminal interaction readout
evidence_used: The otherwise identical 124-unit dual-readout model missed qualification by only two examples at 84.79% with strong 0.3884 cross-entropy; adding nonlinear temporal interaction costs only 992 additional classifier MACs per example while preserving most of its recurrent savings.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6161263017622613421, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57336, "peak_hidden_elements": 127488, "recurrent_macs": 1178763840, "recurrent_steps": 22005, "total_inference_macs": 1181189280, "training_seconds": 59.9718262499664, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4170377017530196}

RECENT RESULT
hypothesis: Independently supervising the mean and terminal recurrent views of a 124-unit, 27-step GRU will recover the two validation examples needed for at least 85% accuracy while retaining the approximately 1.180B inference-MAC cost of the prior 124-unit dual-readout design.
change: Use the lower-cost 124-unit GRU and qualified early-drop schedule, classify from concatenated mean and terminal states, and add two auxiliary classifiers that execute only during training.
mechanism: Training-only dual-view deep supervision
evidence_used: The prior 124-unit dual-readout model missed qualification by only two examples with 0.3884 cross-entropy, while adding terminal information previously raised the 127-unit design from 84.91% to 85.40%; direct training supervision for both predictive views targets that narrow optimization gap without adding validation MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6157045862923489149, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58344, "peak_hidden_elements": 127488, "recurrent_macs": 1178763840, "recurrent_steps": 22005, "total_inference_macs": 1180380800, "training_seconds": 81.08205604203977, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.3916617364239839}

RECENT RESULT
hypothesis: A 123-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 124-unit design.
change: Reduce the recurrent state and both readout widths from 124 to 123 units, preserving the schedule, dual readout, auxiliary losses, and all training settings.
mechanism: Deep-supervised recurrent-width boundary search
evidence_used: Training-only dual-view supervision raised the 124-unit design from 84.79% to 86.38% without adding inference MACs, leaving 1.38 percentage points of margin and motivating the adjacent structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6065037920228762065, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 57505, "peak_hidden_elements": 126464, "recurrent_macs": 1161137835, "recurrent_steps": 22005, "total_inference_macs": 1162741755, "training_seconds": 102.38246341701597, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3980303430849789}

RECENT RESULT
hypothesis: A 122-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 123-unit design’s 1.163B.
change: Reduce recurrent and readout width to 122 units and add the qualified training-only auxiliary mean-state and terminal-state classifiers.
mechanism: Deep-supervised adjacent-width reduction
evidence_used: The otherwise identical deep-supervised 123-unit design achieved 85.77% accuracy at 1.163B MACs, leaving 0.77 percentage points of margin for an adjacent structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5973718666103851817, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 56672, "peak_hidden_elements": 125440, "recurrent_macs": 1143643860, "recurrent_steps": 22005, "total_inference_macs": 1145234740, "training_seconds": 102.06052891700529, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4081423332354774}

RECENT RESULT
hypothesis: A 121-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 122-unit design.
change: Reduce recurrent and readout width to 121 units and add auxiliary mean-state and terminal-state classifiers used only during training.
mechanism: Deep-supervised adjacent-width reduction
evidence_used: The otherwise identical deep-supervised 122-unit design achieved 85.77% accuracy at 1.145B MACs, matching the 123-unit design’s accuracy while using fewer MACs; this supports testing the adjacent lower-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5883088100548758405, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 55845, "peak_hidden_elements": 124416, "recurrent_macs": 1126281915, "recurrent_steps": 22005, "total_inference_macs": 1127859755, "training_seconds": 89.98986204201356, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.40845211473710696}



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
        self.gru = nn.GRU(20, 121, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(242, 8)
        self.mean_aux = nn.Linear(121, 8)
        self.terminal_aux = nn.Linear(121, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 121, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
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
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(246, 8)
        self.mean_aux = nn.Linear(123, 8)
        self.terminal_aux = nn.Linear(123, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
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
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(244, 8)
        self.mean_aux = nn.Linear(122, 8)
        self.terminal_aux = nn.Linear(122, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
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
