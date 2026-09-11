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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2449078584812962760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20040, "peak_hidden_elements": 104960, "recurrent_macs": 468188160, "recurrent_steps": 26080, "total_inference_macs": 469518240, "training_seconds": 51.24883804190904, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3946163060474981}
prior_hypothesis: A 68-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.6% versus the verified 69-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 79-unit GRU with the proven mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by about 2.2% versus the verified 80-unit model.
change: Reduce the GRU and recurrent summary widths from 80 to 79 units and the classifier input from 240 to 237 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 80-unit model achieved 85.64% accuracy, while the 82-unit model achieved 86.75%; a one-unit reduction is the smallest structural probe of the accuracy-cost boundary after larger two-unit reductions showed measurable accuracy sensitivity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3199902167817717041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25881, "peak_hidden_elements": 121856, "recurrent_macs": 611915040, "recurrent_steps": 26080, "total_inference_macs": 613460280, "training_seconds": 102.87217433284968, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3965815725502061}

RECENT RESULT
hypothesis: A 78-unit GRU with the proven mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by roughly 2.3% versus the verified 79-unit model.
change: Reduce the GRU and recurrent summaries from 79 to 78 units and the classifier input from 237 to 234 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 79-unit model achieved 87.24% accuracy, 2.24 points above the requirement, and outperformed the qualifying 80-unit model; this supports another minimal one-unit structural reduction at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3127564361002882040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25320, "peak_hidden_elements": 120320, "recurrent_macs": 598066560, "recurrent_steps": 26080, "total_inference_macs": 599592240, "training_seconds": 85.23865499999374, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4022246752779908}

RECENT RESULT
hypothesis: A 77-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.3% versus the verified 78-unit model.
change: Reduce the GRU and recurrent summary widths from 78 to 77 units and the classifier input from 234 to 231 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 78-unit model achieved 86.13% accuracy, 1.13 points above the requirement, after both 79- and 78-unit one-unit reductions qualified; this motivates one more minimal structural probe of the accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3056042777678200325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24765, "peak_hidden_elements": 118784, "recurrent_macs": 584374560, "recurrent_steps": 26080, "total_inference_macs": 585880680, "training_seconds": 83.19529145793058, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3903258902894939}

RECENT RESULT
hypothesis: A 76-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.3% versus the verified 77-unit model.
change: Reduce the GRU and recurrent summaries from 77 to 76 units and the classifier input from 231 to 228 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 77-unit model achieved 86.87% accuracy, 1.87 points above the requirement, after the 78- and 79-unit models also qualified; this supports another minimal structural reduction at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2985337417843671896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 570839040, "recurrent_steps": 26080, "total_inference_macs": 572325600, "training_seconds": 82.21133245783858, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.37948165612718077}

RECENT RESULT
hypothesis: A 75-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.3% versus the verified 76-unit model.
change: Reduce the GRU and recurrent summary widths from 76 to 75 units and the classifier input from 228 to 225 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 76-unit model achieved 87.36% validation accuracy, 2.36 points above the requirement, after the 77-, 78-, and 79-unit models also qualified; this supports another minimal structural reduction at the current cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2915448281499296753, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23673, "peak_hidden_elements": 115712, "recurrent_macs": 557460000, "recurrent_steps": 26080, "total_inference_macs": 558927000, "training_seconds": 85.2473029170651, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4085738012395753}

RECENT RESULT
hypothesis: A 74-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.4% versus the verified 75-unit model.
change: Reduce the GRU and all temporal summaries from 75 to 74 units and reduce the classifier input from 225 to 222 features, while preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 75-unit model achieved 86.50% validation accuracy, and every one-unit reduction from 79 through 75 units qualified; this supports one more minimal structural probe of the accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2846375368645074896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23136, "peak_hidden_elements": 114176, "recurrent_macs": 544237440, "recurrent_steps": 26080, "total_inference_macs": 545684880, "training_seconds": 63.53336408408359, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4089873705904908}

RECENT RESULT
hypothesis: A 73-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.4% versus the verified 74-unit model.
change: Reduce the GRU and all temporal summaries from 74 to 73 units and reduce the classifier input from 222 to 219 features, preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 74-unit model achieved 86.13% validation accuracy, and every one-unit reduction from 79 through 74 units met the requirement, motivating another minimal structural probe of the accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2778118679281006325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22605, "peak_hidden_elements": 112640, "recurrent_macs": 531171360, "recurrent_steps": 26080, "total_inference_macs": 532599240, "training_seconds": 65.29236908303574, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.41808663584703315}

RECENT RESULT
hypothesis: A 72-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact recurrent MACs by approximately 2.4% versus the verified 73-unit model.
change: Reduce the GRU and all temporal summaries from 73 to 72 units and reduce the classifier input from 219 to 216 features, preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 73-unit model achieved 87.36% validation accuracy, 2.36 points above the requirement, and every one-unit reduction from 79 through 73 units qualified; this supports another minimal structural probe of the accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2710678213407091040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 22080, "peak_hidden_elements": 111104, "recurrent_macs": 518261760, "recurrent_steps": 26080, "total_inference_macs": 519670080, "training_seconds": 69.89805737510324, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.40923808279213}

RECENT RESULT
hypothesis: A 71-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact recurrent MACs by approximately 2.5% versus the verified 72-unit model.
change: Reduce the GRU and all temporal summaries from 72 to 71 units and reduce the classifier input from 216 to 213 features, preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 72-unit model achieved 86.63% validation accuracy, 1.63 points above the requirement, and every one-unit reduction from 79 through 72 units qualified; this supports another minimal structural probe of the accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2644053971023329041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21561, "peak_hidden_elements": 109568, "recurrent_macs": 505508640, "recurrent_steps": 26080, "total_inference_macs": 506897400, "training_seconds": 62.884207624942064, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.400436055002037}

RECENT RESULT
hypothesis: A 70-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.5% versus the verified 71-unit model.
change: Reduce the GRU and all recurrent summary widths from 71 to 70 units and reduce the classifier input from 213 to 210 features, preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 71-unit model achieved 86.38% validation accuracy, 1.38 points above the requirement, and every one-unit reduction from 79 through 71 units qualified; this motivates the smallest structural probe at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2578245952129720328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 21048, "peak_hidden_elements": 108032, "recurrent_macs": 492912000, "recurrent_steps": 26080, "total_inference_macs": 494281200, "training_seconds": 68.49435162497684, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.39858897390541126}

RECENT RESULT
hypothesis: A 69-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.5% versus the verified 70-unit model.
change: Reduce the GRU and all recurrent summary widths from 70 to 69 units and reduce the classifier input from 210 to 207 features, preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 70-unit model achieved 87.12% validation accuracy, 2.12 points above the requirement, and every one-unit reduction from 79 through 70 units qualified; this supports another minimal structural probe at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2513254156726264901, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20541, "peak_hidden_elements": 106496, "recurrent_macs": 480471840, "recurrent_steps": 26080, "total_inference_macs": 481821480, "training_seconds": 60.21748624998145, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.3980777951105972}

RECENT RESULT
hypothesis: A 68-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.6% versus the verified 69-unit model.
change: Reduce the GRU and all recurrent summary widths from 69 to 68 units and reduce the classifier input from 207 to 204 features, preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 69-unit model achieved 87.48% validation accuracy, 2.48 points above the requirement, and every one-unit reduction from 79 through 69 units qualified; this supports another minimal structural probe at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2449078584812962760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 20040, "peak_hidden_elements": 104960, "recurrent_macs": 468188160, "recurrent_steps": 26080, "total_inference_macs": 469518240, "training_seconds": 51.24883804190904, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3946163060474981}



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
    """A compact causal GRU with mean, final, and max temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(204, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, running_max, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, running_max, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(running_max, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, running_max, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, running_max, count = state
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                hidden[:, 0, :],
                running_max,
            ),
            dim=1,
        )
        return self.classifier(features)

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


