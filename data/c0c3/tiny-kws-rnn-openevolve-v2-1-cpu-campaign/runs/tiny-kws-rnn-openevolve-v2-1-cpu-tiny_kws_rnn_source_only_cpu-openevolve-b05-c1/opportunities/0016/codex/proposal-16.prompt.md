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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3239935379229828373, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 620078895, "recurrent_steps": 25265, "total_inference_macs": 621135135, "training_seconds": 71.20170233398676, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4404342464142782}
prior_hypothesis: The 81-unit GRU processing 31 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1% versus the verified 32-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 22% versus the verified 24-step, 128-unit design.
change: Reduce the GRU, temporal summary, and classifier width from 128 to 96, and restore the complete 32-frame causal schedule.
mechanism: Trade recurrent width for full temporal resolution
evidence_used: The 128-unit model achieved 87.24% at 32 steps, while temporal subsampling to 16 and 20 steps failed and 24 steps only reached 85.77%; this suggests preserving temporal detail while reducing excess recurrent capacity is the better cost tradeoff.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 106.57166195916943, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4620833508076112}

RECENT RESULT
hypothesis: A 92-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing dense recurrent MACs by approximately 7.5% versus the verified 96-unit model.
change: Reduce the GRU state, temporal summary, and classifier width from 96 to 92 while preserving the full causal frame schedule and training procedure.
mechanism: Incremental recurrent-width compression
evidence_used: The 96-unit, 32-step model achieved 85.40% accuracy at 871.9M MACs, whereas reducing temporal coverage to 16 or 20 steps failed; this motivates preserving all frames and probing a small width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4208312280590011128, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32248, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 806784800, "training_seconds": 126.96434125001542, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4665855548133148}

RECENT RESULT
hypothesis: A 91-unit GRU using both the temporal mean and final recurrent output will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the verified 92-unit model.
change: Reduce recurrent width from 92 to 91, preserve all 32 causal steps, and compensate for the smaller state by classifying the concatenated mean and terminal GRU outputs.
mechanism: Dual-timescale mean-and-terminal recurrent readout
evidence_used: The 92-unit full-frame model narrowly passed at 85.03%, while temporal subsampling failed at 16 and 20 steps; preserving every frame and adding a richer, low-cost readout is therefore safer than removing temporal observations.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4128526434427528113, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32353, "peak_hidden_elements": 93696, "recurrent_macs": 790302240, "recurrent_steps": 26080, "total_inference_macs": 791488880, "training_seconds": 100.87191074993461, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.42103065537528755}

RECENT RESULT
hypothesis: A 90-unit GRU retaining the mean-and-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.0% versus the verified 91-unit model.
change: Reduce the GRU state and temporal-summary width from 91 to 90 and resize the concatenated classifier input from 182 to 180, preserving all 32 causal steps and the verified training procedure.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The 91-unit dual-readout model achieved 86.01% accuracy at 791.49M total MACs, giving a 1.01-point margin above the requirement and motivating the next incremental structural reduction without sacrificing temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4046427955042943408, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31728, "peak_hidden_elements": 92672, "recurrent_macs": 774576000, "recurrent_steps": 26080, "total_inference_macs": 775749600, "training_seconds": 65.46588225010782, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.38939220217839343}

RECENT RESULT
hypothesis: An 88-unit GRU retaining the full 32-frame schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 4% versus the verified 90-unit model.
change: Reduce the GRU state and summary width from 90 to 88 and resize the concatenated classifier input from 180 to 176, leaving training and temporal coverage unchanged.
mechanism: Two-unit dual-timescale GRU width compression
evidence_used: The verified 90-unit model achieved 87.12% accuracy, 2.12 points above the requirement, at 775.75M total MACs; this unusually strong margin supports testing a two-unit structural reduction while preserving the readout and all 32 frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 92.70229770801961, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.40472788430430406}

RECENT RESULT
hypothesis: An 86-unit GRU preserving all 32 frames and the mean-plus-terminal readout will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.1% versus the verified 88-unit model.
change: Reduce the GRU state and summary width from 88 to 86 and resize the concatenated classifier input from 176 to 172, leaving temporal coverage and training unchanged.
mechanism: Two-unit dual-timescale GRU width compression
evidence_used: The 88-unit full-frame model achieved 86.87% accuracy—1.87 points above the requirement—at 744.74M total MACs, supporting another cautious two-unit structural reduction without revisiting temporal subsampling, which failed at 16 and 20 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3726196272406137448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 713235840, "recurrent_steps": 26080, "total_inference_macs": 714357280, "training_seconds": 114.65576104214415, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.42406827774515915}

RECENT RESULT
hypothesis: Encoding every frame with a 64-unit RNN reset every four frames, then updating a 64-unit GRU from each ordered block descriptor, will retain at least 85% accuracy while reducing recurrent MACs from 713.2M to approximately 300.4M.
change: Replace the monolithic 86-unit full-rate GRU with a two-timescale recurrent hierarchy: a short-horizon local RNN processes all 32 frames, and a persistent GRU processes eight terminal-plus-mean block summaries.
mechanism: Hierarchical four-frame acoustic recurrence with a slower lexical GRU
evidence_used: Skipping frames at 16 and 20 steps failed, while 24 and 32 steps passed, indicating that retaining acoustic observations is load-bearing. This patch preserves every frame but challenges the assumption that the entire gated state must update at full rate.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1571502295649839760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31920, "peak_hidden_elements": 131584, "recurrent_macs": 300441600, "recurrent_steps": 26080, "total_inference_macs": 301276160, "training_seconds": 97.52072033309378, "validation_accuracy": 0.8294478527607362, "validation_cross_entropy": 0.4815550821690472}

RECENT RESULT
hypothesis: An 84-unit GRU preserving all 32 frames and the mean-plus-terminal readout will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.2% versus the verified 86-unit model.
change: Reduce the GRU and summary width from 86 to 84 and resize the concatenated classifier input from 172 to 168, leaving temporal coverage and training unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 86-unit model achieved 86.63% accuracy, 1.63 points above the requirement, while every tested dual-readout width from 86 through 91 passed; this supports another cautious two-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3570977772028654184, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 684600000, "training_seconds": 124.01448708283715, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.40037742567939993}

RECENT RESULT
hypothesis: An 82-unit GRU with the full 32-frame schedule and mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.3% versus the verified 84-unit model.
change: Reduce the GRU and summary width from 84 to 82 and resize the concatenated classifier input from 168 to 164, leaving training and temporal coverage unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 84-unit model achieved 87.73% accuracy, 2.73 points above the requirement, and all tested dual-readout widths from 84 through 91 passed; this supports another cautious two-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3419024165611784064, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26944, "peak_hidden_elements": 84480, "recurrent_macs": 654399360, "recurrent_steps": 26080, "total_inference_macs": 655468640, "training_seconds": 181.46486658300273, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223508308270226}

RECENT RESULT
hypothesis: An 80-unit GRU with all 32 frames and the mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.5% versus the verified 82-unit model.
change: Reduce the GRU and summary width from 82 to 80 and resize the concatenated classifier input from 164 to 160, leaving training and temporal coverage unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 82-unit model achieved 86.01% accuracy at 655.47M total MACs, and every tested dual-readout width from 82 through 91 passed, making the next two-unit reduction the most direct probe of the minimum viable width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 68.48943950003013, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4433465594894316}

RECENT RESULT
hypothesis: An 81-unit GRU using all 32 frames and the mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.2% versus the verified 82-unit model.
change: Reduce the GRU state and summary width from 82 to 81 and resize the concatenated classifier input from 164 to 162, preserving temporal coverage and training.
mechanism: One-unit dual-timescale GRU width compression
evidence_used: The 82-unit model passed at 86.01% while the 80-unit model narrowly failed at 84.29%; testing 81 units directly resolves the remaining minimum-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3344271697638578933, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 640081440, "recurrent_steps": 26080, "total_inference_macs": 641137680, "training_seconds": 111.35484620905481, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.43995596294754125}

RECENT RESULT
hypothesis: The 81-unit GRU processing 31 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1% versus the verified 32-step model.
change: Reduce the causal frame schedule from 32 to 31 unique increasing indices while preserving both recording endpoints, the mean-plus-terminal readout, and training procedure.
mechanism: Single-frame endpoint-preserving temporal compression
evidence_used: The 81-unit 32-step model passed at 85.64% accuracy, while 24-frame processing has previously passed at higher width; with the recurrent-width boundary already isolated between 80 and 81 units, removing one frame is the most cautious probe of the remaining temporal redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3239935379229828373, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 620078895, "recurrent_steps": 25265, "total_inference_macs": 621135135, "training_seconds": 71.20170233398676, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4404342464142782}



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
    """A one-layer causal GRU with mean and terminal-state readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 81, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(162, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 81, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 81, device=device, dtype=dtype)
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
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(31, available_frames)
        return [
            i * (available_frames - 1) // (steps - 1)
            for i in range(steps)
        ]


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


