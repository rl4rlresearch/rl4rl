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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2093613254443706637, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 21432, "peak_hidden_elements": 166400, "recurrent_macs": 399258720, "recurrent_steps": 22005, "total_inference_macs": 401371200, "training_seconds": 82.76438416587189, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4320945224878978}
prior_hypothesis: Three independent 36-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.4% versus two 48-unit heads, because the aggregate recurrent representation grows from 96 to 108 features while each costly hidden-to-hidden matrix becomes smaller.

## Recent verification evidence

RECENT RESULT
hypothesis: A 48-unit GRU encoding every four-frame chunk into an 80-unit slower GRU will retain at least 85% accuracy while reducing recurrent MACs by approximately 27%, because it preserves every input frame and gated recurrence while replacing redundant full-rate long-timescale updates with eight learned chunk updates.
change: Reset a fast local GRU after each four-frame chunk, feed its endpoint into a slower command-level GRU, and classify from the local temporal mean plus the slow endpoint and trajectory mean.
mechanism: Hierarchical chunk-to-command gated recurrence
evidence_used: Full-rate gated models consistently exceeded 85%, whereas discarding frames and replacing GRU gating with a 160-unit Elman RNN failed. This challenges the shared assumption that one state must model both local acoustics and the full utterance at every frame: all 32 frames remain encoded, but long-timescale recurrence runs only eight times.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2383916742849082288, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 43088, "peak_hidden_elements": 132096, "recurrent_macs": 455669760, "recurrent_steps": 26080, "total_inference_macs": 457025920, "training_seconds": 61.60338733298704, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.41811969967707535}

RECENT RESULT
hypothesis: Processing frames 1–31 with the verified 80-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by 3.125%.
change: Omit only the first input frame while preserving the remaining 31 causal frames, recurrent width, temporal-mean readout, and training procedure.
mechanism: Single boundary-frame omission
evidence_used: The 80-unit full-rate model achieved 85.64% accuracy; although reducing to 28 frames failed, omitting one likely low-information boundary frame is a substantially more conservative temporal reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3168307516804866273, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 606360000, "recurrent_steps": 25265, "total_inference_macs": 607403200, "training_seconds": 43.745184750063345, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.42254018959092216}

RECENT RESULT
hypothesis: Processing frames 2–31 with the verified 80-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.2% versus the successful 31-step model.
change: Omit one additional leading input frame, reducing recurrent execution from 31 to 30 steps while preserving model capacity, temporal-mean readout, and training procedure.
mechanism: Two-frame leading-boundary omission
evidence_used: The 80-unit GRU achieved 85.52% after omitting frame 0, while more aggressive 28-frame schedules failed; removing only the adjacent likely boundary frame is the smallest informative temporal-cost reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3066279580454205458, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 586800000, "recurrent_steps": 24450, "total_inference_macs": 587843200, "training_seconds": 45.05209599994123, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.42810384364215875}

RECENT RESULT
hypothesis: Processing frames 3–31 with the verified 80-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the successful 30-step model.
change: Omit one additional leading input frame, reducing recurrent execution from 30 to 29 steps while preserving model capacity, temporal-mean readout, and training procedure.
mechanism: Three-frame leading-boundary omission
evidence_used: The 80-unit GRU remained above threshold after successive reductions from 32 steps (85.64%) to 31 steps (85.52%) and 30 steps (85.28%); removing the adjacent leading boundary frame is the smallest untested temporal-cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2964251644103544643, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 567240000, "recurrent_steps": 23635, "total_inference_macs": 568283200, "training_seconds": 38.78608241584152, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4323313988059576}

RECENT RESULT
hypothesis: Processing frames 2–30 with the verified 80-unit GRU will achieve at least 85% validation accuracy while reducing recurrent execution from 30 to 29 steps.
change: Preserve the model and training procedure, but omit the final frame instead of the third leading frame.
mechanism: Terminal-boundary frame omission
evidence_used: Frames 2–31 achieved 85.28%, while frames 3–31 narrowly missed at 84.91%; comparing frames 2–30 tests whether the terminal boundary carries less useful information than frame 2 at the same lower cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2964251644103544643, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 567240000, "recurrent_steps": 23635, "total_inference_macs": 568283200, "training_seconds": 45.3903569590766, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43515265529141106}

RECENT RESULT
hypothesis: Processing frames 2–29 with the verified 80-unit GRU will achieve at least 85% validation accuracy while reducing recurrent execution and MACs from 29 to 28 steps.
change: Omit one additional terminal frame while preserving the successful model capacity, leading-frame offset, temporal-mean readout, and training procedure.
mechanism: Progressive terminal-boundary trimming
evidence_used: Frames 2–30 achieved 85.52% accuracy at 29 steps, outperforming the 30-step frames 2–31 schedule’s 85.28%; this indicates the terminal boundary is the most promising location for the next single-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2862223707752883828, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 547680000, "recurrent_steps": 22820, "total_inference_macs": 548723200, "training_seconds": 31.951306459028274, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.44099111732529717}

RECENT RESULT
hypothesis: Processing frames 2–28 while augmenting the endpoint-and-mean classifier with an online temporal maximum will retain at least 85% validation accuracy and reduce total inference MACs below the verified 28-step model.
change: Remove one additional terminal frame, track the elementwise maximum recurrent output, and classify from the final state, temporal mean, and temporal maximum.
mechanism: Terminal trimming with online temporal-max readout
evidence_used: Frames 2–29 achieved 85.28% at 28 steps, while removing leading frame 2 previously caused failure; preserving the early boundary and trimming the terminal boundary is therefore the strongest 27-step candidate, with max pooling adding complementary trajectory evidence at only a small classifier cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2762916516369401253, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 528120000, "recurrent_steps": 22005, "total_inference_macs": 529684800, "training_seconds": 38.68669312517159, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.40201921521520323}

RECENT RESULT
hypothesis: A 76-unit GRU retaining the successful 27-step endpoint, mean, and temporal-maximum readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 8.8%.
change: Reduce the GRU hidden width from 80 to 76 and resize all recurrent state summaries and the classifier input from 240 to 228 features.
mechanism: Temporal-max-assisted recurrent width reduction
evidence_used: The 80-unit 27-step model with temporal maximum achieved 87.61%, substantially above the threshold, while the 76-unit mean-only model narrowly missed at 84.79%; the complementary maximum readout provides a plausible margin for revisiting that width at materially lower cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2520090028048798221, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 481645440, "recurrent_steps": 22005, "total_inference_macs": 483132000, "training_seconds": 31.469110749894753, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.42248674638432226}

RECENT RESULT
hypothesis: A 78-unit GRU retaining the successful 27-step endpoint, mean, and temporal-maximum readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.4%.
change: Reduce the GRU hidden width from 80 to 78 and resize its recurrent states and classifier input from 240 to 234 features.
mechanism: Temporal-max-assisted width interpolation
evidence_used: The 80-unit temporal-maximum model achieved 87.61%, while the 76-unit version narrowly missed at 84.79%; testing their midpoint locates the capacity boundary for this specific readout at lower cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2640125895069466065, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 25320, "peak_hidden_elements": 120320, "recurrent_macs": 504618660, "recurrent_steps": 22005, "total_inference_macs": 506144340, "training_seconds": 51.69099379098043, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4260737694114264}

RECENT RESULT
hypothesis: A 77-unit GRU with the successful 27-step endpoint, mean, and temporal-maximum readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the verified 78-unit model.
change: Reduce the GRU hidden width from 78 to 77 and resize its recurrent summaries and classifier input from 234 to 231 features, preserving the frame schedule and training procedure.
mechanism: Single-unit recurrent width boundary search
evidence_used: The 78-unit temporal-maximum model achieved 85.64%, while the 76-unit version narrowly missed at 84.79%; the untested 77-unit midpoint is the smallest structural cost reduction and directly resolves the remaining width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2579763617274223725, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24765, "peak_hidden_elements": 118784, "recurrent_macs": 493066035, "recurrent_steps": 22005, "total_inference_macs": 494572155, "training_seconds": 52.10705554182641, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.409486248741852}

RECENT RESULT
hypothesis: Two independent 48-unit GRU heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the verified 77-unit GRU.
change: Replace the single densely coupled recurrent state with two continuous 48-unit GRU states that independently encode every scheduled frame, then fuse their endpoints, temporal means, and temporal maxima for classification.
mechanism: Parallel block-diagonal recurrent ensemble
evidence_used: The 77-unit model achieved 86.38%, providing accuracy margin, while its dense hidden-to-hidden matrix dominates cost. Unlike the failed chunk hierarchy, this design never resets local memory or compresses chunks through an intermittently updated bottleneck; both heads preserve the proven 27-step schedule and readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2257674175964516997, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 22512, "peak_hidden_elements": 147968, "recurrent_macs": 430945920, "recurrent_steps": 22005, "total_inference_macs": 432823680, "training_seconds": 64.13959562499076, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4356593313392686}

RECENT RESULT
hypothesis: Three independent 36-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.4% versus two 48-unit heads, because the aggregate recurrent representation grows from 96 to 108 features while each costly hidden-to-hidden matrix becomes smaller.
change: Replace the two 48-unit GRU heads with three 36-unit heads and resize the state summaries and endpoint/mean/maximum classifier input accordingly.
mechanism: Three-way block-diagonal recurrent ensemble
evidence_used: Two independent 48-unit heads achieved 86.13% accuracy and substantially outperformed the denser 77-unit GRU on MACs, showing that partitioned recurrent state preserves accuracy; a third smaller block tests whether further structural factorization can exploit the same effect with more aggregate features and fewer MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2093613254443706637, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 21432, "peak_hidden_elements": 166400, "recurrent_macs": 399258720, "recurrent_steps": 22005, "total_inference_macs": 401371200, "training_seconds": 82.76438416587189, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4320945224878978}



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
    """Three parallel causal GRU heads with complementary learned dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 36, num_layers=1, batch_first=True),
                nn.GRU(20, 36, num_layers=1, batch_first=True),
                nn.GRU(20, 36, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(324, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 3, 36, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 108, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for head_index, gru in enumerate(self.gru_heads):
            head_output, head_hidden = gru(
                normalized,
                hidden[:, head_index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(head_output[:, 0, :])
            next_hidden.append(head_hidden[0])
        output = torch.cat(outputs, dim=-1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        normalized = self.input_norm(frames)
        head_outputs = []
        next_hidden = []
        for head_index, gru in enumerate(self.gru_heads):
            output, head_hidden = gru(
                normalized,
                hidden[:, head_index, :].unsqueeze(0).contiguous(),
            )
            head_outputs.append(output)
            next_hidden.append(head_hidden[0])
        outputs = torch.cat(head_outputs, dim=-1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((hidden.flatten(start_dim=1), mean_output, peak), dim=-1)
        )

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


