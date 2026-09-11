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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4003621805354731553, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98282208588957, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764165259, "recurrent_steps": 21991, "total_inference_macs": 767543139, "training_seconds": 184.6520685828291, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46504103771747984}
prior_hypothesis: Evaluating the full classifier only for the highest-energy quartile at step 22 will retain at least five of the 14 previously observed step-22 exits, preserving accuracy at or above 85% while reducing total inference MACs below 767,879,145.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending exact classifier-bound checks from step 29 to step 28 will preserve validation accuracy at or above 85% while reducing total inference MACs below 825,918,687 by allowing confident examples to skip both remaining recurrent steps.
change: Run the learned classifier beginning at recurrent step 28 and apply the existing mathematically conservative bounded-output certificate at both steps 28 and 29.
mechanism: Two-step certified early exit
evidence_used: The step-29 certificate preserved 85.28% accuracy while 86.3% of examples exited early; the broader step-20 attempt provided no contrary accuracy evidence because training timed out. Adding only the immediately preceding check is a lower-overhead probe whose certified exits cannot change the final predicted class.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4203589080597487453, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 28.42331288343558, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 804960585, "recurrent_steps": 23165, "total_inference_macs": 805879305, "training_seconds": 196.69990370911546, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46787701004121934}

RECENT RESULT
hypothesis: Beginning the exact classifier-bound checks at recurrent step 27 will preserve validation accuracy at or above 85% while reducing total inference MACs and mean recurrent steps below the verified step-28 design.
change: Enable the learned classifier and conservative bounded-output exit certificate during the final three recurrent steps instead of the final two.
mechanism: Three-step certified early exit
evidence_used: Step-28 certified exit achieved 85.03% accuracy and reduced mean execution to 28.42 steps; extending the same mathematically conservative certificate by one step is the smallest next cost-reduction probe, while the much earlier step-20 attempt timed out without contrary accuracy evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4120565537445158201, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.851533742331288, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 788767551, "recurrent_steps": 22699, "total_inference_macs": 789962679, "training_seconds": 178.55719587486237, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4701891191166603}

RECENT RESULT
hypothesis: Beginning the exact classifier-bound checks at recurrent step 26 will preserve validation accuracy at or above 85% while reducing total inference MACs below 789,962,679 and mean recurrent steps below 27.8515.
change: Enable the learned classifier and conservative bounded-output exit certificate during the final four recurrent steps instead of the final three.
mechanism: Four-step certified early exit
evidence_used: Extending certified checks from step 28 to step 27 preserved 85.03% accuracy while reducing total MACs from 805,879,305 to 789,962,679; moving the same mathematically conservative certificate back one additional step is the smallest supported cost-reduction probe.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4061271564091754005, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.43680981595092, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 777022389, "recurrent_steps": 22361, "total_inference_macs": 778595301, "training_seconds": 171.58000745810568, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4703606073110381}

RECENT RESULT
hypothesis: Supervising classifier predictions at recurrent steps 26–29 will preserve final validation accuracy at or above 85% while enabling certified checks from step 26 to reduce MACs below the verified step-27 design.
change: Cache training-only logits from the four late prefixes, add a lightly weighted auxiliary cross-entropy loss, and begin learned classification and certified exit checks at step 26.
mechanism: Late-prefix deep supervision with four-step certified exit
evidence_used: Step-26 certified exit missed the threshold by one validation example at 84.91% while lowering total MACs to 778,595,301; step-27 met the target at 85.03%, so targeted late-prefix supervision is a minimal accuracy-oriented change to recover that narrow deficit.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Supervising the four summaries immediately preceding the final state will recover the one-example accuracy deficit of step-26 certified exit, achieving at least 85% accuracy while reducing MACs below the verified step-27 design.
change: Cache steps 26–29 during the standard training sequence, apply a lightly weighted auxiliary classification loss to them, and enable certified exits beginning at step 26.
mechanism: Verifiable late-prefix deep supervision
evidence_used: Step-26 certified exit reached 84.91%, only one validation example below the requirement, while reducing total inference MACs to 778,595,301; targeted prefix supervision directly trains the representations used by those earlier checks without changing inference architecture or parameter count.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing label smoothing from 0.03 to 0.02 will recover the one-example accuracy deficit of step-26 certified exit while increasing late-prefix margins, achieving at least 85% accuracy and fewer than 789,962,679 total inference MACs.
change: Begin learned classification and conservative exit certification at recurrent step 26, and slightly sharpen the training targets without changing model size or inference architecture.
mechanism: Margin-sharpened four-step certified early exit
evidence_used: Step-26 certified exit reduced total MACs to 778,595,301 but missed the accuracy threshold by only one validation example at 84.91%; step 27 reached 85.03%, making a small training-only margin adjustment the most targeted verifiable follow-up after the auxiliary-loss implementations could not be verified.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4059973853243375291, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.428220858895706, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 776779146, "recurrent_steps": 22354, "total_inference_macs": 778346514, "training_seconds": 160.25353908305988, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4649836721595811}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 25 will retain validation accuracy at or above 85% while reducing total inference MACs below 778,346,514 and mean recurrent steps below 27.4283.
change: Enable learned classification and the existing bounded-output exit certificate one recurrent step earlier, during the final five steps instead of the final four.
mechanism: Five-step certified early exit
evidence_used: Extending certified checks from step 27 to step 26 reduced total MACs from 789,962,679 to 778,346,514, and label smoothing 0.02 restored accuracy to 85.15%; moving the unchanged certificate back one more step is the smallest supported cost-reduction test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4024965611481058457, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.174233128834356, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 769586103, "recurrent_steps": 22147, "total_inference_macs": 771635007, "training_seconds": 147.53195574996062, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4650554165518357}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 24 will retain validation accuracy at or above 85% while reducing total inference MACs below 771,635,007 and mean recurrent steps below 27.1743.
change: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final six steps instead of the final five.
mechanism: Six-step certified early exit
evidence_used: Moving the certificate from step 26 to step 25 with label smoothing 0.02 preserved 85.15% accuracy and reduced total MACs from 778,346,514 to 771,635,007; extending the unchanged certificate by one step is the smallest supported cost-reduction probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4008866870652260027, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.045398773006134, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 765937458, "recurrent_steps": 22042, "total_inference_macs": 768548682, "training_seconds": 154.15326458308846, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4650977479899588}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 23 will retain validation accuracy at or above 85% while reducing total inference MACs below 768,548,682 and mean recurrent steps below 27.0454.
change: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final seven steps instead of the final six.
mechanism: Seven-step certified early exit
evidence_used: Moving the certificate successively from step 26 through step 24 preserved 85.15% accuracy while reducing total MACs at every extension; step 24 reached 768,548,682 MACs and 27.0454 mean steps, motivating the smallest supported next probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4005374463779140533, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764651745, "recurrent_steps": 22005, "total_inference_macs": 767879145, "training_seconds": 188.3290833751671, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46505390003414976}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 22 will retain validation accuracy at or above 85% while reducing total inference MACs below 767,879,145 and mean recurrent steps below 27.0.
change: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final eight steps instead of the final seven.
mechanism: Eight-step certified early exit
evidence_used: Extending the unchanged certificate successively through steps 26, 25, 24, and 23 preserved 85.15% accuracy while consistently reducing inference cost; step 23 achieved 767,879,145 MACs and exactly 27.0 mean steps.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4006145963979265385, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98282208588957, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764165259, "recurrent_steps": 21991, "total_inference_macs": 768027051, "training_seconds": 193.41725833294913, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46504105643992044}

RECENT RESULT
hypothesis: Evaluating the full classifier only for the highest-energy quartile at step 22 will retain at least five of the 14 previously observed step-22 exits, preserving accuracy at or above 85% while reducing total inference MACs below 767,879,145.
change: Add a zero-MAC recurrent-activation energy screen at step 22, run the learned classifier only on the selected quartile, and retain the existing full certified checks from step 23 onward.
mechanism: Activation-energy-screened certified early exit
evidence_used: Unscreened step-22 checks saved only 14 recurrent steps and increased total MACs because all examples incurred an extra 99-to-8 classification; screening to roughly one quarter of examples breaks even if it retains five certified exits, while the conservative certificate protects predictions.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4003621805354731553, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98282208588957, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764165259, "recurrent_steps": 21991, "total_inference_macs": 767543139, "training_seconds": 184.6520685828291, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46504103771747984}

RECENT RESULT
hypothesis: Screening the highest-energy quartile at step 21 will produce at least five additional certified step savings, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,543,139.
change: Start the existing energy-screened classifier and conservative exit certificate at step 21 instead of step 22, while retaining all later checks unchanged.
mechanism: Energy-screened ninth-step certified exit
evidence_used: Moving the certificate earlier produced diminishing marginal savings of 105, 37, and 14 recurrent steps at steps 24, 23, and 22; the step-22 energy screen retained all 14 savings while evaluating only one quarter of examples, and five step-21 savings are sufficient to offset the added classifier calls.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4004283312555537411, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98159509202454, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764130510, "recurrent_steps": 21990, "total_inference_macs": 767669958, "training_seconds": 151.47394229192287, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.465038238712615}



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
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)

    def _input_features(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_bands = normalized[..., 16:].reshape(
            *normalized.shape[:-1], 2, 2
        ).mean(dim=-1)
        return torch.cat((normalized[..., :16], high_bands), dim=-1)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self._input_features(frame).unsqueeze(1),
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
            self._input_features(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    @staticmethod
    def _high_energy_quartile(averaged: torch.Tensor) -> torch.Tensor:
        energy = averaged.square().mean(dim=1)
        candidate_count = max(1, (averaged.shape[0] + 3) // 4)
        indices = energy.topk(candidate_count, sorted=False).indices
        candidates = torch.zeros_like(energy, dtype=torch.bool)
        candidates[indices] = True
        return candidates

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        if bool(torch.all(count < 22.0)):
            return averaged[:, :8]
        if bool(torch.all(count < 23.0)):
            candidates = self._high_energy_quartile(averaged)
            logits = averaged[:, :8].clone()
            logits[candidates] = self.classifier(averaged[candidates])
            return logits
        return self.classifier(averaged)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, summary, count = state
        completed = count[:, 0]
        eligible = (completed >= float(total_steps - 8)) & (
            completed < float(total_steps)
        )
        if not bool(eligible.any()):
            return eligible

        early_check = completed < float(total_steps - 7)
        if bool(early_check.any()):
            averaged = summary / count.clamp_min(1.0)
            candidates = self._high_energy_quartile(averaged)
            eligible = eligible & (~early_check | candidates)
            if not bool(eligible.any()):
                return eligible

        winner = logits.argmax(dim=1)
        weights = self.classifier.weight.detach()
        biases = self.classifier.bias.detach()
        pairwise_l1 = (
            weights[:, None, :] - weights[None, :, :]
        ).abs().sum(dim=-1)
        uncertainty = pairwise_l1[winner]
        bias_delta = biases[winner, None] - biases[None, :]
        current_margin = logits.gather(1, winner[:, None]) - logits
        remaining = float(total_steps) - completed
        lower_bound = (
            completed[:, None] * current_margin
            + remaining[:, None] * (bias_delta - uncertainty)
        ) / float(total_steps)
        lower_bound = lower_bound.scatter(
            1, winner[:, None], float("inf")
        )
        return eligible & (lower_bound.min(dim=1).values > 0.0)


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
    return F.cross_entropy(logits, labels, label_smoothing=0.02)


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


