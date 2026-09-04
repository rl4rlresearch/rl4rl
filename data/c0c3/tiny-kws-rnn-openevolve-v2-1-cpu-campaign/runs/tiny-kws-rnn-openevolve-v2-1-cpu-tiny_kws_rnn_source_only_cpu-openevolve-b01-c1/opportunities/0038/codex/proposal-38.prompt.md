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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3150486637025353332, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28592, "peak_hidden_elements": 99840, "recurrent_macs": 602721840, "recurrent_steps": 22820, "total_inference_macs": 603986720, "training_seconds": 56.07529004220851, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4429512913241708}
prior_hypothesis: Folding normalized upper-skip bands 16–17 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the highest mel band while reducing label smoothing from 0.03 to 0.02 will recover at least one validation decision and reach 85% accuracy, while retaining the previously measured reduction to approximately 699.7M total MACs.
change: Use bands 0–18 with a 19-input GRU in both recurrent paths and slightly reduce label smoothing to strengthen class margins.
mechanism: High-frequency band pruning with sharper supervision
evidence_used: The prior high-band-pruned model reached 84.91% accuracy—just below the requirement—while improving validation cross-entropy from 0.4833 to 0.4759 and reducing total MACs from 706.0M to 699.7M, indicating that its representation remained strong enough for a small loss adjustment to plausibly cross the threshold.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3649811357126625750, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31970, "peak_hidden_elements": 94720, "recurrent_macs": 699113520, "recurrent_steps": 22820, "total_inference_macs": 699713360, "training_seconds": 85.23320354195312, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4718180276133531}

RECENT RESULT
hypothesis: Using 19 mel bands with label smoothing reduced to 0.01 will recover at least one validation decision, exceed 85% accuracy, and retain the measured reduction to approximately 699.7M total MACs.
change: Remove the highest mel band from both recurrent paths, structurally reduce the GRU input width to 19, and lower label smoothing from 0.03 to 0.01.
mechanism: High-frequency band pruning with further margin sharpening
evidence_used: The 19-band model reached 84.91%, one validation example short of 85%; reducing smoothing from 0.03 to 0.02 improved cross-entropy from 0.4759 to 0.4718 without changing accuracy, motivating one further margin-sharpening step.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3649811357126625750, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31970, "peak_hidden_elements": 94720, "recurrent_macs": 699113520, "recurrent_steps": 22820, "total_inference_macs": 699713360, "training_seconds": 65.76537287514657, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46854710491157014}

RECENT RESULT
hypothesis: Averaging the two highest mel bands into one feature will retain their combined information and recover at least 85% validation accuracy while matching the 19-band model’s approximately 699.7M total inference MACs.
change: Structurally reduce the GRU input to 19 features while replacing high-band deletion with a fixed average of bands 18 and 19 in both recurrent execution paths.
mechanism: Fixed adjacent-band spectral folding
evidence_used: Simply dropping the highest band narrowly missed the target at 84.91% while improving cross-entropy and reducing MACs; folding that band into its adjacent retained band preserves the cost reduction while addressing the likely information loss.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3649811357126625750, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31970, "peak_hidden_elements": 94720, "recurrent_macs": 699113520, "recurrent_steps": 22820, "total_inference_macs": 699713360, "training_seconds": 70.16910841711797, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4768146631907832}

RECENT RESULT
hypothesis: Averaging bands 16–17 and 18–19 into two features will preserve at least 85% validation accuracy while reducing total inference MACs from 699.7M to approximately 693.4M.
change: Reduce the GRU input width from 19 to 18 and replace the existing single high-band fold with two adjacent high-band averages in both recurrent execution paths.
mechanism: Paired high-frequency spectral folding
evidence_used: Folding bands 18–19 recovered accuracy to 85.28% where deleting the highest band reached only 84.91%, showing that fixed adjacent-band aggregation can preserve useful spectral information; extending that mechanism to the neighboring high-frequency pair is the most evidence-supported remaining structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3616958361647955952, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 31692, "peak_hidden_elements": 94720, "recurrent_macs": 692815200, "recurrent_steps": 22820, "total_inference_macs": 693415040, "training_seconds": 69.77482612500899, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4954918048133148}

RECENT RESULT
hypothesis: Two recurrent stages of 50 units, with the upper stage receiving both lower-stage output and normalized acoustic input, will preserve at least 85% accuracy while reducing total inference MACs from 699.7M to approximately 645M.
change: Replace the single 92-unit GRU with two 50-unit GRUs, retain the verified 28-frame and folded-19-band inputs, and classify from pooled and final states at both temporal abstraction levels.
mechanism: Raw-skip hierarchical recurrent factorization
evidence_used: Reducing a single dense state to 91 units failed at 83.31%, and reducing temporal coverage to 27 steps also failed, while the non-recurrent segmented alternative reached only 81.47%. This motivates retaining 28-step learned recurrence and total state capacity while challenging the assumption that every recurrent feature needs dense interaction in one wide state.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3363520967955361266, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30446, "peak_hidden_elements": 102912, "recurrent_macs": 643524000, "recurrent_steps": 22820, "total_inference_macs": 644828000, "training_seconds": 35.9363703338895, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.44480377384490033}

RECENT RESULT
hypothesis: Reducing only the lower GRU from 50 to 49 units will preserve at least 85% validation accuracy while lowering total inference MACs from 644.8M to approximately 633.3M.
change: Keep the verified 28-step, folded-19-band hierarchy and 50-unit upper GRU, but reduce the lower recurrent state by one unit and adjust dependent dimensions.
mechanism: Raw-skip lower-stage width compression
evidence_used: The 50+50 raw-skip hierarchy achieved 85.77% accuracy and substantially outperformed the wider single-GRU alternatives; because the upper stage also receives normalized acoustic input directly, it can plausibly tolerate the smallest structural reduction in lower-stage width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3303460522804915220, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29920, "peak_hidden_elements": 101888, "recurrent_macs": 632022720, "recurrent_steps": 22820, "total_inference_macs": 633313680, "training_seconds": 37.78242283407599, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.431624261586944}

RECENT RESULT
hypothesis: Reducing the lower GRU from 49 to 48 units will preserve at least 85% validation accuracy while lowering total inference MACs by approximately 11.4M from 633.3M.
change: Reduce the lower recurrent state by one unit and update the upper-stage and classifier input dimensions accordingly.
mechanism: Incremental raw-skip lower-stage compression
evidence_used: The previous 50-to-49 lower-stage reduction preserved exactly 85.77% accuracy while reducing total MACs by 11.5M, indicating that the raw acoustic skip lets the upper GRU tolerate incremental compression of the lower stage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3244114273208353300, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29400, "peak_hidden_elements": 100864, "recurrent_macs": 620658360, "recurrent_steps": 22820, "total_inference_macs": 621936280, "training_seconds": 26.771171000087634, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.45241182479390335}

RECENT RESULT
hypothesis: Reducing only the upper GRU from 50 to 49 units will retain at least 85% validation accuracy while lowering total inference MACs from 633.3M to approximately 621.9M.
change: Preserve the verified 49-unit lower GRU, folded 19-band input, and 28-step schedule, while reducing the upper recurrent state by one unit and adjusting the classifier input width.
mechanism: Incremental raw-skip upper-stage compression
evidence_used: The 49+50 hierarchy achieved 85.77% accuracy, while reducing the lower stage again failed at 84.54%; testing the uncompressed upper stage is the most direct alternative one-unit reduction and preserves the lower-stage capacity implicated by that failure.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3243757175431411237, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29397, "peak_hidden_elements": 100864, "recurrent_macs": 620589900, "recurrent_steps": 22820, "total_inference_macs": 621867820, "training_seconds": 52.80284862499684, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4379131855408838}

RECENT RESULT
hypothesis: Reducing only the upper GRU from 49 to 48 units will preserve at least 85% validation accuracy while lowering total inference MACs, because the verified 49+49 model reached 86.50% and prior failure from reducing the lower stage indicates lower-stage capacity is more load-bearing.
change: Preserve the 49-unit lower GRU, folded 19-band input, and 28-step schedule while reducing the upper recurrent state to 48 units and adjusting its state tensors and classifier width.
mechanism: Incremental upper-stage width compression
evidence_used: The current 49+49 hierarchy achieved 86.50%, providing 1.50 percentage points of margin; by contrast, the 48+50 hierarchy achieved only 84.54%, motivating compression of the upper rather than lower stage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3184768023611791380, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28880, "peak_hidden_elements": 99840, "recurrent_macs": 609294000, "recurrent_steps": 22820, "total_inference_macs": 610558880, "training_seconds": 53.20526549988426, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4449613793495974}

RECENT RESULT
hypothesis: Reducing the upper GRU from 48 to 47 units will preserve at least 85% validation accuracy while lowering exact inference MACs below the verified 610.56M, because the preceding 49-to-48 reduction retained 86.01% accuracy.
change: Preserve the 49-unit lower GRU, folded 19-band input, 28-step schedule, summaries, and training procedure while reducing the upper recurrent state by one unit and adjusting its classifier dimensions.
mechanism: Incremental upper-stage width compression
evidence_used: The 49+48 hierarchy achieved 86.01% accuracy, leaving 1.01 percentage points of margin, while the failed 48+50 result indicates the lower stage is more sensitive to compression than the upper stage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3126493067346055649, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28369, "peak_hidden_elements": 98816, "recurrent_macs": 598135020, "recurrent_steps": 22820, "total_inference_macs": 599386860, "training_seconds": 54.5244680410251, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.45730922207510544}

RECENT RESULT
hypothesis: Removing only the folded highest-band feature from the upper GRU’s raw acoustic skip will preserve at least 85% validation accuracy because the lower GRU still receives all 19 features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input from 68 to 67 features and pass it the first 18 normalized acoustic features plus the complete 49-unit lower recurrent output in both execution paths.
mechanism: Redundant upper-stage spectral-skip pruning
evidence_used: The 49+48 hierarchy achieved 86.01% accuracy, while removing a spectral feature from the entire recurrent path failed; retaining all spectral information in the more load-bearing lower stage while pruning only its redundant direct copy is a narrower structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167627330318572356, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28736, "peak_hidden_elements": 99840, "recurrent_macs": 606007920, "recurrent_steps": 22820, "total_inference_macs": 607272800, "training_seconds": 48.76711895898916, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43410169917381614}

RECENT RESULT
hypothesis: Folding normalized upper-skip bands 16–17 into one feature will preserve at least 85% validation accuracy because the lower GRU retains all 19 spectral features, while reducing recurrent inference by exactly 3,286,080 MACs.
change: Reduce the upper GRU input width from 67 to 66 and replace its 18-feature acoustic skip with the first 16 normalized bands plus the average of bands 16–17 in both recurrent paths.
mechanism: Upper-skip adjacent-band folding
evidence_used: Removing the folded highest-band feature only from the upper skip retained 85.77% accuracy and saved 3,286,080 recurrent MACs; prior whole-path experiments also showed adjacent-band folding preserves information better than deletion.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3150486637025353332, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28592, "peak_hidden_elements": 99840, "recurrent_macs": 602721840, "recurrent_steps": 22820, "total_inference_macs": 603986720, "training_seconds": 56.07529004220851, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4429512913241708}



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
    """A raw-skip hierarchy of two compact causal recurrent stages."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(19)
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_gru = nn.GRU(66, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)

    @staticmethod
    def _fold_bands(frames: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                frames[..., :18],
                frames[..., 18:20].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return lower_hidden, upper_hidden, lower_summary, upper_summary, count

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
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :16],
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
        upper_output, upper_hidden = self.upper_gru(
            upper_input.unsqueeze(1),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        upper_output = upper_output[:, 0, :]
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_output,
            upper_summary + upper_output,
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
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :16],
                normalized[..., 16:18].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
        upper_outputs, upper_hidden = self.upper_gru(
            upper_inputs,
            upper_hidden.transpose(0, 1).contiguous(),
        )
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary + upper_outputs.sum(dim=1),
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
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        divisor = count.clamp_min(1.0)
        features = torch.cat(
            (
                lower_summary / divisor,
                upper_summary / divisor,
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))


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


