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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1536540722251086223, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294573600, "training_seconds": 56.65610625012778, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4427040802189178}
prior_hypothesis: A 60-unit GRU processing frames 3–26 and frame 28 will achieve at least 85% validation accuracy while reducing total dense inference MACs from 306,309,600 to approximately 294,573,600.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067302, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16027, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 38.866629750002176, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4617033975987347}
prior_hypothesis: Removing LayerNorm’s foldable affine parameters while adopting the qualified 25-step, seven-logit design will retain at least 85% validation accuracy at 294,426,900 MACs and reduce parameters from 16,067 to 16,027.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067295, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 30.942437916994095, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4581207088166219}
prior_hypothesis: Disabling LayerNorm affine parameters and the seven-logit classifier bias will retain at least 85% validation accuracy at 294,426,900 MACs while reducing learned parameters from 16,067 to 16,020.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1720191007780076668, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 328608000, "recurrent_steps": 22820, "total_inference_macs": 329781600, "training_seconds": 76.99711412494071, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4377409110040021}
prior_hypothesis: The 60-unit GRU can process frames 3–30 and retain at least 85% validation accuracy while reducing total inference MACs from 341,517,600 to approximately 329,781,600.

## Recent verification evidence

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–29 will retain at least 85% validation accuracy while reducing total inference MACs from 329,781,600 to approximately 318,045,600.
change: Widen the current GRU from 58 to 60 units and adopt a 27-step causal schedule that removes one additional trailing boundary frame from the qualified 60-unit design.
mechanism: Incremental trailing-boundary pruning
evidence_used: The 60-unit model processing frames 3–30 achieved 86.50% accuracy at 329,781,600 MACs, while excluding frame 3 was worse; its 1.50-point margin motivates preserving the informative leading boundary and pruning the next trailing frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1658974245937079853, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 316872000, "recurrent_steps": 22005, "total_inference_macs": 318045600, "training_seconds": 72.58885912504047, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.43758639470199867}

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–28 will retain at least 85% validation accuracy while reducing total inference MACs from 318,045,600 to approximately 306,309,600.
change: Use the qualified 60-unit architecture and remove one additional trailing frame, reducing recurrent execution from 27 to 26 steps.
mechanism: Incremental trailing-boundary pruning
evidence_used: The 60-unit model processing frames 3–29 achieved 86.26% accuracy at 318,045,600 MACs, only 0.25 points below the 28-step design; this margin and stability motivate one more trailing-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1597757484094083038, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 305136000, "recurrent_steps": 21190, "total_inference_macs": 306309600, "training_seconds": 78.6076197498478, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4400964093354582}

RECENT RESULT
hypothesis: The 60-unit GRU processing frames 3–27 will retain at least 85% validation accuracy while reducing total inference MACs from 306,309,600 to approximately 294,573,600.
change: Remove one additional trailing frame from the causal schedule, reducing recurrent execution from 26 to 25 steps while preserving model capacity, summaries, and training.
mechanism: Incremental trailing-boundary pruning
evidence_used: The qualified 60-unit models retained 86.50%, 86.26%, and 85.40% accuracy across 28, 27, and 26 steps respectively; testing 25 steps is the smallest direct probe of the trailing-context boundary and saves one complete recurrent step per example.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1536540722251086223, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294573600, "training_seconds": 70.1498493750114, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.44786478054304063}

RECENT RESULT
hypothesis: A 61-unit GRU processing frame 3 and frames 5–28 will retain at least 85% accuracy while reducing total inference MACs below the qualified 306,309,600-MAC design to approximately 303,211,785.
change: Widen the GRU by one unit and execute 25 steps, preserving the informative frame-28 endpoint while omitting the likely redundant early frame 4.
mechanism: Endpoint-preserving capacity-for-time exchange
evidence_used: The 60-unit frames-3–28 design qualified at 85.40%, whereas removing frame 28 fell to 84.17%; retaining that endpoint and exchanging one early frame for modest added capacity directly targets the observed boundary, while the 58-to-60 width evidence showed a 1.35-point accuracy gain.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1581598809605954461, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16701, "peak_hidden_elements": 94208, "recurrent_macs": 302018625, "recurrent_steps": 20375, "total_inference_macs": 303211785, "training_seconds": 60.547172790858895, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.49923794986280196}

RECENT RESULT
hypothesis: The 60-unit GRU processing frames 3–26 and frame 28 will retain at least 85% accuracy while reducing total inference MACs from 306,309,600 to approximately 294,573,600.
change: Reduce execution from 26 to 25 recurrent steps by omitting frame 27 while preserving the informative frame-28 endpoint and the contiguous early sequence.
mechanism: Penultimate-frame pruning with endpoint preservation
evidence_used: Frames 3–28 achieved 85.40%, while truncating to frames 3–27 fell to 84.17%; preserving frame 28 and pruning its predecessor is the most direct test of whether the endpoint matters more than one adjacent late frame. The failed frame-4 omission also argues against disrupting early continuity.
result: the patch search text did not match the source

RECENT RESULT
hypothesis: A 60-unit GRU processing frames 3–26 and frame 28 will achieve at least 85% validation accuracy while reducing total dense inference MACs from 306,309,600 to approximately 294,573,600.
change: Replace the 128-unit mean-only baseline with the qualified 60-unit mean/maximum/final-state architecture and a 25-step schedule that omits frame 27 while retaining frame 28.
mechanism: Penultimate-frame pruning with endpoint preservation
evidence_used: The 60-unit contiguous frames 3–28 design achieved 85.40% accuracy, while truncating it to frames 3–27 fell to 84.17%; this directly tests whether preserving the final endpoint while removing its predecessor recovers the accuracy threshold. The prior attempt did not apply, so it provides no contradictory verification result.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1536540722251086223, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16248, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294573600, "training_seconds": 56.65610625012778, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4427040802189178}

RECENT RESULT
hypothesis: A rank-17 input projection will preserve at least 85% validation accuracy on the qualified 25-step schedule while reducing total dense inference MACs from 294,573,600 to approximately 290,498,600.
change: Preserve the 60-unit GRU, temporal summaries, and frames 3–26 plus frame 28, but factor each GRU input transform through an orthogonally initialized bias-free 20-to-17 Linear projection.
mechanism: Learned low-rank spectral bottleneck
evidence_used: The current 25-step schedule qualified at 85.15%, whereas further temporal truncation and disrupting early continuity failed; an orthogonal spectral-rank reduction is a conservative, independent way to reduce matrix cost without removing another observed frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1515284902195011023, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16048, "peak_hidden_elements": 92672, "recurrent_macs": 289325000, "recurrent_steps": 20375, "total_inference_macs": 290498600, "training_seconds": 54.23638283298351, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4677789980648485}

RECENT RESULT
hypothesis: A seven-logit reference-class head is expressively equivalent under softmax and will preserve at least 85% accuracy on the qualified 25-step schedule while reducing total dense inference MACs from 294,573,600 to approximately 294,426,900.
change: Adopt frames 3–26 plus frame 28, and replace the redundant eight-output affine classifier with seven learned relative logits and one zero reference logit.
mechanism: Reference-class softmax with endpoint-preserving frame pruning
evidence_used: The 25-step endpoint-preserving schedule achieved 85.15% accuracy at 294,573,600 MACs. Softmax is invariant to a shared logit offset, so eight-class probabilities require only seven learned logit contrasts.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067342, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16067, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 43.60041083302349, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.45144006459990893}

RECENT RESULT
hypothesis: Removing LayerNorm’s foldable affine parameters while adopting the qualified 25-step, seven-logit design will retain at least 85% validation accuracy at 294,426,900 MACs and reduce parameters from 16,067 to 16,027.
change: Process frames 3–26 and frame 28, learn seven relative logits with a fixed zero reference logit, and disable the redundant affine transform in the input LayerNorm.
mechanism: Affine-folded input normalization with reference-class logits
evidence_used: Reference Design 2 achieved 85.77% accuracy at 294,426,900 MACs over 25 steps. LayerNorm’s default affine transform is followed immediately by the GRU’s learned affine input maps, so its scale and offset are structurally absorbable without reducing model expressivity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067302, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16027, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 38.866629750002176, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4617033975987347}

RECENT RESULT
hypothesis: A 59-unit GRU using the proven 25-frame endpoint-preserving schedule and seven-logit head will retain at least 85% validation accuracy while reducing total inference MACs from 294,426,900 to approximately 285,913,410.
change: Reduce GRU width from 60 to 59, adopt frames 3–26 plus frame 28, and use seven learned relative logits with a fixed zero reference logit.
mechanism: One-unit recurrent width pruning on the qualified sparse schedule
evidence_used: Reference Design 3 achieved 85.77% accuracy with 60 units, 25 steps, and seven logits; its 0.77-point margin motivates the smallest structural width reduction, which saves about 8.5 million MACs without discarding another observed frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1491367853467915008, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15623, "peak_hidden_elements": 91136, "recurrent_macs": 284903625, "recurrent_steps": 20375, "total_inference_macs": 285913410, "training_seconds": 42.615200750064105, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.47942975752192774}

RECENT RESULT
hypothesis: Removing one final-state coordinate from the classifier input will retain at least 85% validation accuracy while reducing total inference MACs by 5,705 and learned parameters by 7.
change: Keep the proven 60-unit GRU, 25-frame schedule, temporal summaries, and seven-logit head, but reduce the classifier input from 180 to 179 features by omitting one final-state coordinate.
mechanism: Single-coordinate classifier-feature pruning
evidence_used: The current design qualifies at 85.03%, while reducing recurrent width to 59 or input rank to 17 failed substantially; pruning one classifier coordinate is a smaller structural reduction, and that hidden coordinate remains represented through both mean and maximum summaries.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535745754580988790, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294421195, "training_seconds": 34.450375667074695, "validation_accuracy": 0.8233128834355828, "validation_cross_entropy": 0.4895681275911858}

RECENT RESULT
hypothesis: Disabling LayerNorm affine parameters and the seven-logit classifier bias will retain at least 85% validation accuracy at 294,426,900 MACs while reducing learned parameters from 16,067 to 16,020.
change: Make the input LayerNorm non-affine and the reference-class classifier bias-free, preserving the qualified 60-unit GRU, 25-frame schedule, and seven-logit formulation.
mechanism: Redundant affine-parameter pruning
evidence_used: The non-affine LayerNorm design already qualified at 85.03% with 16,027 parameters and identical MACs. Removing only seven classifier biases is a smaller additional restriction than the failed recurrent-width, input-rank, or classifier-feature pruning experiments.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1535775512729067295, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 16020, "peak_hidden_elements": 92672, "recurrent_macs": 293400000, "recurrent_steps": 20375, "total_inference_macs": 294426900, "training_seconds": 30.942437916994095, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4581207088166219}



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
    """A one-layer causal GRU with online temporal summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]


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
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]


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
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]


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
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, 1, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        maximum = torch.full(
            (batch_size, self.hidden_size),
            -1.0,
            device=device,
            dtype=dtype,
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 1))


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
