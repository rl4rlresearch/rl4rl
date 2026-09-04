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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1672543960564380418, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27498, "peak_hidden_elements": 149504, "recurrent_macs": 318749760, "recurrent_steps": 13040, "total_inference_macs": 320647080, "training_seconds": 118.59717333316803, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40993053225651843}
prior_hypothesis: A 97-unit state with 31/32/33 reset-update-candidate feedback and mean-max-endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs to approximately 320,647,080.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1689786681793868740, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27780, "peak_hidden_elements": 151040, "recurrent_macs": 322035840, "recurrent_steps": 13040, "total_inference_macs": 323952720, "training_seconds": 113.36673708399758, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4024184899827454}
prior_hypothesis: A 98-unit 31/32/33 gated state with explicit temporal maximum pooling will recover at least 85% validation accuracy while reducing total inference MACs below the qualified 99-unit model, to approximately 323,952,720.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1703662481126473990, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27270, "peak_hidden_elements": 101888, "recurrent_macs": 325321920, "recurrent_steps": 13040, "total_inference_macs": 326612880, "training_seconds": 131.3961366249714, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4360813538721003}
prior_hypothesis: A 99-unit state with 31/32/33 reset-update-candidate feedback widths will retain at least 85% validation accuracy while reducing total inference MACs to approximately 326,612,880.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1638058518105403774, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 26934, "peak_hidden_elements": 146432, "recurrent_macs": 312177600, "recurrent_steps": 13040, "total_inference_macs": 314035800, "training_seconds": 133.6423793330323, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4118773665164877}
prior_hypothesis: A 95-unit state retaining the 31/32/33 feedback allocation and mean-max-endpoint readout will achieve at least 85% validation accuracy while reducing total inference MACs from 317,341,440 to approximately 314,035,800.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the gated state from 104 to 103 units while retaining all three 32-unit feedback bottlenecks will preserve at least 85% validation accuracy and reduce total inference MACs from 343,108,480 to approximately 339,809,360.
change: Resize the input gate projections, recurrent state, gate expansions, temporal summary, and endpoint-augmented classifier from 104 to 103 units while preserving the qualified training procedure and 16-frame schedule.
mechanism: Gated-state width boundary trim
evidence_used: The qualified 104-state, 32/32/32 bottleneck design achieved 85.521% accuracy; unlike the failed bottleneck-rank trims, this nearest structural reduction preserves the full demonstrated feedback capacity of every gate and would improve upon the 340,396,160-MAC selective-reset design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1772497328796068366, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28366, "peak_hidden_elements": 105984, "recurrent_macs": 338466240, "recurrent_steps": 13040, "total_inference_macs": 339809360, "training_seconds": 111.88948058406822, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4285078645483848}

RECENT RESULT
hypothesis: A 102-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 339,809,360 to approximately 336,510,240.
change: Replace the dense 93-unit GRU with a 102-unit bottleneck-gated recurrence, preserving the proven 16-frame schedule and mean/final-state readout.
mechanism: Full-rank gate feedback with a narrower recurrent state
evidence_used: The 103-state design with three 32-unit bottlenecks achieved 85.399% accuracy at 339,809,360 MACs, while bottleneck-rank reductions were less reliable; trimming state width by one is the closest untested structural reduction that preserves full gate-feedback capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1755288616878669772, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28092, "peak_hidden_elements": 104960, "recurrent_macs": 335180160, "recurrent_steps": 13040, "total_inference_macs": 336510240, "training_seconds": 108.23541283304803, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4422338474016248}

RECENT RESULT
hypothesis: A 101-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 336,510,240 to approximately 333,211,120.
change: Replace the dense 96-unit GRU with a 101-unit bottleneck-gated recurrence, preserving the 16-frame schedule and mean/final-state readout.
mechanism: Full gate-feedback capacity with a narrower recurrent state
evidence_used: The adjacent 102-unit bottleneck-gated design qualified at 85.276% accuracy and 336,510,240 MACs; prior bottleneck-rank reductions were less reliable, so trimming state width by one is the closest supported structural reduction that preserves all demonstrated gate-feedback capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1738079904961271178, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27818, "peak_hidden_elements": 103936, "recurrent_macs": 331894080, "recurrent_steps": 13040, "total_inference_macs": 333211120, "training_seconds": 108.37409804109484, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.428412754550302}

RECENT RESULT
hypothesis: A 100-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 333,211,120 to approximately 329,912,000.
change: Reduce the recurrent state, input gate projections, gate expansions, temporal summary, and endpoint-augmented classifier from 101 to 100 units while preserving the qualified 16-frame schedule and training procedure.
mechanism: Full gate-feedback capacity with a one-unit state trim
evidence_used: The adjacent 101-unit design achieved 85.399% accuracy at 333,211,120 MACs, and each prior one-unit state trim from 103 through 101 qualified while saving exactly 3,299,120 MACs; keeping all three bottlenecks at 32 avoids the accuracy failures observed when feedback rank was reduced.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1720871193043872584, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27544, "peak_hidden_elements": 102912, "recurrent_macs": 328608000, "recurrent_steps": 13040, "total_inference_macs": 329912000, "training_seconds": 96.84299454209395, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.43103351241971816}

RECENT RESULT
hypothesis: A 100-unit state with 31/32/33 reset-update-candidate feedback widths will achieve at least 85% validation accuracy at approximately 329,912,000 MACs by shifting one coordinate from the less capacity-sensitive reset path to the candidate path.
change: Reduce the state to 100 units and reallocate the fixed 96-dimensional recurrent bottleneck from 32/32/32 to 31/32/33, preserving the 16-frame schedule and training procedure.
mechanism: Cost-neutral feedback-rank reallocation toward candidate generation
evidence_used: The 100-unit 32/32/32 model narrowly failed at 84.417%, while the qualified 104-unit 31/32/32 result shows the reset path can tolerate a one-coordinate reduction; the failure of 31/31/31 indicates update/candidate capacity is more valuable, motivating a cost-neutral transfer to the candidate path.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1720871193043872584, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27544, "peak_hidden_elements": 102912, "recurrent_macs": 328608000, "recurrent_steps": 13040, "total_inference_macs": 329912000, "training_seconds": 106.52642687503248, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.42420507325716544}

RECENT RESULT
hypothesis: A 99-unit state with 31/32/33 reset-update-candidate feedback widths will retain at least 85% validation accuracy while reducing total inference MACs to approximately 326,612,880.
change: Reduce the recurrent state and readout from 103 to 99 units, while reallocating the fixed 96-dimensional feedback bottleneck from 32/32/32 to 31/32/33.
mechanism: Candidate-biased feedback with a one-unit state trim
evidence_used: The qualified 100-unit 31/32/33 design achieved 85.890% accuracy at 329,912,000 MACs, providing the strongest margin among bottleneck-gated designs and making a one-unit state reduction the closest supported cost improvement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1703662481126473990, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27270, "peak_hidden_elements": 101888, "recurrent_macs": 325321920, "recurrent_steps": 13040, "total_inference_macs": 326612880, "training_seconds": 131.3961366249714, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4360813538721003}

RECENT RESULT
hypothesis: A 128-unit gated model that retains all 16 sampled frames but performs one learned transition per ordered two-frame block will reach at least 85% validation accuracy while reducing total inference MACs to approximately 262,886,400.
change: Replace per-frame learned transitions with alternating frame buffering and eight gated block transitions over local mean/difference features; widen the state to 128 units and classify from temporal mean, maximum, and final state.
mechanism: Ordered two-frame block recurrence with mean-max-endpoint readout
evidence_used: The 100-unit 31/32/33 model reached 85.890% accuracy, showing the candidate-biased gated mechanism has useful margin. Unlike the failed 15-frame reduction, this patch preserves all 16 observations and challenges the load-bearing assumption that every observation requires a separate expensive recurrent transition.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1371255464761567360, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 43920, "peak_hidden_elements": 207872, "recurrent_macs": 260382720, "recurrent_steps": 13040, "total_inference_macs": 262886400, "training_seconds": 85.1744798750151, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.47758421166542847}

RECENT RESULT
hypothesis: A 98-unit state retaining the qualified 31/32/33 reset-update-candidate feedback allocation will achieve at least 85% validation accuracy while reducing total inference MACs from 326,612,880 to approximately 323,313,760.
change: Reduce the recurrent state, gate projections and expansions, temporal summary, and classifier input from 99 to 98 units while preserving the 16-frame schedule and training procedure.
mechanism: Candidate-biased gated-state width trim
evidence_used: The adjacent 99-unit 31/32/33 design qualified at 85.399% accuracy, and the same candidate-biased allocation enabled both the 100- and 99-unit models to qualify; a one-unit state trim is the closest supported structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1686453769209075396, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 26996, "peak_hidden_elements": 100864, "recurrent_macs": 322035840, "recurrent_steps": 13040, "total_inference_macs": 323313760, "training_seconds": 129.8985569998622, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.46683738100016775}

RECENT RESULT
hypothesis: A 98-unit 31/32/33 gated state with explicit temporal maximum pooling will recover at least 85% validation accuracy while reducing total inference MACs below the qualified 99-unit model, to approximately 323,952,720.
change: Reduce the recurrent state to 98 units and augment its mean/final-state classifier with a learned readout from the elementwise maximum recurrent activation across all 16 steps.
mechanism: Mean-max-endpoint recurrent readout
evidence_used: The 99-unit 31/32/33 model qualified at 85.399% and 326,612,880 MACs, while the adjacent 98-unit mean/final model fell to 83.681%. This tests whether preserving transient keyword evidence in the readout can recover that boundary loss for only 638,960 MACs above the failed 98-unit design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1689786681793868740, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27780, "peak_hidden_elements": 151040, "recurrent_macs": 322035840, "recurrent_steps": 13040, "total_inference_macs": 323952720, "training_seconds": 113.36673708399758, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4024184899827454}

RECENT RESULT
hypothesis: A 97-unit state with 31/32/33 reset-update-candidate feedback and mean-max-endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs to approximately 320,647,080.
change: Replace the current 101-unit recurrence with a 97-unit candidate-biased bottleneck recurrence and add elementwise temporal maximum pooling to the classifier state.
mechanism: Max-pooled candidate-biased gated-state width trim
evidence_used: The 98-unit 31/32/33 model with mean-max-endpoint readout achieved 86.871% accuracy at 323,952,720 MACs, providing enough observed margin to motivate the nearest one-unit state reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1672543960564380418, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27498, "peak_hidden_elements": 149504, "recurrent_macs": 318749760, "recurrent_steps": 13040, "total_inference_macs": 320647080, "training_seconds": 118.59717333316803, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40993053225651843}

RECENT RESULT
hypothesis: A 96-unit state with 31/32/33 reset-update-candidate feedback and mean-max-endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 320,647,080 to approximately 317,341,440.
change: Reduce the qualified 97-unit recurrent state and readout to 96 units while preserving its 16-frame schedule, candidate-biased feedback allocation, temporal maximum pooling, and training procedure.
mechanism: Max-pooled candidate-biased gated-state width trim
evidence_used: The adjacent 97-unit mean-max-endpoint design achieved 86.503% accuracy at 320,647,080 MACs, leaving a 1.503-point accuracy margin and making a one-unit state reduction the closest supported cost improvement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1655301239334892096, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27216, "peak_hidden_elements": 147968, "recurrent_macs": 315463680, "recurrent_steps": 13040, "total_inference_macs": 317341440, "training_seconds": 114.45663625001907, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.40923603268488784}

RECENT RESULT
hypothesis: A 95-unit state retaining the 31/32/33 feedback allocation and mean-max-endpoint readout will achieve at least 85% validation accuracy while reducing total inference MACs from 317,341,440 to approximately 314,035,800.
change: Reduce the recurrent state, gate projections and expansions, temporal summaries, and classifier input from 96 to 95 units while preserving the 16-frame schedule and training procedure.
mechanism: Max-pooled candidate-biased gated-state width trim
evidence_used: The adjacent 96-unit design qualified at 85.276% and 317,341,440 MACs, while the 98-, 97-, and 96-unit versions all qualified; the one-unit trim is the closest untested structural reduction and should save exactly 3,305,640 MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1638058518105403774, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 26934, "peak_hidden_elements": 146432, "recurrent_macs": 312177600, "recurrent_steps": 13040, "total_inference_macs": 314035800, "training_seconds": 133.6423793330323, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4118773665164877}



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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 291)
        self.recurrent_down = nn.Linear(97, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 97),
                nn.Linear(32, 97),
                nn.Linear(33, 97),
            ]
        )
        self.classifier = nn.Linear(291, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 97), -1.0, device=device, dtype=dtype
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
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return (
            hidden.unsqueeze(1),
            summary + hidden,
            torch.maximum(maximum, hidden),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))


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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 294)
        self.recurrent_down = nn.Linear(98, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 98),
                nn.Linear(32, 98),
                nn.Linear(33, 98),
            ]
        )
        self.classifier = nn.Linear(294, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 98), -1.0, device=device, dtype=dtype
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
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return (
            hidden.unsqueeze(1),
            summary + hidden,
            torch.maximum(maximum, hidden),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))


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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 297)
        self.recurrent_down = nn.Linear(99, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 99),
                nn.Linear(32, 99),
                nn.Linear(33, 99),
            ]
        )
        self.classifier = nn.Linear(198, 8)

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
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return hidden.unsqueeze(1), summary + hidden, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))


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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 285)
        self.recurrent_down = nn.Linear(95, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 95),
                nn.Linear(32, 95),
                nn.Linear(33, 95),
            ]
        )
        self.classifier = nn.Linear(285, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 95, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 95), -1.0, device=device, dtype=dtype
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
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return (
            hidden.unsqueeze(1),
            summary + hidden,
            torch.maximum(maximum, hidden),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))


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
