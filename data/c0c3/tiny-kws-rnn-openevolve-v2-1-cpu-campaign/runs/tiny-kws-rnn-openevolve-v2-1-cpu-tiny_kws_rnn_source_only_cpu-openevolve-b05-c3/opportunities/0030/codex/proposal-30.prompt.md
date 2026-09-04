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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4293335559836322196, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 53096, "peak_hidden_elements": 123392, "recurrent_macs": 821520000, "recurrent_steps": 16300, "total_inference_macs": 823084800, "training_seconds": 97.80494499998167, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.42089620016835216}
prior_hypothesis: A 120-unit GRU with the qualified 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 835,860,740 to approximately 823,084,800.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3904201010905741252, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 48392, "peak_hidden_elements": 117248, "recurrent_macs": 746996400, "recurrent_steps": 16300, "total_inference_macs": 748482960, "training_seconds": 82.15217825002037, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.3901598690477617}
prior_hypothesis: A 114-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 760,672,100 to approximately 748,482,960.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3967781419857473561, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 49161, "peak_hidden_elements": 118272, "recurrent_macs": 759172500, "recurrent_steps": 16300, "total_inference_macs": 760672100, "training_seconds": 85.1479240830522, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.39398240165476417}
prior_hypothesis: A 115-unit GRU trained with batch size 64 will recover at least 85% validation accuracy through roughly twice as many optimizer updates while retaining the previously measured 760,672,100 inference MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: A 121-unit GRU with the qualified 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 848,734,480 to approximately 835,860,740.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 126 to 121 units while preserving the proven schedule and training procedure.
mechanism: One-unit width-boundary reduction of the dual-readout GRU
evidence_used: The adjacent 122-unit design qualified at 85.40% with 848,734,480 MACs, while the 123-unit design reached 86.01%; testing 121 units is the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4359976806876129341, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 53901, "peak_hidden_elements": 124416, "recurrent_macs": 834282900, "recurrent_steps": 16300, "total_inference_macs": 835860740, "training_seconds": 87.02054087491706, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4305505764265002}

RECENT RESULT
hypothesis: A 120-unit GRU with the qualified 20-frame dual readout will retain at least 85% validation accuracy while reducing total inference MACs from 835,860,740 to approximately 823,084,800.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 121 to 120 units while preserving the proven schedule and training procedure.
mechanism: One-unit width-boundary reduction of the qualified dual-readout GRU
evidence_used: The adjacent 121-unit design achieved 85.40% accuracy at 835,860,740 MACs, and every tested 121–126-unit dual-readout design qualified; testing 120 units is the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4293335559836322196, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 53096, "peak_hidden_elements": 123392, "recurrent_macs": 821520000, "recurrent_steps": 16300, "total_inference_macs": 823084800, "training_seconds": 97.80494499998167, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.42089620016835216}

RECENT RESULT
hypothesis: A 112-unit full-rate GRU augmented by a 40-unit GRU that processes ten adjacent-frame averages will retain at least 85% accuracy while reducing total validation inference MACs from 823,084,800 to approximately 783,599,680.
change: Replace the monolithic 122-unit recurrence with fine- and coarse-timescale recurrent states, then fuse their mean and endpoint representations for prediction.
mechanism: Paired-frame multirate recurrent pyramid
evidence_used: The 120-unit, 20-frame dual-readout model reached 86.26% at 823,084,800 MACs. This tests the load-bearing assumption that all recurrent capacity must execute on every frame: most capacity remains full-rate, while a structurally cheaper branch receives all frames through pairwise temporal pooling.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 119-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 823,084,800 to approximately 810,406,660.
change: Reduce the recurrent hidden state and both classifier input widths from 123 to 119 units while preserving the proven schedule and training procedure.
mechanism: Adjacent width-boundary reduction of the dual-readout GRU
evidence_used: The 120-unit dual-readout model achieved 86.26% accuracy at 823,084,800 MACs, and every tested width from 120 through 126 qualified; testing the adjacent 119-unit width is the most informative remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4227204452477860857, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 52297, "peak_hidden_elements": 122368, "recurrent_macs": 808854900, "recurrent_steps": 16300, "total_inference_macs": 810406660, "training_seconds": 75.62333866721019, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4197836846661714}

RECENT RESULT
hypothesis: A 118-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 810,406,660 to approximately 797,826,320.
change: Reduce the recurrent hidden state and both classifier input widths from 119 to 118 units while preserving the proven frame schedule and training procedure.
mechanism: Adjacent width-boundary reduction of the dual-readout GRU
evidence_used: The adjacent 119-unit design achieved 85.89% accuracy, and every tested dual-readout width from 119 through 126 qualified; 118 units is therefore the most informative next cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4161583484800745324, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 51504, "peak_hidden_elements": 121344, "recurrent_macs": 796287600, "recurrent_steps": 16300, "total_inference_macs": 797826320, "training_seconds": 77.1984111671336, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4225021268692485}

RECENT RESULT
hypothesis: The qualified 120-unit dual-readout GRU will retain at least 85% validation accuracy on 19 uniformly distributed frames while reducing total inference MACs from 823,084,800 to approximately 782,008,800 and recurrent steps from 20 to 19 per example.
change: Reduce the uniform causal frame schedule from 20 frames to 19 while preserving recurrent capacity, readouts, and training procedure.
mechanism: Nineteen-frame endpoint-aware recurrent readout
evidence_used: The current 120-unit, 20-frame design achieved 86.26% accuracy, providing a 1.26-point margin; removing one of its uniformly distributed frames yields a larger structural cost reduction than the riskier width reduction below the barely qualifying 118-unit design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4079076893589585381, "maximum_recurrent_steps": 19, "mean_recurrent_steps": 19.0, "median_recurrent_steps": 19, "optimizer_steps": 397, "p95_recurrent_steps": 19, "parameters": 53096, "peak_hidden_elements": 123392, "recurrent_macs": 780444000, "recurrent_steps": 15485, "total_inference_macs": 782008800, "training_seconds": 59.54352758289315, "validation_accuracy": 0.838036809815951, "validation_cross_entropy": 0.45935937612334643}

RECENT RESULT
hypothesis: A 117-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 797,826,320 to approximately 785,343,780.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 122 to 117 units while preserving the qualified frame schedule and training procedure.
mechanism: Adjacent width-boundary reduction of the dual-readout GRU
evidence_used: The adjacent 118-unit design qualified at 85.03% with 797,826,320 MACs, and every tested 118–126-unit dual-readout model qualified; testing 117 units is the most direct remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4096472656804975597, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 50717, "peak_hidden_elements": 120320, "recurrent_macs": 783818100, "recurrent_steps": 16300, "total_inference_macs": 785343780, "training_seconds": 66.99786174995825, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4273368133357697}

RECENT RESULT
hypothesis: A 116-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 785,343,780 to approximately 772,959,040.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 117 to 116 units while preserving the qualified schedule and training procedure.
mechanism: Adjacent width-boundary reduction of the dual-readout GRU
evidence_used: The adjacent 117-unit design achieved 85.52% accuracy at 785,343,780 MACs, and every tested 117–126-unit dual-readout design qualified; testing 116 units is the most direct remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4031871968490551676, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 49936, "peak_hidden_elements": 119296, "recurrent_macs": 771446400, "recurrent_steps": 16300, "total_inference_macs": 772959040, "training_seconds": 53.945663374848664, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42321504932239745}

RECENT RESULT
hypothesis: A 115-unit GRU with the qualified 20-frame mean/endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 772,959,040 to approximately 760,672,100.
change: Reduce the recurrent hidden state, temporal summary, and both classifier input widths from 118 to 115 units while preserving the qualified schedule and training procedure.
mechanism: Adjacent width-boundary reduction of the dual-readout GRU
evidence_used: The adjacent 116-unit design achieved 85.77% accuracy at 772,959,040 MACs, and every tested width from 116 through 126 qualified; testing 115 units is the most direct remaining cost-boundary probe.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3967781419857473561, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 49161, "peak_hidden_elements": 118272, "recurrent_macs": 759172500, "recurrent_steps": 16300, "total_inference_macs": 760672100, "training_seconds": 58.56240000016987, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.44506098273341643}

RECENT RESULT
hypothesis: A 116-unit GRU with a rank-4 endpoint correction will retain at least 85% validation accuracy while reducing total inference MACs from the qualified 772,959,040 to approximately 772,606,960.
change: Reduce the current GRU from 120 to the proven 116 units and factor the endpoint classifier through a four-unit linear bottleneck, preserving its zero-initialized residual behavior.
mechanism: Low-rank endpoint residual readout
evidence_used: The 116-unit full endpoint-readout model achieved 85.77% accuracy, whereas 115 units failed at 84.17%; retaining the proven recurrent width while compressing only the complementary endpoint correction is the lower-risk next structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4030035465637706364, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 397, "p95_recurrent_steps": 20, "parameters": 49504, "peak_hidden_elements": 119296, "recurrent_macs": 771446400, "recurrent_steps": 16300, "total_inference_macs": 772606960, "training_seconds": 52.88176641683094, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4605172186541411}

RECENT RESULT
hypothesis: A 115-unit GRU trained with batch size 64 will recover at least 85% validation accuracy through roughly twice as many optimizer updates while retaining the previously measured 760,672,100 inference MACs.
change: Reduce the qualified recurrent width from 116 to 115 units and halve the training batch size without changing inference structure, augmentation, loss, or learning-rate schedule.
mechanism: Smaller-batch optimization rescue at the recurrent-width boundary
evidence_used: The 116-unit model qualified at 85.77%, while the adjacent 115-unit model reached 84.17% using batch size 128; retraining that exact lower-cost width with more optimizer updates directly tests whether the narrow miss was optimization-limited rather than capacity-limited.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3967781419857473561, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 49161, "peak_hidden_elements": 118272, "recurrent_macs": 759172500, "recurrent_steps": 16300, "total_inference_macs": 760672100, "training_seconds": 85.1479240830522, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.39398240165476417}

RECENT RESULT
hypothesis: A 114-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 760,672,100 to approximately 748,482,960.
change: Reduce the recurrent and readout widths from 118 to 114 units and halve the batch size to provide roughly twice as many optimizer updates.
mechanism: Smaller-batch recurrent-width boundary probe
evidence_used: The 115-unit model improved from 84.17% with batch size 128 to 86.87% with batch size 64 at identical inference cost, leaving a 1.87-point margin and motivating the adjacent 114-unit boundary test under the proven smaller-batch procedure.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3904201010905741252, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 48392, "peak_hidden_elements": 117248, "recurrent_macs": 746996400, "recurrent_steps": 16300, "total_inference_macs": 748482960, "training_seconds": 82.15217825002037, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.3901598690477617}

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
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
        self.endpoint_classifier = nn.Linear(120, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

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
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
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

BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 114, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
        self.endpoint_classifier = nn.Linear(114, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 114, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 114, device=device, dtype=dtype)
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
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
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
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 115, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(115, 8)
        self.endpoint_classifier = nn.Linear(115, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 115, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 115, device=device, dtype=dtype)
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
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
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
