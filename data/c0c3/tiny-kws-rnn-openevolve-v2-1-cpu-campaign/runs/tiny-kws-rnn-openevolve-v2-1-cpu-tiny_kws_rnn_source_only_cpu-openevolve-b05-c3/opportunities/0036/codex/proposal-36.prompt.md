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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3841130741635354749, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 47629, "peak_hidden_elements": 116224, "recurrent_macs": 734918100, "recurrent_steps": 16300, "total_inference_macs": 736391620, "training_seconds": 85.78211450017989, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.37745496129697087}
prior_hypothesis: A 113-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 748,482,960 to approximately 736,391,620.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3654980771912270076, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 45376, "peak_hidden_elements": 113152, "recurrent_macs": 699270000, "recurrent_steps": 16300, "total_inference_macs": 700704400, "training_seconds": 77.22509833402, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4034261036504266}
prior_hypothesis: A 110-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 712,502,340 to approximately 700,704,400.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3593951061367266797, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 44637, "peak_hidden_elements": 112128, "recurrent_macs": 687582900, "recurrent_steps": 16300, "total_inference_macs": 689004260, "training_seconds": 78.46343158301897, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.3816709647149396}
prior_hypothesis: A 109-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 724,398,080 to approximately 689,004,260.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 180-unit recurrent cell with one low-rank state-aware update gate will retain at least 85% accuracy under the proven batch-64 training regime while reducing estimated validation inference MACs from 748,482,960 to approximately 694,771,200.
change: Replace the 120-unit three-gate GRU with a wider 180-unit tanh recurrence whose update gate uses input projection plus an 8-unit recurrent bottleneck; preserve the qualified 20-frame mean/endpoint readout and switch to batch size 64.
mechanism: Bottleneck-gated highway recurrence
evidence_used: The 114-unit GRU reached 86.01% at 748,482,960 MACs with batch size 64. This tests the load-bearing assumption that three full recurrent gate matrices are necessary: the alternative spends fewer MACs on gating while providing substantially more recurrent state capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3624032297910625284, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 45784, "peak_hidden_elements": 184832, "recurrent_macs": 692424000, "recurrent_steps": 16300, "total_inference_macs": 694771200, "training_seconds": 130.6935564170126, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4184940548762222}

RECENT RESULT
hypothesis: A 113-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 748,482,960 to approximately 736,391,620.
change: Reduce the recurrent state and both readout widths from 114 to 113 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 114-unit design achieved 86.01% accuracy at 748,482,960 MACs after the 115-unit design achieved 86.87%; this makes 113 units the most informative remaining width-boundary test under the proven batch-64 regime.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3841130741635354749, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 47629, "peak_hidden_elements": 116224, "recurrent_macs": 734918100, "recurrent_steps": 16300, "total_inference_macs": 736391620, "training_seconds": 85.78211450017989, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.37745496129697087}

RECENT RESULT
hypothesis: A 112-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from the qualified 736,391,620 to approximately 724,398,080.
change: Reduce the recurrent state and both readout widths from 115 to 112 units while preserving the qualified 20-frame schedule and batch-64 training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 113-unit design achieved 86.63% accuracy at 736,391,620 MACs, leaving a 1.63-point margin; testing 112 units is the most informative remaining width-boundary probe under the proven batch-64 regime.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3778570612046314052, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 46872, "peak_hidden_elements": 115200, "recurrent_macs": 722937600, "recurrent_steps": 16300, "total_inference_macs": 724398080, "training_seconds": 81.19652283401228, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.3892794462800757}

RECENT RESULT
hypothesis: A 111-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 724,398,080 to approximately 712,502,340.
change: Reduce the recurrent state and both readout widths from 120 to 111 units and use the proven batch-64 training procedure, preserving the qualified 20-frame schedule and all other training choices.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 112-unit design achieved 87.61% accuracy at 724,398,080 MACs, leaving a 2.61-point margin; testing 111 units is the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3716520622138619161, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 46121, "peak_hidden_elements": 114176, "recurrent_macs": 711054900, "recurrent_steps": 16300, "total_inference_macs": 712502340, "training_seconds": 77.19785395800136, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.38025378829862444}

RECENT RESULT
hypothesis: A 110-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 712,502,340 to approximately 700,704,400.
change: Reduce the recurrent state, temporal summary, and both readout widths from 111 to 110 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 111-unit design achieved 86.75% accuracy at 712,502,340 MACs, leaving a 1.75-point margin after every tested 111–114-unit batch-64 GRU qualified; 110 units is therefore the most informative remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3654980771912270076, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 45376, "peak_hidden_elements": 113152, "recurrent_macs": 699270000, "recurrent_steps": 16300, "total_inference_macs": 700704400, "training_seconds": 77.22509833402, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4034261036504266}

RECENT RESULT
hypothesis: A 109-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 724,398,080 to approximately 689,004,260.
change: Reduce the recurrent state, temporal summary, and both readout widths from 112 to 109 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch recurrent-width boundary probe
evidence_used: The 110-unit batch-64 design qualified at 85.77% and 700,704,400 MACs; testing the adjacent 109-unit width is the most informative remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3593951061367266797, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 44637, "peak_hidden_elements": 112128, "recurrent_macs": 687582900, "recurrent_steps": 16300, "total_inference_macs": 689004260, "training_seconds": 78.46343158301897, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.3816709647149396}



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
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 113, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(113, 8)
        self.endpoint_classifier = nn.Linear(113, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 113, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 113, device=device, dtype=dtype)
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
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)
        self.endpoint_classifier = nn.Linear(110, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
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
        self.gru = nn.GRU(20, 109, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(109, 8)
        self.endpoint_classifier = nn.Linear(109, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 109, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 109, device=device, dtype=dtype)
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
