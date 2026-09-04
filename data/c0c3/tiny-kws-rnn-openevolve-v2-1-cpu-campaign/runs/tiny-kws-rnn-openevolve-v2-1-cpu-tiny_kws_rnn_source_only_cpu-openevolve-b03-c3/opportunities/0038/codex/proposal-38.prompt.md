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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3205547713298616558, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 35138, "peak_hidden_elements": 561152, "recurrent_macs": 608335560, "recurrent_steps": 22820, "total_inference_macs": 614542600, "training_seconds": 84.91341395792551, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4190197716461369}
prior_hypothesis: Processing frames 1–28 in seven four-step bins will retain at least 85% validation accuracy while reducing execution to 28 recurrent steps and total inference cost to approximately 614.5 million MACs.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3436828040327760136, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36226, "peak_hidden_elements": 631296, "recurrent_macs": 651788100, "recurrent_steps": 24450, "total_inference_macs": 658881860, "training_seconds": 86.76066350005567, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3904654941675853}
prior_hypothesis: The qualified 45/45/46 GRU can omit the final input frame in addition to the first while retaining at least 85% validation accuracy, reducing execution to 30 recurrent steps and approximately 658,881,860 total MACs.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2978892652713675988, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 35138, "peak_hidden_elements": 561152, "recurrent_macs": 564883020, "recurrent_steps": 21190, "total_inference_macs": 571090060, "training_seconds": 83.0646908748895, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4280375193964484}
prior_hypothesis: Processing 26 evenly distributed frames across the 30-frame interior will retain at least 85% validation accuracy while reducing total inference MACs from 592,816,330 to approximately 571,090,060.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3323500510035289851, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36226, "peak_hidden_elements": 631296, "recurrent_macs": 630061830, "recurrent_steps": 23635, "total_inference_macs": 637155590, "training_seconds": 79.59930045809597, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.39242549966449386}
prior_hypothesis: The qualified 45/45/46 GRU can omit frame 30 after already omitting frames 0 and 31, retaining at least 85% validation accuracy while reducing execution to 29 recurrent steps and approximately 637,155,590 total MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: A 46/46/47-unit eight-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 737,594,560 to approximately 728,701,280.
change: Reduce the qualified 46/47/47 architecture by one unit in a 47-unit branch, balancing widths at 46/46/47 and resizing the temporal state and classifier to 139 features.
mechanism: Balanced one-unit recurrent-width reduction
evidence_used: The 46/47/47 eight-bin design achieved 86.13% accuracy at 737,594,560 MACs; its 1.13-point margin supports the smallest further capacity probe, and balancing the three branches minimizes quadratic recurrent cost at the new aggregate width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3801016759003529601, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37441, "peak_hidden_elements": 645120, "recurrent_macs": 721451040, "recurrent_steps": 26080, "total_inference_macs": 728701280, "training_seconds": 91.7218647908885, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.3926870240755608}

RECENT RESULT
hypothesis: Three 46-unit GRUs with eight ordered temporal bins will retain at least 85% validation accuracy while reducing total inference MACs from 728,701,280 to approximately 719,808,000.
change: Reduce the remaining 47-unit branch in the qualified 46/46/47 design to 46 units and resize its eight-bin state and classifier to 138 features.
mechanism: Balanced three-way block-diagonal GRU with eight-bin temporal readout
evidence_used: The 46/46/47 eight-bin model achieved 86.38% accuracy, improving upon 46/47/47 despite its lower recurrent capacity; the 1.38-point margin supports the smallest balanced width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3754628057313151112, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37032, "peak_hidden_elements": 640512, "recurrent_macs": 712609920, "recurrent_steps": 26080, "total_inference_macs": 719808000, "training_seconds": 83.53243420901708, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.41623826290200827}

RECENT RESULT
hypothesis: A 45/46/46-unit eight-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 719,808,000 to approximately 711,071,200.
change: Install the qualified eight-bin causal readout and reduce one branch of the 46/46/46 design to 45 units, resizing aggregate state and classifier features to 137.
mechanism: Balanced three-way block-diagonal GRU with eight-bin temporal readout
evidence_used: The balanced 46/46/46 eight-bin model achieved 86.01% accuracy at 719,808,000 MACs, and each preceding one-unit branch reduction remained qualified; reducing one branch is the smallest structural cost probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3709055579112925909, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36629, "peak_hidden_elements": 635904, "recurrent_macs": 703925280, "recurrent_steps": 26080, "total_inference_macs": 711071200, "training_seconds": 93.60752349998802, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.40177793912361004}

RECENT RESULT
hypothesis: A 45/45/46-unit eight-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 711,071,200 to approximately 702,334,400.
change: Reduce one 46-unit GRU branch to 45 units and resize the aggregate temporal state and classifier from 137 to 136 features.
mechanism: Balanced one-unit recurrent-width reduction
evidence_used: The qualified 45/46/46 model achieved 86.26% accuracy, a 1.26-point margin, and every preceding one-unit branch reduction from 46/47/47 through 45/46/46 remained qualified; this is the smallest further structural cost probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3663483100912700706, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36226, "peak_hidden_elements": 631296, "recurrent_macs": 695240640, "recurrent_steps": 26080, "total_inference_macs": 702334400, "training_seconds": 99.96725729084574, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.38481914309636217}

RECENT RESULT
hypothesis: A 40-unit fast GRU feeding a 72-unit GRU every fourth frame will retain at least 85% accuracy while reducing expected total inference MACs from 702,334,400 to approximately 351,349,760.
change: Replace three independent every-frame GRUs with one fast acoustic GRU and a causally coupled slow semantic GRU; classify eight boundary readouts containing both timescales.
mechanism: Two-timescale hierarchical GRU with quarter-rate semantic updates
evidence_used: The 45/45/46 model reached 87.61% accuracy, but its three same-rate branches spend 695,240,640 recurrent MACs processing redundant inputs. The failed four-branch repartition showed that parallel width alone is insufficient, motivating learned cross-timescale composition instead.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Three 45-unit GRUs will retain at least 85% validation accuracy while reducing total inference MACs from 702,334,400 to approximately 693,597,600.
change: Reduce all recurrent branches to 45 units and resize the aggregate temporal state and classifier to 135 features.
mechanism: Balanced three-way block-diagonal GRU with eight-bin temporal readout
evidence_used: The qualified 45/45/46 model achieved 87.61% accuracy at 702,334,400 MACs, the strongest observed margin; reducing its remaining 46-unit branch is the smallest further structural cost probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3617910622712475503, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35823, "peak_hidden_elements": 626688, "recurrent_macs": 686556000, "recurrent_steps": 26080, "total_inference_macs": 693597600, "training_seconds": 112.85352124995552, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4059344426254553}

RECENT RESULT
hypothesis: The qualified 45/45/46 GRU model can omit the first input frame while retaining at least 85% validation accuracy, reducing execution to 31 recurrent steps and approximately 680,608,130 total MACs.
change: Restore the high-margin 45/45/46 eight-bin model and causally process frames 1–31 instead of all 32 frames.
mechanism: Boundary-frame pruning with a high-margin recurrent backbone
evidence_used: The 45/45/46 design achieved 87.61% accuracy at 702,334,400 MACs—the strongest observed margin—so removing one boundary frame is a conservative step-reduction probe with more accuracy headroom than the 85.89% three-45-unit model.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3550155570620230421, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36226, "peak_hidden_elements": 631296, "recurrent_macs": 673514370, "recurrent_steps": 25265, "total_inference_macs": 680608130, "training_seconds": 111.24788216594607, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.3902458986621693}

RECENT RESULT
hypothesis: The qualified 45/45/46 GRU can omit the final input frame in addition to the first while retaining at least 85% validation accuracy, reducing execution to 30 recurrent steps and approximately 658,881,860 total MACs.
change: Process only frames 1–30 for 32-frame inputs, preserving the current recurrent backbone and eight-bin readout.
mechanism: Symmetric boundary-frame pruning
evidence_used: The current model retained 86.75% accuracy after removing the first boundary frame, leaving a 1.75-point margin; removing the opposite boundary is the smallest additional step-reduction probe and saves exactly one full recurrent update per example.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3436828040327760136, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36226, "peak_hidden_elements": 631296, "recurrent_macs": 651788100, "recurrent_steps": 24450, "total_inference_macs": 658881860, "training_seconds": 86.76066350005567, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3904654941675853}

RECENT RESULT
hypothesis: The qualified 45/45/46 GRU can omit frame 30 after already omitting frames 0 and 31, retaining at least 85% validation accuracy while reducing execution to 29 recurrent steps and approximately 637,155,590 total MACs.
change: Restore the high-margin 45/45/46 eight-bin backbone and process only frames 1–29 for 32-frame inputs.
mechanism: Progressive trailing-boundary frame pruning
evidence_used: The 30-step 45/45/46 model achieved 86.87% accuracy at 658,881,860 MACs, and removing the final frame slightly improved accuracy over the 31-step result; this motivates pruning the adjacent trailing frame while preserving 1.87 points of observed margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3323500510035289851, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36226, "peak_hidden_elements": 631296, "recurrent_macs": 630061830, "recurrent_steps": 23635, "total_inference_macs": 637155590, "training_seconds": 79.59930045809597, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.39242549966449386}

RECENT RESULT
hypothesis: Processing frames 1–28 in seven four-step bins will retain at least 85% validation accuracy while reducing execution to 28 recurrent steps and total inference cost to approximately 614.5 million MACs.
change: Prune frames 0 and 29–31, replace the eight-bin state/readout with seven complete four-step bins, and resize the classifier accordingly.
mechanism: Seven-bin trailing-boundary pruning
evidence_used: The 29-step 45/45/46 model achieved 87.12% accuracy at 637,155,590 MACs; its 2.12-point margin supports removing the adjacent trailing frame, whose removal also makes seven complete bins more structurally efficient than retaining an empty eighth bin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3205547713298616558, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 35138, "peak_hidden_elements": 561152, "recurrent_macs": 608335560, "recurrent_steps": 22820, "total_inference_macs": 614542600, "training_seconds": 84.91341395792551, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4190197716461369}

RECENT RESULT
hypothesis: Processing frames 1–27 in seven causal temporal bins will retain at least 85% validation accuracy while reducing execution from 28 to 27 recurrent steps and total inference MACs from 614,542,600 to approximately 592,816,330.
change: Restore the qualified 45/45/46 recurrent backbone, resize the temporal readout to seven bins, and schedule only frames 1–27 for standard 32-frame inputs.
mechanism: Seven-bin progressive trailing-frame pruning
evidence_used: The 28-step seven-bin 45/45/46 design achieved 86.38% accuracy at 614,542,600 MACs; its 1.38-point margin supports the smallest additional trailing-boundary pruning probe while preserving the same backbone and readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3092220183006146273, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 35138, "peak_hidden_elements": 561152, "recurrent_macs": 586609290, "recurrent_steps": 22005, "total_inference_macs": 592816330, "training_seconds": 87.42245399998501, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42537401819521664}

RECENT RESULT
hypothesis: Processing 26 evenly distributed frames across the 30-frame interior will retain at least 85% validation accuracy while reducing total inference MACs from 592,816,330 to approximately 571,090,060.
change: Replace contiguous frames 1–27 with 26 uniformly spaced indices spanning frames 1 through 30, preserving seven causal readout bins while eliminating one recurrent step.
mechanism: Uniform core-frame subsampling with full-utterance coverage
evidence_used: The current 27-step model achieved 85.77% accuracy, but progressive pruning discarded the utterance tail; distributing one fewer step across the full interior tests whether temporal coverage can preserve accuracy more efficiently than further trailing-frame removal.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2978892652713675988, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 35138, "peak_hidden_elements": 561152, "recurrent_macs": 564883020, "recurrent_steps": 21190, "total_inference_macs": 571090060, "training_seconds": 83.0646908748895, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4280375193964484}



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
    """Three balanced GRUs with a seven-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(7 * 136, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 7, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 7, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        output = torch.cat(
            (output_a[:, 0, :], output_b[:, 0, :], output_c[:, 0, :]),
            dim=-1,
        )
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=6)
        assignment = F.one_hot(bin_index[:, 0], num_classes=7).to(output.dtype)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + assignment.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + assignment,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b, outputs_c), dim=-1)
        positions = count.to(dtype=torch.long) + torch.arange(
            frames.shape[1], device=frames.device
        ).view(1, -1)
        bin_indices = (positions // 4).clamp(max=6)
        assignments = F.one_hot(bin_indices, num_classes=7).to(outputs.dtype)
        bin_sums = (
            outputs.unsqueeze(2) * assignments.unsqueeze(-1)
        ).sum(dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + bin_sums,
            bin_counts + assignments.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(1, available_frames - 3))


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
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 136, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        output = torch.cat(
            (output_a[:, 0, :], output_b[:, 0, :], output_c[:, 0, :]),
            dim=-1,
        )
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(bin_index[:, 0], num_classes=8).to(output.dtype)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + assignment.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + assignment,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b, outputs_c), dim=-1)
        positions = count.to(dtype=torch.long) + torch.arange(
            frames.shape[1], device=frames.device
        ).view(1, -1)
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(bin_indices, num_classes=8).to(outputs.dtype)
        bin_sums = (
            outputs.unsqueeze(2) * assignments.unsqueeze(-1)
        ).sum(dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + bin_sums,
            bin_counts + assignments.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(1, available_frames - 1))


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
    """Three balanced GRUs with a seven-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(7 * 136, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 7, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 7, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        output = torch.cat(
            (output_a[:, 0, :], output_b[:, 0, :], output_c[:, 0, :]),
            dim=-1,
        )
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=6)
        assignment = F.one_hot(bin_index[:, 0], num_classes=7).to(output.dtype)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + assignment.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + assignment,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b, outputs_c), dim=-1)
        positions = count.to(dtype=torch.long) + torch.arange(
            frames.shape[1], device=frames.device
        ).view(1, -1)
        bin_indices = (positions // 4).clamp(max=6)
        assignments = F.one_hot(bin_indices, num_classes=7).to(outputs.dtype)
        bin_sums = (
            outputs.unsqueeze(2) * assignments.unsqueeze(-1)
        ).sum(dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + bin_sums,
            bin_counts + assignments.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 7:
            return list(range(available_frames))
        steps = available_frames - 6
        span = available_frames - 3
        return [
            1 + (index * span + (steps - 2) // 2) // (steps - 1)
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

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 136, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        output = torch.cat(
            (output_a[:, 0, :], output_b[:, 0, :], output_c[:, 0, :]),
            dim=-1,
        )
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(bin_index[:, 0], num_classes=8).to(output.dtype)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + assignment.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + assignment,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b, outputs_c), dim=-1)
        positions = count.to(dtype=torch.long) + torch.arange(
            frames.shape[1], device=frames.device
        ).view(1, -1)
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(bin_indices, num_classes=8).to(outputs.dtype)
        bin_sums = (
            outputs.unsqueeze(2) * assignments.unsqueeze(-1)
        ).sum(dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + bin_sums,
            bin_counts + assignments.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(1, available_frames - 2))


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
