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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3893794162384286579, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38259, "peak_hidden_elements": 654336, "recurrent_macs": 739133280, "recurrent_steps": 26080, "total_inference_macs": 746487840, "training_seconds": 96.58226724993438, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3978621219564801}
prior_hypothesis: Three 47-unit GRUs with eight ordered temporal bins will retain at least 85% validation accuracy while reducing total inference MACs from 755,537,600 to approximately 746,487,840.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3968750686230025713, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34513, "peak_hidden_elements": 368640, "recurrent_macs": 757128480, "recurrent_steps": 26080, "total_inference_macs": 760857920, "training_seconds": 118.50576454098336, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.398963029545509}
prior_hypothesis: Reducing one GRU branch from 48 to 47 units will retain at least 85% validation accuracy because the four-bin temporal readout qualified with a 1.38-point margin, while lowering total inference MACs to approximately 760,857,920.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3940999087564818354, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38674, "peak_hidden_elements": 658944, "recurrent_macs": 748130880, "recurrent_steps": 26080, "total_inference_macs": 755537600, "training_seconds": 108.0177561249584, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40489840829299273}
prior_hypothesis: Increasing temporal resolution from four to eight ordered bins will recover the 47/47/48 model’s one-example accuracy shortfall, reaching at least 85% accuracy at approximately 755,537,600 total MACs.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4048910634825493045, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31805, "peak_hidden_elements": 148992, "recurrent_macs": 775280160, "recurrent_steps": 26080, "total_inference_macs": 776225560, "training_seconds": 111.75462499982677, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.47407264007381134}
prior_hypothesis: Parallel 49-, 48-, and 48-unit GRUs will correct the three-48-unit model’s one-example validation shortfall, achieving at least 85% accuracy while reducing total inference MACs from 794,546,760 to approximately 776,225,560.

## Recent verification evidence

RECENT RESULT
hypothesis: Parallel 64- and 63-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified two-64-unit model’s 842,071,040.
change: Replace the single 98-unit GRU with full-input parallel 64- and 63-unit GRUs, concatenate their temporal outputs into a 127-dimensional mean summary, and preserve the established training procedure and full-frame schedule.
mechanism: Asymmetric parallel block-diagonal gated recurrence
evidence_used: Two parallel 64-unit GRUs achieved 85.28% accuracy at 842,071,040 MACs, while reducing each branch’s spectral input failed; retaining every band and frame while removing only one hidden unit from one branch is the most precise next structural capacity probe.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4332344241781221361, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 33641, "peak_hidden_elements": 130560, "recurrent_macs": 829735200, "recurrent_steps": 26080, "total_inference_macs": 830563240, "training_seconds": 115.03466716688126, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4770930471595811}

RECENT RESULT
hypothesis: Two 64-unit GRUs with seven learned logits plus one fixed reference logit will retain at least 85% accuracy while reducing total inference MACs from 842,071,040 to approximately 841,966,720 without reducing recurrent capacity.
change: Restore the qualified parallel 64-unit GRUs and replace the eight-output classifier with an expressively equivalent seven-output classifier whose eighth logit is fixed at zero.
mechanism: Reference-logit elimination with parallel block-diagonal recurrence
evidence_used: Two parallel 64-unit GRUs achieved 85.28% accuracy and 842,071,040 MACs; unlike the failed 64+63 reduction, fixing one softmax reference logit removes classifier MACs without reducing the classifier’s representable probability distributions.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4391826528626141967, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 33967, "peak_hidden_elements": 131584, "recurrent_macs": 841236480, "recurrent_steps": 26080, "total_inference_macs": 841966720, "training_seconds": 114.0139632918872, "validation_accuracy": 0.838036809815951, "validation_cross_entropy": 0.4775640007908359}

RECENT RESULT
hypothesis: Three independent 48-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs from 842,071,040 to approximately 767,064,960.
change: Replace the two 64-unit recurrent branches with three 48-unit branches, expanding aggregate recurrent capacity from 128 to 144 units while reducing quadratic hidden-to-hidden cost.
mechanism: Three-way block-diagonal gated recurrence
evidence_used: Two 64-unit GRUs qualified at 85.28%, demonstrating that parallel block-diagonal recurrence is viable; the failed 64+63 model indicates aggregate capacity should not be reduced, motivating an additional smaller branch factorization that increases total units while lowering the sum of squared branch widths.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4001127551339436080, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31440, "peak_hidden_elements": 147968, "recurrent_macs": 766126080, "recurrent_steps": 26080, "total_inference_macs": 767064960, "training_seconds": 107.21042350004427, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.458553599726203}

RECENT RESULT
hypothesis: Three parallel 49-unit GRUs processing all 32 frames will achieve at least 85% validation accuracy at approximately 794,546,760 total inference MACs, below the qualified two-64-unit model’s 842,071,040 MACs.
change: Replace the single 98-unit GRU with three full-input 49-unit GRU branches, concatenate their outputs into a 147-dimensional temporal mean, and retain the established training procedure.
mechanism: Three-way near-boundary block-diagonal GRU
evidence_used: Three 48-unit GRUs missed qualification by only one validation example at 84.91% while using 767,064,960 MACs; adding one unit per branch is a precise capacity increase that remains substantially cheaper than the qualified two-64-unit design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4144476801797606975, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32535, "peak_hidden_elements": 151040, "recurrent_macs": 793588320, "recurrent_steps": 26080, "total_inference_macs": 794546760, "training_seconds": 105.14707025000826, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4558886849807084}

RECENT RESULT
hypothesis: Parallel 49-, 48-, and 48-unit GRUs will correct the three-48-unit model’s one-example validation shortfall, achieving at least 85% accuracy while reducing total inference MACs from 794,546,760 to approximately 776,225,560.
change: Replace the 100-unit GRU with three full-input recurrent branches of widths 49, 48, and 48, concatenating their temporal means for classification.
mechanism: Asymmetric three-way block-diagonal GRU
evidence_used: Three 48-unit GRUs reached 84.91%, one validation example below qualification, while three 49-unit GRUs reached 85.52%; enlarging only one branch is the smallest structural interpolation between those results.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4048910634825493045, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31805, "peak_hidden_elements": 148992, "recurrent_macs": 775280160, "recurrent_steps": 26080, "total_inference_macs": 776225560, "training_seconds": 111.75462499982677, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.47407264007381134}

RECENT RESULT
hypothesis: Skipping only the first log-mel frame will preserve at least 85% validation accuracy while reducing execution from 32 to 31 recurrent steps and total inference MACs from 776,225,560 to approximately 751,998,055.
change: Keep the qualified 49/48/48 GRU capacity and training procedure unchanged, but begin the causal frame schedule at index 1.
mechanism: Single-edge-frame causal truncation
evidence_used: The current full-frame model achieved 85.52% accuracy, providing a four-example margin; the failed 24- and 16-step schedules motivate testing the smallest possible temporal reduction rather than another aggressive subsampling scheme.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3922536282182603925, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 31805, "peak_hidden_elements": 148992, "recurrent_macs": 751052655, "recurrent_steps": 25265, "total_inference_macs": 751998055, "training_seconds": 100.63707212498412, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4792716465113353}

RECENT RESULT
hypothesis: Four parallel 40-unit GRUs will retain at least 85% validation accuracy by increasing aggregate recurrent width from 145 to 160, while reducing expected total inference MACs from 776,225,560 to approximately 752,147,200.
change: Replace the three 49-unit GRU branches with four 40-unit branches, concatenate their outputs into a 160-dimensional temporal mean, and preserve the full-frame training procedure.
mechanism: Four-way block-diagonal gated recurrence
evidence_used: Three 48-unit GRUs missed qualification by one example, while 49/48/48 qualified at 85.52%; four 40-unit blocks provide more aggregate units than either design while lowering the quadratic recurrent cost per step from 29,727 to 28,800 MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3923314245278156368, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31088, "peak_hidden_elements": 164352, "recurrent_macs": 751104000, "recurrent_steps": 26080, "total_inference_macs": 752147200, "training_seconds": 93.04229416605085, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.47818596026648774}

RECENT RESULT
hypothesis: Three 48-unit GRUs with separate eight-frame temporal summaries will exceed 85% accuracy while using approximately 769,881,600 MACs, below the qualified 49/48/48 model.
change: Replace the two 64-unit branches and global mean with three 48-unit branches whose outputs are pooled into four ordered causal segments before classification.
mechanism: Four-bin causal temporal-pyramid readout
evidence_used: Three 48-unit GRUs missed qualification by one validation example at 767,064,960 MACs, while 49/48/48 qualified; this tests whether global temporal averaging—not recurrent capacity—caused that narrow miss.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4015819574162198576, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34896, "peak_hidden_elements": 371200, "recurrent_macs": 766126080, "recurrent_steps": 26080, "total_inference_macs": 769881600, "training_seconds": 95.57945120800287, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4027781667884873}

RECENT RESULT
hypothesis: Reducing one GRU branch from 48 to 47 units will retain at least 85% validation accuracy because the four-bin temporal readout qualified with a 1.38-point margin, while lowering total inference MACs to approximately 760,857,920.
change: Change the three recurrent branch widths from 48/48/48 to 47/48/48 and resize the temporal-bin state and classifier from 144 to 143 features.
mechanism: Asymmetric temporal-pyramid block-diagonal recurrence
evidence_used: The current 48/48/48 temporal-pyramid model achieved 86.38% accuracy at 769,881,600 MACs, substantially above the same-width global-summary model’s 84.91%; removing one unit from only one branch is the smallest structural capacity probe that preserves the successful ordered readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3968750686230025713, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34513, "peak_hidden_elements": 368640, "recurrent_macs": 757128480, "recurrent_steps": 26080, "total_inference_macs": 760857920, "training_seconds": 118.50576454098336, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.398963029545509}

RECENT RESULT
hypothesis: A 47/47/48-unit four-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 760,857,920 to approximately 751,834,240.
change: Replace the global mean with four ordered eight-frame summaries and reduce two recurrent branches to 47 units, yielding a balanced 142-unit aggregate state.
mechanism: Balanced three-way block-diagonal GRU with temporal-pyramid readout
evidence_used: The 47/48/48 temporal-pyramid design achieved 86.63% accuracy at 760,857,920 MACs; its 1.63-point margin motivates the smallest further width reduction, while balanced branch sizes minimize recurrent MACs for the resulting aggregate width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3921681798297852850, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34130, "peak_hidden_elements": 366080, "recurrent_macs": 748130880, "recurrent_steps": 26080, "total_inference_macs": 751834240, "training_seconds": 117.96163354208693, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4149572922408215}

RECENT RESULT
hypothesis: Increasing temporal resolution from four to eight ordered bins will recover the 47/47/48 model’s one-example accuracy shortfall, reaching at least 85% accuracy at approximately 755,537,600 total MACs.
change: Use 47/47/48-unit parallel GRUs and classify eight four-frame temporal summaries instead of a global mean.
mechanism: Eight-bin causal temporal-pyramid readout
evidence_used: The four-bin 48/48/48 model improved accuracy from 84.91% with global pooling to 86.38%; the four-bin 47/47/48 model then missed qualification by only one example, so finer temporal structure is a targeted capacity increase that remains cheaper than the qualified 47/48/48 design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3940999087564818354, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38674, "peak_hidden_elements": 658944, "recurrent_macs": 748130880, "recurrent_steps": 26080, "total_inference_macs": 755537600, "training_seconds": 108.0177561249584, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.40489840829299273}

RECENT RESULT
hypothesis: Three 47-unit GRUs with eight ordered temporal bins will retain at least 85% validation accuracy while reducing total inference MACs from 755,537,600 to approximately 746,487,840.
change: Replace the 128-unit GRU with three parallel 47-unit GRUs and classify eight four-frame temporal summaries.
mechanism: Balanced three-way block-diagonal GRU with eight-bin causal temporal readout
evidence_used: The 47/47/48 eight-bin design achieved 86.13% accuracy, ten correct examples above the otherwise identical four-bin design and nine above qualification; removing one unit from its largest branch is the smallest recurrent-cost reduction and produces balanced branches.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3893794162384286579, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38259, "peak_hidden_elements": 654336, "recurrent_macs": 739133280, "recurrent_steps": 26080, "total_inference_macs": 746487840, "training_seconds": 96.58226724993438, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3978621219564801}



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
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 141, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 141, device=device, dtype=dtype)
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
    """Three asymmetric GRUs with an ordered causal temporal-pyramid readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(4 * 143, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 4, 143, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 4, device=device, dtype=dtype)
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
        bin_index = (count.to(dtype=torch.long) // 8).clamp(max=3)
        assignment = F.one_hot(bin_index[:, 0], num_classes=4).to(output.dtype)
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
        bin_indices = (positions // 8).clamp(max=3)
        assignments = F.one_hot(bin_indices, num_classes=4).to(outputs.dtype)
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

BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0


class KeywordGRU(nn.Module):
    """Three asymmetric GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 142, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 142, device=device, dtype=dtype)
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
    """Three asymmetric parallel GRUs with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 49, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(145, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden_a = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 145, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, summary, count

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
        hidden_a, hidden_b, hidden_c, summary, count = state
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
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            summary + output,
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
        hidden_a, hidden_b, hidden_c, summary, count = state
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
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            summary + outputs.sum(dim=1),
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
        _hidden_a, _hidden_b, _hidden_c, summary, count = state
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
