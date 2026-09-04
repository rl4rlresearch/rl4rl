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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 95.61552495881915, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42309365301775786}
prior_hypothesis: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing exact dense inference MACs relative to the verified 92-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing every second frame will preserve at least 85% validation accuracy because adjacent log-mel frames are redundant, while halving recurrent MACs and executed recurrent steps.
change: Change the frame schedule from all 32 frames to the 16 odd-indexed frames spanning the full recording.
mechanism: Uniform causal frame decimation
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps, leaving a 2.24-point accuracy margin for temporal decimation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 46.12407416687347, "validation_accuracy": 0.8171779141104294, "validation_cross_entropy": 0.5212684678153757}

RECENT RESULT
hypothesis: A 112-unit GRU will retain at least 85% accuracy while reducing recurrent MACs by about 22%, because it preserves all 32 frames whose decimation caused accuracy to fall to 81.72%.
change: Reduce the GRU, recurrent state, temporal summary, and classifier input width from 128 to 112 while retaining the full frame schedule.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 128-unit, 32-step model achieved 87.24%, whereas reducing temporal coverage to 16 steps achieved only 81.72%; this motivates reducing recurrent capacity instead of discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 73.4652672498487, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4392715641325968}

RECENT RESULT
hypothesis: A 104-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by roughly 12% relative to the verified 112-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 112 to 104 while preserving all 32 causal frames and the proven training procedure.
mechanism: Conservative recurrent-width reduction
evidence_used: Reducing width from 128 to 112 preserved 86.26% accuracy and cut recurrent MACs to 1,156,700,160, while temporal decimation failed at 81.72%; the remaining 1.26-point margin supports a smaller, conservative width reduction without discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 96.02999812504277, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44775579721649733}

RECENT RESULT
hypothesis: Worst-case logit-margin certification will preserve the 104-unit GRU’s full-sequence class decisions while reducing mean recurrent steps below 32 and therefore lowering exact inference MACs.
change: Add an `exit_mask` that exits only when the current winning class cannot be overturned by any possible remaining GRU outputs, using the GRU’s bounded hidden range and classifier weights.
mechanism: Certified adaptive early exit
evidence_used: The 104-unit full-resolution model barely met the target at 85.03%, while fixed 16-frame decimation fell to 81.72%; this motivates preserving all steps for uncertain examples and skipping computation only when the final class is certified unchanged.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Omitting only the earliest frame will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125% relative to the verified 104-unit, 32-step GRU.
change: Process frames 1–31, preserving the latest 31 causal frames and the proven 104-unit architecture and training procedure.
mechanism: Single-edge-frame causal pruning
evidence_used: Aggressive 16-frame decimation failed at 81.72%, while all 32 frames with 104 units reached 85.03%; removing just one boundary frame is the smallest temporal reduction and directly tests whether near-full resolution preserves the threshold.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5102077002226359857, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 977452320, "recurrent_steps": 25265, "total_inference_macs": 978130400, "training_seconds": 99.17054420802742, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.44433152836524636}

RECENT RESULT
hypothesis: Concatenating the final GRU state with the temporal mean will recover the 0.34-point accuracy deficit of the 31-frame model and reach at least 85%, while retaining nearly all of its 3.125% recurrent-MAC reduction.
change: Process frames 1–31 and classify from a learned combination of the 104-dimensional temporal mean and final recurrent state.
mechanism: Dual temporal-mean and final-state readout
evidence_used: The 31-frame model reached 84.66% with lower cross-entropy than the qualifying 32-frame model, so a richer low-cost readout is a targeted way to recover the small accuracy gap without restoring the omitted recurrent step.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5105613970683691569, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 41024, "peak_hidden_elements": 107008, "recurrent_macs": 977452320, "recurrent_steps": 25265, "total_inference_macs": 978808480, "training_seconds": 87.27757845888846, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.3932182546042226}

RECENT RESULT
hypothesis: A 96-unit GRU with the proven 31-frame temporal-mean/final-state readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 14% versus the verified 104-unit design.
change: Reduce the GRU hidden state, online summary, and classifier input width from 104/208 to 96/192 while preserving the 31-frame schedule and training procedure.
mechanism: Dual-readout recurrent width reduction
evidence_used: The 104-unit, 31-step dual-readout model achieved 87.73% accuracy—2.73 points above the requirement—whereas the earlier mean-only 104-unit model achieved 85.03%; this margin supports testing a structural width reduction without discarding additional frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4409239296334579393, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 844053120, "recurrent_steps": 25265, "total_inference_macs": 845304960, "training_seconds": 99.53542054118589, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.40978392618565473}

RECENT RESULT
hypothesis: A 92-unit GRU processing all 32 frames with the dual mean/final-state readout will retain at least 85% accuracy while reducing exact MACs versus the verified 96-unit, 31-step model.
change: Reduce recurrent width from 96 to 92 and restore frame 0, trading one additional step for a structurally cheaper recurrent matrix at every step.
mechanism: Width-for-temporal-coverage rebalancing
evidence_used: The 104-unit dual-readout model reached 87.73%, showing the readout benefits from near-full temporal coverage; the 96-unit model still met 85.03%, while prior single-frame pruning caused a measurable accuracy loss. Since 92 units over 32 steps require fewer recurrent MACs than 96 units over 31 steps, restoring the omitted frame may offset the smaller state without sacrificing the primary cost objective.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4211441137302266104, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807384640, "training_seconds": 90.52424379205331, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.3870472633034174}

RECENT RESULT
hypothesis: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing exact dense inference MACs relative to the verified 92-unit model.
change: Reduce the recurrent hidden state and temporal summary from 92 to 88 units, and reduce the dual-readout classifier input from 184 to 176 features.
mechanism: Conservative dual-readout width reduction
evidence_used: The 92-unit, 32-step dual-readout model achieved 87.12% accuracy, 2.12 points above the requirement, while already outperforming the 96-unit, 31-step model in both accuracy and MACs; this supports testing a modest structural width reduction without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 95.61552495881915, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42309365301775786}



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
