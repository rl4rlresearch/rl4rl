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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 90.91042829095386, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}
prior_hypothesis: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the successful 104-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Training and inference on 16 uniformly spaced causal frames will retain at least 85% validation accuracy because adjacent log-mel frames are highly redundant, while halving recurrent MACs and executed steps from the 87.24%-accurate baseline.
change: Process every second input frame, selecting the odd-indexed frames so the schedule spans through the final frame.
mechanism: Two-to-one causal frame decimation
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per clip, leaving a 2.24-point accuracy margin for temporal downsampling.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 66.97952787508257, "validation_accuracy": 0.8159509202453987, "validation_cross_entropy": 0.5274510295844517}

RECENT RESULT
hypothesis: Processing 24 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 25% versus the 32-step baseline.
change: Use 24 unique, approximately uniform causal frame indices spanning the complete recording, including the first and final frames.
mechanism: Three-of-four temporal sampling with endpoint preservation
evidence_used: The 32-step model achieved 87.24% accuracy, while aggressive 16-step decimation fell to 81.60%; testing the midpoint isolates whether moderate temporal reduction preserves the required accuracy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 56.38068737508729, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.5135083157592024}

RECENT RESULT
hypothesis: Processing 28 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 12.5% versus the 32-step baseline.
change: Select 28 unique, approximately uniform causal frame indices spanning the first through final input frames.
mechanism: Seven-of-eight temporal sampling with endpoint preservation
evidence_used: The 32-step model achieved 87.24% accuracy, while 24 steps achieved 83.93%; 28 steps is the most informative midpoint for locating the minimum schedule that clears 85%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 109.27272541704588, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4741032419029189}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will preserve at least 85% accuracy while reducing recurrent MACs by approximately 22% versus the successful 128-unit baseline.
change: Reduce the GRU hidden state and classifier input from 128 to 112 units while retaining the complete causal frame schedule.
mechanism: Full-rate recurrent width reduction
evidence_used: Temporal reduction was costly—28 frames achieved only 83.93%—while the 32-frame baseline achieved 87.24%; preserving temporal evidence and reducing recurrent width targets compute without repeating the demonstrated failure mode.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 146.85255179204978, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42283919632800515}

RECENT RESULT
hypothesis: A 108-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 6.5% versus the successful 112-unit model.
change: Reduce the GRU hidden state and classifier input from 112 to 108 units while preserving the full causal frame schedule and training procedure.
mechanism: Conservative full-rate recurrent width reduction
evidence_used: The 112-unit full-rate GRU achieved 85.77% accuracy, whereas reducing temporal coverage to 28 frames failed at 83.93%; this supports preserving all frames and making a smaller structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5645409772253230232, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 1081589760, "recurrent_steps": 26080, "total_inference_macs": 1082293920, "training_seconds": 80.48066208302043, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4493955928123802}

RECENT RESULT
hypothesis: A 104-unit full-rate GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.7% versus the successful 108-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input from 108 to 104 units while preserving all 32 causal frames and the established training procedure.
mechanism: Conservative recurrent width reduction
evidence_used: The 108-unit full-rate model achieved 86.01% accuracy, while temporal subsampling repeatedly failed; the remaining accuracy margin supports another small structural width reduction without discarding temporal evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 69.56130520813167, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}

RECENT RESULT
hypothesis: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the successful 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input from 104 to 100 units while preserving the full causal schedule and established training procedure.
mechanism: Conservative full-rate recurrent width reduction
evidence_used: The 104-unit full-rate GRU achieved 85.52% accuracy, and prior 112- and 108-unit models also cleared 85%; temporal subsampling repeatedly failed, so another small width reduction is the best-supported route to lower compute without discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 90.91042829095386, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}

RECENT RESULT
hypothesis: Replacing the eight learned logits with seven learned relative logits and one fixed reference logit will retain at least 85% accuracy while eliminating 100 classifier MACs per example and 101 parameters, because softmax is invariant to a shared logit offset.
change: Change the classifier to produce seven logits and append a fixed zero as the eighth class logit.
mechanism: Reference-class softmax head
evidence_used: The 100-unit full-rate GRU barely cleared the requirement at 85.03%, making further recurrent reductions risky; preserving its recurrent computation while removing the mathematically redundant eighth affine score is the lowest-risk structural cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900316758335593927, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37347, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939450500, "training_seconds": 96.30393720907159, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4956174323895226}



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
