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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 108.62356691597961, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4702767307772958}
prior_hypothesis: A 98-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.84% versus the verified 99-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 28 uniformly distributed frames will reduce recurrent MACs and steps by 12.5% while retaining at least 85% validation accuracy.
change: Replace the full 32-frame schedule with 28 unique increasing indices spanning the first through final frame.
mechanism: Moderate uniform causal frame subsampling
evidence_used: The 32-step model achieved 87.24% accuracy, while the more aggressive 24-step schedule fell to 83.93%; 28 steps tests the midpoint of this observed accuracy–compute boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 64.56146404100582, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4928505365102569}

RECENT RESULT
hypothesis: Reducing the GRU width from 128 to 120 will preserve at least 85% validation accuracy while lowering recurrent MACs by approximately 11.3%, because the failed 24- and 28-step trials indicate that retaining all 32 temporal observations is more valuable than retaining the full hidden width.
change: Use a 120-unit GRU and matching classifier/state tensors while keeping the complete 32-frame schedule and training procedure unchanged.
mechanism: Recurrent-width compression with full temporal coverage
evidence_used: The 128-unit, 32-step model reached 87.24% accuracy, whereas temporal subsampling fell below 85% at both 28 and 24 steps; this motivates compressing state capacity instead of discarding frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6860358437346396608, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1314432000, "recurrent_steps": 26080, "total_inference_macs": 1315214400, "training_seconds": 110.69780558301136, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4206474725454131}

RECENT RESULT
hypothesis: A 112-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by 12% versus the verified 120-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 120 to 112 while preserving all 32 causal steps and the training procedure.
mechanism: Recurrent-width compression with full temporal coverage
evidence_used: The 120-unit full-frame model achieved 86.75% accuracy, while 24- and 28-frame models failed; this supports preserving temporal coverage and testing a modest 6.7% reduction in state capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 140.04110266687348, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42283919632800515}

RECENT RESULT
hypothesis: A 108-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.5% versus the verified 112-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 112 to 108 while preserving all 32 causal steps and the training procedure.
mechanism: Recurrent-width boundary search with full temporal coverage
evidence_used: The 112-unit full-frame model achieved 85.77% accuracy after the 120-unit model achieved 86.75%; this supports testing 108 units as the next likely feasible point before the observed accuracy margin is exhausted.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5645409772253230232, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 1081589760, "recurrent_steps": 26080, "total_inference_macs": 1082293920, "training_seconds": 144.95902220788412, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4493955928123802}

RECENT RESULT
hypothesis: A 104-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.7% versus the verified 108-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 108 to 104 while preserving all 32 causal steps and the verified training procedure.
mechanism: Recurrent-width boundary search with full temporal coverage
evidence_used: The 108-unit full-frame model achieved 86.01% accuracy, slightly exceeding the 112-unit model’s 85.77%, so another modest four-unit reduction is the most direct test of the feasible width boundary without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 105.70535412500612, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}

RECENT RESULT
hypothesis: A 100-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7% versus the verified 104-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 104 to 100 while preserving all 32 causal steps and the training procedure.
mechanism: Recurrent-width boundary search with full temporal coverage
evidence_used: The 104-unit full-frame model achieved 85.52% accuracy, and each prior width reduction from 120 through 108 to 104 remained feasible; another four-unit reduction directly probes the remaining width boundary without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 93.20833162497729, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}

RECENT RESULT
hypothesis: A 99-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.8% versus the verified 100-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 100 to 99 while preserving all 32 causal steps and the verified training procedure.
mechanism: Fine-grained recurrent-width boundary search
evidence_used: The 100-unit full-frame GRU achieved 85.03% accuracy, while all tested widths from 104 through 120 also passed; because the remaining accuracy margin is narrow, a one-unit reduction is the most informative low-risk refinement of the feasible width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 100.78154466697015, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4628224940387749}

RECENT RESULT
hypothesis: A 98-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.84% versus the verified 99-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 99 to 98 while preserving all 32 causal steps and the verified training procedure.
mechanism: Fine-grained recurrent-width boundary search
evidence_used: The 99-unit full-frame GRU achieved 85.03% accuracy, identical to the 100-unit result, so a one-unit reduction is the most informative next probe of the feasible width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 108.62356691597961, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4702767307772958}

RECENT RESULT
hypothesis: A 97-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.9% versus the verified 98-unit model.
change: Reduce the GRU hidden width, recurrent state, temporal summary, and classifier input from 98 to 97 while preserving all 32 causal steps and the verified training procedure.
mechanism: One-unit recurrent-width boundary probe
evidence_used: The 98-unit model achieved 85.03% accuracy, matching the verified 99- and 100-unit models; this stable accuracy across consecutive widths motivates testing the next one-unit structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634959100755551733, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888578200, "training_seconds": 100.91902008303441, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.45978345812464055}

RECENT RESULT
hypothesis: Removing the eight classifier bias parameters will retain at least 85% validation accuracy while preserving exact MACs and recurrent steps, improving the parameter tiebreaker.
change: Disable the bias in the 98-to-8 classifier while leaving the verified recurrent architecture and training procedure unchanged.
mechanism: Bias-free classification head
evidence_used: The 98-unit model achieved 85.03% accuracy, whereas reducing recurrent width to 97 failed; a smaller head-only structural reduction is the lowest-risk next optimization.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119704, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36104, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 146.27364645898342, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4729783602287433}

RECENT RESULT
hypothesis: A 99-unit GRU processing 31 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 1.3% versus the verified 98-unit, 32-step model.
change: Increase GRU width from 98 to 99 and use 31 unique increasing frame indices spanning the complete recording.
mechanism: Capacity-for-time exchange with mild uniform subsampling
evidence_used: The 98- and 99-unit full-frame models both achieved 85.03%, while 97 units failed; restoring one unit provides the smallest evidenced capacity buffer for a much milder temporal reduction than the unsuccessful 24- and 28-frame trials.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4661082503522968417, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893586375, "training_seconds": 213.3947905418463, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4816822824302627}

RECENT RESULT
hypothesis: Training the 97-unit GRU with batch size 64 will approximately double optimizer updates and recover validation accuracy to at least 85%, while retaining the previously measured 97-unit inference cost.
change: Reduce the GRU, state, summary, and classifier width from 98 to 97, and reduce training batch size from 128 to 64.
mechanism: Smaller-batch training–compensated recurrent-width compression
evidence_used: The 97-unit model narrowly missed at 84.42% while achieving lower cross-entropy (0.4598) than the passing 98-unit model (0.4703), suggesting its representation remains viable and may benefit from more optimization updates per fixed 50,000-example exposure.
result: training did not finish within the verification time limit



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
