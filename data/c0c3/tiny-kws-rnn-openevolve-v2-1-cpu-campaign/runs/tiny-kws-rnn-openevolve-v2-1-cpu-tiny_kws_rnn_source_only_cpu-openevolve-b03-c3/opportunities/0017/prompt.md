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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 109.62608612515032, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}
prior_hypothesis: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the qualified 104-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4392370677619577616, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34096, "peak_hidden_elements": 131584, "recurrent_macs": 841236480, "recurrent_steps": 26080, "total_inference_macs": 842071040, "training_seconds": 149.85610191593878, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4525330151516967}
prior_hypothesis: Two independent 64-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing expected total inference MACs to approximately 842,071,040, below the qualified 98-unit GRU’s 905,406,320.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4144476801797606975, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32535, "peak_hidden_elements": 151040, "recurrent_macs": 793588320, "recurrent_steps": 26080, "total_inference_macs": 794546760, "training_seconds": 105.14707025000826, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4558886849807084}
prior_hypothesis: Three parallel 49-unit GRUs processing all 32 frames will achieve at least 85% validation accuracy at approximately 794,546,760 total inference MACs, below the qualified two-64-unit model’s 842,071,040 MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 6.7% versus the qualified 108-unit design.
change: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 128 to 104 while preserving the full-frame schedule and training procedure.
mechanism: Conservative full-resolution GRU width reduction
evidence_used: The 108-unit full-resolution GRU achieved 86.01% accuracy, while reducing temporal resolution to 24 or 16 steps failed below 83%; this supports preserving all frames and testing a small additional width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 122.09208366600797, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44803061456036714}

RECENT RESULT
hypothesis: A 100-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the qualified 104-unit design.
change: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 104 to 100 while preserving the full-frame schedule and training procedure.
mechanism: Conservative full-resolution GRU width reduction
evidence_used: The 104-unit full-resolution GRU achieved 85.52% accuracy, and the preceding 112-to-108-to-104 width reductions all remained qualified; meanwhile, reducing temporal resolution to 24 or 16 steps produced less than 83% accuracy, motivating another four-unit width reduction while retaining every frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 109.62608612515032, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4658823680292609}

RECENT RESULT
hypothesis: A 98-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing total inference MACs below the qualified 100-unit model’s 939,532,000 MACs.
change: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 108 to 98 while preserving the full-frame schedule and training procedure.
mechanism: Fine-grained full-resolution GRU width reduction
evidence_used: The full-resolution 100-unit GRU qualified at 85.03% accuracy after successive 112→108→104→100 reductions, while temporal subsampling failed below 83%; a two-unit width step probes the apparent capacity boundary more precisely.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 101.06091408384964, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4702767307772958}

RECENT RESULT
hypothesis: A 97-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 1.9% versus the qualified 98-unit design.
change: Reduce the GRU hidden width, classifier input width, and recurrent state tensors from 112 to 97 while preserving the full-frame schedule and training procedure.
mechanism: One-unit full-resolution GRU width reduction
evidence_used: The 98-unit full-resolution GRU qualified at 85.03% accuracy with 905,406,320 MACs; because this is the observed capacity boundary, a one-unit reduction is the most precise next probe while retaining every frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634959100755551733, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888578200, "training_seconds": 95.62258195807226, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.45978345812464055}

RECENT RESULT
hypothesis: A 97-unit full-frame GRU trained with random ±2-frame temporal shifts will recover at least 85% validation accuracy while reducing total inference MACs below the qualified 98-unit model’s 905,406,320 MACs.
change: Reduce the GRU and recurrent state width from 98 to 97, and add random zero-padded temporal shifts during training while retaining all 32 recurrent steps at inference.
mechanism: Training-only temporal translation regularization at the recurrent-width boundary
evidence_used: The 97-unit model missed the target by only 0.58 percentage points while achieving lower cross-entropy than the qualified 98-unit model; unlike the failed 24- and 16-frame schedules, temporal translation augmentation preserves full frame density and adds no inference cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634959100755551733, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888578200, "training_seconds": 157.04429112491198, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.45863346029644364}

RECENT RESULT
hypothesis: A 192-dimensional residual state with a rank-56 recurrent correction and input-conditioned retention will preserve at least 85% accuracy across all 32 frames while reducing total inference MACs below the qualified 98-unit GRU’s 905,406,320.
change: Replace the dense three-gate GRU with a high-capacity diagonal-memory cell whose cross-state interaction is structurally low-rank, and classify from both its final state and temporal mean.
mechanism: Diagonal-plus-low-rank input-gated recurrent state
evidence_used: Full-frame GRUs reached a width boundary at 98 units (85.03%) versus 97 units (84.42%), while 24- and 16-frame schedules stayed below 83%. This challenges the load-bearing assumption that further savings require less state or fewer frames: the patch retains every frame and expands state capacity while replacing three dense recurrent transforms with a gated diagonal path plus one rank-56 correction.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Two independent 64-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing expected total inference MACs to approximately 842,071,040, below the qualified 98-unit GRU’s 905,406,320.
change: Replace the single 112-unit GRU with two parallel 64-unit GRUs, concatenate their outputs into a 128-dimensional temporal summary, and preserve the full-frame schedule and training procedure.
mechanism: Parallel block-diagonal gated recurrence
evidence_used: The full 128-unit GRU achieved 87.24% accuracy, while the 98-unit GRU exposed a capacity boundary at 85.03% and temporal subsampling failed below 83%. Parallel 64-unit recurrent blocks retain 128 aggregate gated units and every input frame while structurally reducing quadratic hidden-to-hidden MACs by about 7% versus the qualified 98-unit model.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4392370677619577616, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34096, "peak_hidden_elements": 131584, "recurrent_macs": 841236480, "recurrent_steps": 26080, "total_inference_macs": 842071040, "training_seconds": 149.85610191593878, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4525330151516967}

RECENT RESULT
hypothesis: Giving each 64-unit GRU 19 overlapping mel bands while collectively retaining all 20 bands and all 32 frames will maintain at least 85% validation accuracy and reduce total inference MACs from 842,071,040 to approximately 832,056,320.
change: Remove one different edge mel band from each parallel GRU input, reducing both input matrices from 20 to 19 features while preserving the 128-unit aggregate recurrent state.
mechanism: Overlapping spectral partition of parallel recurrent inputs
evidence_used: The parallel 64-unit GRUs qualified at 85.28%, whereas temporal subsampling failed below 83%; this motivates preserving every recurrent step and hidden unit while testing whether broadcasting the complete 20-band input to both independent branches is redundant.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4340132374249767312, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 33712, "peak_hidden_elements": 131584, "recurrent_macs": 831221760, "recurrent_steps": 26080, "total_inference_macs": 832056320, "training_seconds": 102.33075516717508, "validation_accuracy": 0.838036809815951, "validation_cross_entropy": 0.4725329065615414}

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
