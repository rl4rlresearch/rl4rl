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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1198556178784957731, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24301, "peak_hidden_elements": 182784, "recurrent_macs": 227866665, "recurrent_steps": 21190, "total_inference_macs": 229777840, "training_seconds": 41.95087758405134, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43152330082618384}
prior_hypothesis: A 67-unit paired-frame GRU with the successful six-transition early pool split into two three-transition summaries will retain at least 85% accuracy while reducing total inference MACs from 234,980,800 to approximately 229,777,840.

## Recent verification evidence

RECENT RESULT
hypothesis: A 56-unit GRU processing the verified 26-frame window at indices 3–28 will retain at least 85% validation accuracy while reducing total dense inference MACs below the successful 58-unit, 25-step model.
change: Reduce GRU width from 58 to 56 units, resize its recurrent state and classifier, and restore the 26-frame central schedule.
mechanism: Capacity-for-time recurrent tradeoff
evidence_used: The 58-unit 26-step model achieved 86.01% accuracy, while increasing width to 59 could not compensate for reducing the schedule to 24 steps. This indicates temporal context is more valuable than marginal width and motivates exchanging two hidden units for one additional frame; 56 units is the largest integer width that makes this 26-step configuration cheaper than the current 25-step model.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1416963981025129766, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 14496, "peak_hidden_elements": 86528, "recurrent_macs": 270553920, "recurrent_steps": 21190, "total_inference_macs": 271649280, "training_seconds": 41.23041675006971, "validation_accuracy": 0.838036809815951, "validation_cross_entropy": 0.4688794375928633}

RECENT RESULT
hypothesis: Mapping normalized 20-band frames into a well-conditioned 19-dimensional Helmert contrast basis will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.55M versus the verified 25-step model.
change: Preserve affine input normalization, replace the raw 20 normalized coordinates with 19 orthonormal spectral contrasts, and structurally reduce the GRU input width from 20 to 19.
mechanism: Orthonormal lossless mel-coordinate elimination
evidence_used: Direct coordinate omission with a 19-input GRU narrowly missed at 84.91%; because normalized frames lie in a 19-dimensional zero-mean subspace, an orthonormal contrast basis preserves that information while avoiding the poor conditioning of reconstructing the omitted coordinate as the sum of the other 19.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1429845007897611091, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 15186, "peak_hidden_elements": 89600, "recurrent_macs": 272984250, "recurrent_steps": 20375, "total_inference_macs": 274118730, "training_seconds": 40.68544124998152, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.46309915554304065}

RECENT RESULT
hypothesis: A 70-unit GRU that jointly consumes adjacent frames from the verified 26-frame window will retain at least 85% accuracy while reducing learned GRU transitions from 25 to 13 and estimated total validation MACs from 277.66M to 246.11M.
change: Cache the first frame of each causal pair without a dense operation, concatenate it with the next frame, and execute one wider GRU transition per pair; vectorize the identical paired computation during sequence training.
mechanism: Lossless causal frame-pair transitions
evidence_used: The 58-unit model reached 86.01% on frames 3–28, while deleting frames and compensating with width failed. This challenges the load-bearing assumption that every retained frame needs its own full recurrent transition: all 26 proven-useful frames remain available, but temporal updates occur at a learned two-frame clock.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1283766510225752138, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25248, "peak_hidden_elements": 118784, "recurrent_macs": 244744500, "recurrent_steps": 21190, "total_inference_macs": 246113700, "training_seconds": 25.08536475012079, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42380269992570935}

RECENT RESULT
hypothesis: A 69-unit paired-frame GRU will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 2.32%, from 246.11M to 240.40M.
change: Reduce the paired GRU hidden width from 70 to 69 and resize its recurrent state and three-way pooled classifier accordingly.
mechanism: One-unit paired-GRU width pruning
evidence_used: The 70-unit paired-frame model achieved 85.77% accuracy with 13 learned transitions, providing a 0.77-point margin and motivating a conservative width-boundary probe that preserves all 26 input frames and the successful temporal mechanism.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1253987106327190496, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24681, "peak_hidden_elements": 117248, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240404625, "training_seconds": 33.33605133299716, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.448084123295509}

RECENT RESULT
hypothesis: A 69-unit paired-frame GRU with separate early/late means will reach at least 85% accuracy while reducing estimated total inference MACs from 246.11M to 240.85M.
change: Reduce the paired GRU from 70 to 69 units and replace its global mean with separate means for the first six and remaining seven learned transitions, expanding the classifier from three to four pooled views.
mechanism: Recurrent-to-readout MAC reallocation with temporal-pyramid pooling
evidence_used: The 69-unit model narrowly missed the threshold at 84.79%, only 0.21 points short, while the otherwise identical 70-unit model reached 85.77%; spending 552 additional classifier MACs per example on explicit temporal structure tests whether cheap readout capacity can recover that narrow loss.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1256333748861381728, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 25233, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240854505, "training_seconds": 32.31677362485789, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4130140246057803}

RECENT RESULT
hypothesis: Early/late mean pooling plus max pooling will retain at least 85% validation accuracy while removing 552 classifier MACs per example, reducing total validation MACs from 240.85M to approximately 240.40M.
change: Remove the final hidden-state view from the classifier while preserving the temporal-pyramid summaries, max pooling, 69-unit paired GRU, and 26-frame schedule.
mechanism: Redundant final-state readout removal
evidence_used: The 69-unit model passed at 85.89% only after adding separate early/late summaries; unlike the failed global-mean three-view model, this patch preserves that successful temporal structure and removes the final state, which is already represented within the late summary and recurrent maximum.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1253987106327190496, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24681, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240404625, "training_seconds": 33.265574417077005, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4450192410521712}

RECENT RESULT
hypothesis: Replacing eight learned logits with seven learned relative logits will preserve at least 85% accuracy while reducing total validation MACs by 168,705, from 240,404,625 to 240,235,920.
change: Produce seven classifier outputs, append a fixed reference logit, and mean-center all eight logits without changing the recurrent model or the representable class probabilities.
mechanism: Softmax gauge-dimension elimination
evidence_used: The current 69-unit paired-GRU design passes at 85.28%. Softmax probabilities have only seven independent logit differences, so removing the unidentifiable common-offset degree reduces classifier cost without reducing predictive expressiveness.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1253107115376868783, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24473, "peak_hidden_elements": 152576, "recurrent_macs": 239054985, "recurrent_steps": 21190, "total_inference_macs": 240235920, "training_seconds": 39.14977308316156, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46183706646316625}

RECENT RESULT
hypothesis: Restoring the informative final-state view while reducing the paired GRU from 69 to 68 units will retain at least 85% validation accuracy and reduce total inference MACs from 240,235,920 to approximately 234,980,800.
change: Resize the paired GRU to 68 units and classify four 68-dimensional views: early mean, late mean, temporal maximum, and final recurrent state.
mechanism: Recurrent-to-readout capacity reallocation
evidence_used: The 69-unit four-view model achieved 85.89% accuracy, substantially above the threshold; removing its final-state view reduced accuracy to 85.28%, showing that cheap readout capacity can support recurrent-width pruning.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1225695609832554381, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24391, "peak_hidden_elements": 150528, "recurrent_macs": 233429040, "recurrent_steps": 21190, "total_inference_macs": 234980800, "training_seconds": 37.61118116695434, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.43288474053692966}

RECENT RESULT
hypothesis: A 67-unit paired-frame GRU with the successful six-transition early pool split into two three-transition summaries will retain at least 85% accuracy while reducing total inference MACs from 234,980,800 to approximately 229,777,840.
change: Reduce GRU width from 68 to 67 and replace the early/late/max/final readout with first-early/second-early/late/max/final pooling, preserving the established transition-six boundary while adding inexpensive temporal resolution.
mechanism: Temporal-pyramid readout for recurrent-width pruning
evidence_used: The 68-unit four-view model achieved 85.77%, and at 69 units the informative final-state view improved accuracy from 85.28% to 85.89%. This shows that cheap temporally structured readout capacity can support recurrent-width pruning.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1198556178784957731, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24301, "peak_hidden_elements": 182784, "recurrent_macs": 227866665, "recurrent_steps": 21190, "total_inference_macs": 229777840, "training_seconds": 41.95087758405134, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43152330082618384}

RECENT RESULT
hypothesis: A 66-unit paired-frame GRU with the seven-transition late pool split into three- and four-transition summaries will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 224,627,040.
change: Reduce GRU width from 67 to 66 and add a sixth pooled classifier view by splitting the late summary at transition nine; preserve all 26 frames, 13 learned transitions, max pooling, final-state pooling, and seven relative logits.
mechanism: Progressive temporal-pyramid readout for recurrent-width pruning
evidence_used: Reducing the successful 68-unit model to 67 units while adding one inexpensive temporal summary retained 85.52% accuracy and lowered MACs; applying the same recurrent-to-readout trade reallocates capacity toward temporal structure while pruning another hidden unit.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1171688822234078833, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 24203, "peak_hidden_elements": 214016, "recurrent_macs": 222367860, "recurrent_steps": 21190, "total_inference_macs": 224627040, "training_seconds": 32.4138487088494, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4517080295305311}

RECENT RESULT
hypothesis: A 76-unit GRU consuming 27 central frames in nine causal three-frame transitions will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 229,611,580.
change: Cache two frames, concatenate each causal frame triplet into a 60-feature GRU input, widen the hidden state to 76 units, and use three equal transition summaries plus maximum and final-state views.
mechanism: Causal three-frame transition packing
evidence_used: Pairing preserved all useful frames and reached 85.77% with only 13 learned transitions, whereas deleting frames failed; triplet packing tests whether further transition reduction can preserve accuracy while a 76-unit width keeps recurrent MACs slightly below the successful 67-unit paired model.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Packing the proven 26-frame window into eight three-frame transitions and one central two-frame transition with a 76-unit GRU will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 229,611,580.
change: Replace 13 paired transitions with nine mixed-width temporal chunks, widen the GRU from 67 to 76 units, and retain three temporal summaries, maximum pooling, final-state pooling, and seven relative logits.
mechanism: Mixed pair–triplet causal transition packing
evidence_used: The 67-unit paired model reached 85.52%, showing every retained frame need not receive an individual transition, while frame deletion failed. The earlier triplet implementation was not verified and therefore provides no negative accuracy evidence; this version preserves exactly the successful 26 input frames with explicit step/sequence-equivalent phase logic.
result: the implementation could not be verified

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
