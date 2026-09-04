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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722290763037942031, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36006, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905320745, "training_seconds": 112.10978870815597, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.44277687306784413}
prior_hypothesis: A seven-output classifier with the eighth logit derived as the negative sum will retain at least 85% accuracy while reducing exact classifier MACs by 79,055, because it preserves all seven independent softmax decision dimensions.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing label smoothing will sharpen class boundaries enough for the 97-unit GRU to recover the 0.58-point accuracy shortfall and reach at least 85%, while retaining its lower measured inference cost and avoiding the batch-64 timeout.
change: Reduce the GRU and classifier width from 98 to 97, keep batch size 128, and replace label-smoothed cross-entropy with standard cross-entropy.
mechanism: Hard-target training-compensated recurrent-width compression
evidence_used: The 97-unit model achieved 84.42% accuracy but lower validation cross-entropy than the passing 98-unit model (0.4598 versus 0.4703), indicating viable representations; the attempted smaller-batch remedy timed out, motivating a zero-runtime-cost objective adjustment.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634959100755551733, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888578200, "training_seconds": 163.8366866670549, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.45162806598686733}

RECENT RESULT
hypothesis: Merging the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.85% versus the verified 98-unit model.
change: Preserve the 98-unit GRU and all 32 causal steps, but average the final two mel bands into one feature and reduce the GRU input width from 20 to 19.
mechanism: Fixed adjacent-band spectral pooling
evidence_used: The 98-unit full-frame model passed at 85.03%, while reducing hidden width to 97 and reducing temporal coverage both failed; this motivates an orthogonal, minimal reduction in redundant adjacent spectral resolution.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4682742184241608696, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35816, "peak_hidden_elements": 100864, "recurrent_macs": 897099840, "recurrent_steps": 26080, "total_inference_macs": 897738800, "training_seconds": 180.0237707500346, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4613856496986436}

RECENT RESULT
hypothesis: Alternating the final GRU input between mel bands 18 and 19 will preserve their distinct spectral cues and achieve at least 85% accuracy while retaining the 19-input model’s lower 897,738,800-MAC inference cost.
change: Keep the passing 98-unit, 32-step architecture, reduce its GRU input to 19 features, and alternate which of the two highest mel bands occupies the final input channel at each causal step.
mechanism: Temporal multiplexing of adjacent high-frequency mel bands
evidence_used: Averaging the two highest bands reduced inference MACs by 0.85% and narrowly missed accuracy at 84.91% with better cross-entropy than the passing baseline; temporal multiplexing tests whether retaining their distinct values avoids the information loss from averaging.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4682742184241608696, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35816, "peak_hidden_elements": 100864, "recurrent_macs": 897099840, "recurrent_steps": 26080, "total_inference_macs": 897738800, "training_seconds": 126.76342004095204, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46272703767554163}

RECENT RESULT
hypothesis: Retaining the verified 98-unit recurrent dynamics while excluding one summary coordinate from the classifier will preserve at least 85% validation accuracy and reduce exact classifier MACs by 6,520 over the validation set.
change: Reduce only the classification head input from 98 to 97 and classify from the first 97 coordinates of the averaged recurrent summary.
mechanism: One-coordinate recurrent readout pruning
evidence_used: The 98-unit GRU passed at 85.03%, while shrinking the recurrent state to 97 failed; pruning one coordinate only at readout preserves the full recurrent capacity and is a lower-risk structural reduction than altering recurrence, temporal coverage, or spectral inputs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722703125947029984, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36104, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905399800, "training_seconds": 63.04487991705537, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.45813509701219807}

RECENT RESULT
hypothesis: Classifying from 96 of the 98 recurrent summary coordinates will retain at least 85% validation accuracy while reducing exact classifier MACs by another 6,520 and learned parameters by 8.
change: Reduce the classifier input width from 97 to 96 and omit one additional summary coordinate, while preserving the verified 98-unit recurrent dynamics and all 32 causal steps.
mechanism: Incremental recurrent readout pruning
evidence_used: The 97-coordinate readout achieved 85.52% accuracy—higher than the 98-coordinate baseline’s 85.03%—while reducing MACs, indicating that readout pruning is better tolerated than shrinking the recurrent state, spectral input, or temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722669116634940256, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36096, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905393280, "training_seconds": 103.60144124994986, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4570938765636982}

RECENT RESULT
hypothesis: Folding all 98 recurrent summary coordinates into a 96-dimensional readout will retain at least 85% accuracy while reducing classifier inference by 6,520 MACs versus the verified 97-coordinate design.
change: Reduce the classifier input from 97 to 96 and fold the final two summary coordinates into the first two with variance-preserving fixed additions, preserving a direct classification path from every recurrent coordinate.
mechanism: Fixed orthogonal recurrent-coordinate folding
evidence_used: The cropped 96-coordinate readout narrowly failed at 84.66% despite achieving lower cross-entropy than the passing 97-coordinate model; retaining information and direct classifier gradients from the two previously discarded coordinates specifically targets that failure while keeping the lower 96-input MAC count.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722669116634940256, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36096, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905393280, "training_seconds": 111.14206658303738, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4578865753361053}

RECENT RESULT
hypothesis: A seven-output classifier with the eighth logit derived as the negative sum will retain at least 85% accuracy while reducing exact classifier MACs by 79,055, because it preserves all seven independent softmax decision dimensions.
change: Replace the 97-to-8 classifier with a 97-to-7 classifier and construct an eight-logit zero-sum output without another learned matrix operation.
mechanism: Zero-sum softmax logit parameterization
evidence_used: The 97-coordinate readout passed at 85.52%, while reducing its input to 96 failed; exploiting softmax’s redundant common-logit dimension reduces MACs without discarding another recurrent feature.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722290763037942031, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36006, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905320745, "training_seconds": 112.10978870815597, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.44277687306784413}

RECENT RESULT
hypothesis: A 97-unit GRU with the seven-output zero-sum classifier will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.86% versus the current 98-unit model.
change: Reduce the GRU hidden state and temporal summary from 98 to 97 units, retaining all 32 frames and classifying from the complete 97-dimensional summary.
mechanism: Zero-sum-head-compensated recurrent compression
evidence_used: The 97-unit model with a conventional eight-output head narrowly missed at 84.42%, while changing the 98-unit model from an eight-output head to the seven-output zero-sum parameterization improved accuracy from 85.52% to 86.26%; a comparable gain would place the compressed model above 85%.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4634546737846463780, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35355, "peak_hidden_elements": 99840, "recurrent_macs": 887945760, "recurrent_steps": 26080, "total_inference_macs": 888499145, "training_seconds": 104.70380000001751, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4641465731193683}



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
