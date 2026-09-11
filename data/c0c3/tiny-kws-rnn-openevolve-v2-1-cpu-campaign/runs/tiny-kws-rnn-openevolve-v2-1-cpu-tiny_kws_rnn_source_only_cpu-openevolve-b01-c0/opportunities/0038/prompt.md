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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4780603979046285397, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 50552, "peak_hidden_elements": 121344, "recurrent_macs": 915730740, "recurrent_steps": 18745, "total_inference_macs": 916500100, "training_seconds": 70.17157133296132, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4331503838849214}
prior_hypothesis: A 118-unit GRU using the passing 23-frame schedule will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the 119-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: A 118-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.55% versus the passing 119-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 119 to 118 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 119 has passed, and the latest 119-unit model achieved 86.135% accuracy, leaving a 1.135-point margin for the next isolated one-channel reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5195959707761037387, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 50552, "peak_hidden_elements": 121344, "recurrent_macs": 995359500, "recurrent_steps": 20375, "total_inference_macs": 996128860, "training_seconds": 50.26036816602573, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.41964874267578123}

RECENT RESULT
hypothesis: A 117-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.56% versus the passing 118-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 118 to 117 while preserving the verified frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 118 has passed; the 118-unit model achieved 85.153% accuracy, making the next isolated one-channel reduction the most informative test of the recurrent-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5114622186734460013, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 979772625, "recurrent_steps": 20375, "total_inference_macs": 980535465, "training_seconds": 70.04316212516278, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4050237082264906}

RECENT RESULT
hypothesis: A 116-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 117-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 117 to 116 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: Every tested additive-readout width from 128 through 117 has passed; the current 117-unit model achieved 85.767% accuracy, providing evidence for one more isolated channel reduction as the next test of the width boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5033922340309564895, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 49000, "peak_hidden_elements": 119296, "recurrent_macs": 964308000, "recurrent_steps": 20375, "total_inference_macs": 965064320, "training_seconds": 73.46372124995105, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4151908453256806}

RECENT RESULT
hypothesis: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 24 scheduled frames while reducing total inference MACs by approximately 4% versus the verified 25-step model.
change: Remove one late intermediate frame from the schedule while preserving the first processed frame pattern, endpoint coverage, architecture, and training procedure.
mechanism: Single-step causal schedule trim
evidence_used: The 117-unit, 25-step model achieved 85.767% accuracy, while reducing width to 116 narrowly failed at 84.908%; testing a minimal temporal reduction at the passing width explores a different cost axis with a larger potential MAC improvement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4910196462845663493, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 940581720, "recurrent_steps": 19560, "total_inference_macs": 941344560, "training_seconds": 73.7940290409606, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.42476536101358797}

RECENT RESULT
hypothesis: The passing 117-unit additive-readout GRU will retain at least 85% validation accuracy with 23 scheduled frames while reducing total inference MACs by approximately 4.2% versus the verified 24-step model.
change: Remove one additional late intermediate frame while preserving the recurrent width, earlier frame pattern, final-frame coverage, and training procedure.
mechanism: Single-step causal schedule trim
evidence_used: The 117-unit model passed at both 25 steps (85.767%) and 24 steps (85.153%), whereas reducing width to 116 narrowly failed; this makes another isolated temporal-step reduction the most informative lower-cost test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4705770738956866973, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 901390815, "recurrent_steps": 18745, "total_inference_macs": 902153655, "training_seconds": 53.92610116698779, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4386127027265865}

RECENT RESULT
hypothesis: The 117-unit GRU will retain at least 85% validation accuracy with 23 recurrent steps when the second early frame is removed instead of a late intermediate frame, reducing total inference MACs by approximately 4.2%.
change: Preserve the passing 24-frame schedule’s first frame, late intermediate coverage, and final frame, but remove its second processed frame.
mechanism: Early-frame redundancy pruning
evidence_used: The 24-step model passed at 85.153%, while the prior 23-step schedule that removed a late intermediate frame fell to 84.294%; this directly motivates testing whether that failure was caused by losing informative late speech rather than by the step count itself.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4705770738956866973, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 901390815, "recurrent_steps": 18745, "total_inference_macs": 902153655, "training_seconds": 48.89489066693932, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4309426243319833}

RECENT RESULT
hypothesis: Re-spacing the 117-unit GRU’s coverage onto a uniform 23-step grid will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.2% versus the passing 24-step model.
change: Reduce the base interpolation grid from 28 to 27 points; the existing slicing then executes 23 uniformly redistributed causal frames while retaining the final frame.
mechanism: Uniform temporal-grid coarsening
evidence_used: The 24-step model passed at 85.153%, while deleting a late frame scored 84.294% and deleting the second frame narrowly missed at 84.908%; redistributing all intermediate samples tests whether temporal spacing, rather than 23-step capacity, caused those failures.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4705770738956866973, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 901390815, "recurrent_steps": 18745, "total_inference_macs": 902153655, "training_seconds": 79.07964308396913, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.43191201613724595}

RECENT RESULT
hypothesis: Averaging the two highest adjacent mel bands into one feature will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.87 million (0.73%) versus the passing 117-unit, 24-step model.
change: Reduce the GRU input from 20 to 19 features using parameter-free averaging of the two highest mel bands, while preserving recurrent width, schedule, readout, and training.
mechanism: Fixed adjacent-band compression
evidence_used: The 117-unit, 24-step design passed at 85.153%, while 116 hidden units and multiple 23-step schedules failed; a minimal compression of redundant adjacent frequency information explores a lower-risk structural cost axis.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4874384657215187980, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 49420, "peak_hidden_elements": 120320, "recurrent_macs": 933716160, "recurrent_steps": 19560, "total_inference_macs": 934479000, "training_seconds": 70.4454430420883, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.43885732077382095}

RECENT RESULT
hypothesis: A 119-unit GRU using the best-performing 23-frame schedule will recover the 0.092-point accuracy shortfall of the 117-unit version, achieving at least 85% accuracy while reducing total inference MACs by approximately 1.1% versus the passing 117-unit, 24-step design.
change: Increase recurrent and readout width from 117 to 119, then remove the second processed frame to execute 23 causal steps.
mechanism: Recurrent capacity-for-time exchange
evidence_used: Removing the second processed frame produced the strongest 117-unit, 23-step result at 84.908%, narrowly missing the threshold, while the 119-unit model previously achieved 86.135% at 25 steps; this tests whether a small capacity increase can enable the lower-step schedule.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4856023879769251497, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 51337, "peak_hidden_elements": 122368, "recurrent_macs": 930183135, "recurrent_steps": 18745, "total_inference_macs": 930959015, "training_seconds": 71.97894679196179, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.41494064799115704}

RECENT RESULT
hypothesis: A 118-unit GRU using the passing 23-frame schedule will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the 119-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 119 to 118 while preserving the 23-frame schedule and training procedure.
mechanism: One-channel recurrent-width trim
evidence_used: The 119-unit, 23-step model passed at 85.399%, while the 117-unit version narrowly missed at 84.908%; testing the intervening width is the most direct probe of the lower-cost accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4780603979046285397, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 50552, "peak_hidden_elements": 121344, "recurrent_macs": 915730740, "recurrent_steps": 18745, "total_inference_macs": 916500100, "training_seconds": 70.17157133296132, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4331503838849214}

RECENT RESULT
hypothesis: A 120-unit GRU using 22 causal steps will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.3% versus the passing 118-unit, 23-step model.
change: Increase recurrent/readout width from 118 to 120 and remove the second currently processed early frame, preserving the first frame and all later coverage.
mechanism: Early-frame pruning with recurrent capacity exchange
evidence_used: Increasing width by two while removing one step previously succeeded: the 119-unit, 23-step model reached 85.399% after the 117-unit, 23-step model missed narrowly, and it cost less than the passing 117-unit, 24-step design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4717771774879028458, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 903672000, "recurrent_steps": 17930, "total_inference_macs": 904454400, "training_seconds": 64.73765429202467, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4604076455707199}

RECENT RESULT
hypothesis: A learned 20-to-16 feature projection feeding a 119-unit GRU will retain at least 85% accuracy while reducing exact dense MACs below the passing 118-unit, 23-step model.
change: Add a bias-free learned input bottleneck, increase recurrent/readout width to 119, and preserve the verified 23-frame schedule and training procedure.
mechanism: Learned shared mel-subspace projection
evidence_used: The 119-unit, 23-step model passed at 85.399%, providing more margin than the 118-unit model’s 85.031%; unlike the failed fixed 19-band averaging, a learned projection can preserve task-discriminative spectral combinations.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4747687216107447329, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 50229, "peak_hidden_elements": 122368, "recurrent_macs": 909413675, "recurrent_steps": 18745, "total_inference_macs": 910189555, "training_seconds": 47.56254187505692, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4567584453184912}



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
