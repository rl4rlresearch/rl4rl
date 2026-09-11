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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3559329811539021375, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 24.520245398773007, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 681534336, "recurrent_steps": 19984, "total_inference_macs": 682366944, "training_seconds": 121.45353691698983, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.5108774290494392}
prior_hypothesis: Lowering the penultimate-step exit threshold from 0.90 to 0.85 will retain at least 85% validation accuracy while reducing recurrent MACs and mean recurrent steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Omitting frame 29 in addition to frames 0–2 and 30–31 will retain at least 85% validation accuracy while reducing recurrent execution from 27 to 26 steps and recurrent MACs by approximately 3.70%.
change: Schedule frames 3–28 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental trailing-frame recurrent step pruning
evidence_used: Successively removing trailing frames 31 and 30 preserved passing accuracy at 85.52% and 85.28%, respectively, while removing frame 3 failed; this makes the next trailing frame the best-supported target for another step reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3772380917734978437, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 722663760, "recurrent_steps": 21190, "total_inference_macs": 723211440, "training_seconds": 70.6754772500135, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4919525708157592}

RECENT RESULT
hypothesis: Omitting frame 28 in addition to frames 0–2 and 29–31 will retain at least 85% validation accuracy while reducing recurrent execution from 26 to 25 steps and recurrent MACs by approximately 3.85%.
change: Schedule frames 3–27 for standard 32-frame inputs while leaving the verified 18-feature, 98-unit GRU and seven-output zero-sum classifier unchanged.
mechanism: Incremental trailing-frame recurrent step pruning
evidence_used: Successive removal of trailing frames 31, 30, and 29 preserved passing accuracy, with the current 26-step model reaching 85.40%; this makes the next trailing frame the most directly supported target for further cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 60.87797804106958, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.5098119255955234}

RECENT RESULT
hypothesis: Retaining frame 28 while omitting the locally redundant middle frame 16 will recover at least 85% accuracy at the lower 25-step inference cost.
change: Change the 32-frame schedule from contiguous frames 3–28 to frames 3–15 and 17–28, leaving the model and training procedure unchanged.
mechanism: Interior-frame temporal thinning
evidence_used: The 26-step schedule over frames 3–28 reached 85.40%, while truncating it to frames 3–27 missed the threshold by only one validation example; preserving the later endpoint and relocating the omission to an interior frame tests whether temporal coverage matters more than one densely sampled middle frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 57.50335258315317, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.5045694082061206}

RECENT RESULT
hypothesis: Retaining frame 28 while omitting adjacent frame 27 will recover at least 85% accuracy at 25 recurrent steps because frame 28 distinguished the passing 26-step schedule from the narrowly failing truncation.
change: Use frames 3–26 and 28 for standard 32-frame inputs, reducing execution from 26 to 25 steps without changing model dimensions or training.
mechanism: Penultimate-frame pruning
evidence_used: Frames 3–28 achieved 85.40%, while removing frame 28 scored 84.91%, only one validation example below threshold; this motivates preserving frame 28 and pruning its temporally redundant predecessor instead.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627399220215001262, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695416680, "training_seconds": 64.51925350003876, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.509380672899492}

RECENT RESULT
hypothesis: Omitting frames 25 and 27 while retaining frames 26 and 28 will preserve at least 85% validation accuracy with 24 recurrent steps, because spaced omissions maintain late temporal coverage better than contiguous tail truncation.
change: Change the 32-frame schedule from frames 3–26 and 28 to frames 3–24, 26, and 28, reducing execution from 25 to 24 steps without changing model capacity or training.
mechanism: Alternating late-tail temporal thinning
evidence_used: The 25-step design passed only when frame 28 was retained and adjacent frame 27 was omitted; removing frame 28 narrowly failed, motivating another alternating omission in the locally redundant late region while preserving frames 26 and 28.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482417522695024087, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 667074240, "recurrent_steps": 19560, "total_inference_macs": 667621920, "training_seconds": 60.96644412493333, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.5228541181131374}

RECENT RESULT
hypothesis: Reducing the classifier input from 96 to 95 recurrent-summary coordinates will preserve at least 85% validation accuracy while lowering total inference MACs at the successful 25-step schedule.
change: Keep the GRU, feature pooling, and frame schedule unchanged, but remove one classifier input coordinate.
mechanism: Incremental recurrent-readout coordinate pruning
evidence_used: The current 25-step design passes at 85.03%, whereas the tested 24-step schedule falls to 84.66%; a one-coordinate readout reduction is a smaller structural change that improves exact MAC cost without further reducing temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3627369462066922750, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35400, "peak_hidden_elements": 100864, "recurrent_macs": 694869000, "recurrent_steps": 20375, "total_inference_macs": 695410975, "training_seconds": 58.80939258309081, "validation_accuracy": 0.8294478527607362, "validation_cross_entropy": 0.5300466712998466}

RECENT RESULT
hypothesis: Retaining frame 25 while omitting frames 26–27 will achieve at least 85% validation accuracy with 24 recurrent steps because the failed 24-step schedule omitted frame 25 while retaining frame 26.
change: Change the 32-frame schedule from frames 3–26 and 28 to frames 3–25 and 28, reducing execution from 25 to 24 steps without changing model capacity or training.
mechanism: Alternate-phase late-tail temporal thinning
evidence_used: The current frames 3–26 and 28 schedule passes at 85.03%, while frames 3–24, 26, and 28 scored 84.66%; swapping which adjacent late frame is retained directly tests whether frame 25 carries the information lost by that unsuccessful pruning.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482417522695024087, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 667074240, "recurrent_steps": 19560, "total_inference_macs": 667621920, "training_seconds": 55.361940749920905, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.5260152898683138}

RECENT RESULT
hypothesis: Retaining the important frame 3 while omitting adjacent frame 4 will preserve at least 85% validation accuracy with 24 recurrent steps.
change: Change the standard 32-frame schedule from frames 3–26 and 28 to frames 3, 5–26, and 28.
mechanism: Anchor-preserving early-neighbor thinning
evidence_used: Removing frame 3 caused a 28-step model to fail, while preserving informative frame 28 and removing its adjacent frame 27 produced the passing 25-step design; this tests the analogous adjacent-frame redundancy around the early anchor without sacrificing late coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482417522695024087, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 667074240, "recurrent_steps": 19560, "total_inference_macs": 667621920, "training_seconds": 61.34749845904298, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.5065500551937548}

RECENT RESULT
hypothesis: Exiting after 24 steps only when confidence is at least 0.97 will preserve at least 85% accuracy and reduce total MACs, provided at least 2% of examples skip the final recurrent step.
change: Add selective early exit before frame 28 while retaining the verified 25-step schedule; use a zero-MAC recurrent-state projection for provisional logits before the exit point.
mechanism: Confidence-gated final-frame skipping
evidence_used: Every tested fixed 24-step schedule missed the threshold by only a few examples, while the current 25-step schedule passed at 85.03%; routing uncertain examples through the informative final frame should retain its accuracy while confident examples avoid that step.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3599237131222227515, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 24.790184049079755, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 689037216, "recurrent_steps": 20204, "total_inference_macs": 690017664, "training_seconds": 153.44055691710673, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.5092092034275546}

RECENT RESULT
hypothesis: Lowering the penultimate-step exit threshold from 0.97 to 0.95 will preserve at least 85% validation accuracy while increasing the fraction of examples that skip the final recurrent step and reducing exact inference MACs.
change: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.95 instead of 0.97.
mechanism: Moderately relaxed confidence-gated final-frame skipping
evidence_used: The 0.97 gate preserved the 25-step model’s exact 85.03% accuracy while reducing mean recurrent steps to 24.79 and slightly improving cross-entropy, showing that confidence-selected examples can safely omit the final frame; a modest threshold relaxation directly tests additional savings.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3588897507486124106, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 24.720245398773006, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 687093288, "recurrent_steps": 20147, "total_inference_macs": 688035432, "training_seconds": 131.0165756670758, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.5093834999880177}

RECENT RESULT
hypothesis: Lowering the penultimate-step exit threshold from 0.95 to 0.90 will preserve at least 85% validation accuracy while allowing more examples to skip the 25th recurrent step, reducing exact inference MACs and mean recurrent steps.
change: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.90 instead of 0.95.
mechanism: Relaxed confidence-gated final-frame skipping
evidence_used: Lowering the threshold from 0.97 to 0.95 increased final-frame skipping, reduced mean recurrent steps from 24.790 to 24.720, and improved validation accuracy from 85.03% to 85.15%; this supports testing a further moderate relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3572208991982237902, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 24.607361963190183, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 683955720, "recurrent_steps": 20055, "total_inference_macs": 684836040, "training_seconds": 150.05493579106405, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.5097779162822326}

RECENT RESULT
hypothesis: Lowering the penultimate-step exit threshold from 0.90 to 0.85 will retain at least 85% validation accuracy while reducing recurrent MACs and mean recurrent steps.
change: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.85 instead of 0.90.
mechanism: Further relaxed confidence-gated final-frame skipping
evidence_used: Lowering the threshold from 0.95 to 0.90 preserved 85.15% accuracy while reducing mean recurrent steps from 24.720 to 24.607 and total inference MACs from 688,035,432 to 684,836,040, supporting another incremental relaxation.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3559329811539021375, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 24.520245398773007, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35407, "peak_hidden_elements": 100864, "recurrent_macs": 681534336, "recurrent_steps": 19984, "total_inference_macs": 682366944, "training_seconds": 121.45353691698983, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.5108774290494392}



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
