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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4308117546380092537, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 29.137423312883435, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 825184503, "recurrent_steps": 23747, "total_inference_macs": 825918687, "training_seconds": 149.85506487521343, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4649688159029908}
prior_hypothesis: A classifier-weight bound at step 29 will preserve every exited example’s final predicted class, maintaining at least 85% accuracy while reducing mean recurrent steps and exact MACs.

## Recent verification evidence

RECENT RESULT
hypothesis: The verified 19-feature, 99-unit GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 28.6 million.
change: Omit the first two input frames instead of only the first, preserving the final 30 causal frames and all model capacity.
mechanism: Second leading-frame omission
evidence_used: The current 31-step design achieved 85.52% accuracy, a 0.52-point margin above the requirement; omitting one leading frame previously retained 85.03%, indicating that one additional boundary-frame omission is the most informative low-risk temporal reduction. The attempted 18-feature design supplied no accuracy evidence because training timed out.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Compressing bands 16–17 and 18–19 independently will retain at least 85% accuracy while reducing recurrent MACs by 7,503,705 versus the verified 19-input design.
change: Reduce the GRU input width from 19 to 18 by retaining bands 0–15 and averaging the two highest adjacent band pairs; preserve the verified 99-unit state, 31-step schedule, and classifier.
mechanism: Second adjacent high-frequency mel-band pooling
evidence_used: Pooling bands 18–19 achieved 85.52% accuracy, improving on the uncompressed 31-step model’s 85.03%. The previous test of this second pooling change timed out and therefore supplied no contrary accuracy evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4582801569420454813, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 877933485, "recurrent_steps": 25265, "total_inference_macs": 878578965, "training_seconds": 133.66296841599979, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.45777399730097296}

RECENT RESULT
hypothesis: Independently averaging bands 14–15, 16–17, and 18–19 will retain at least 85% validation accuracy while reducing recurrent MACs by 7,503,705 versus the verified 18-input design.
change: Reduce the GRU input width from 18 to 17 by retaining bands 0–13 and averaging the three highest adjacent mel-band pairs; preserve the 99-unit state, 31-step schedule, classifier, and training procedure.
mechanism: Progressive adjacent high-frequency mel-band pooling
evidence_used: Successive pooling reduced input width from 20 to 19 and then 18 while achieving 85.52% and 85.28% accuracy; the latest reduction cost only 0.24 percentage points, making one additional locality-preserving pair compression the most informative next structural-cost probe.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4543661102369198011, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 35886, "peak_hidden_elements": 101888, "recurrent_macs": 870429780, "recurrent_steps": 25265, "total_inference_macs": 871075260, "training_seconds": 148.7672857081052, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.46785125732421873}

RECENT RESULT
hypothesis: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 28.3 million.
change: Omit the first two input frames while preserving the final 30 causal frames, 18-feature mel compression, 99-unit recurrent state, classifier, and training procedure.
mechanism: Second leading-frame omission
evidence_used: The current 31-step model achieved 85.28% accuracy, and omitting the first frame previously retained 85.03%. The prior 30-step attempt timed out without accuracy evidence, so removing one additional low-information boundary frame remains the most informative temporal-cost probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4435077871113243963, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850258530, "training_seconds": 151.97869416698813, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45974077096014665}

RECENT RESULT
hypothesis: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy on frames 3–31 while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 28.3 million.
change: Omit the first three input frames while preserving the 18-feature mel compression, 99-unit recurrent state, temporal summary, classifier, and training procedure.
mechanism: Third leading-frame omission
evidence_used: Omitting frames 0–1 retained 85.03% accuracy with 30 steps, and the failed 17-feature model indicates that another input-width reduction is less promising; removing one additional leading boundary frame is the smallest remaining temporal-cost probe.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4287354172806033113, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 821292615, "recurrent_steps": 23635, "total_inference_macs": 821938095, "training_seconds": 124.07881591608748, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.461545576785971}

RECENT RESULT
hypothesis: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy on frames 2–30 while reducing recurrent execution from 30 to 29 steps and recurrent MACs by approximately 28.3 million.
change: Omit the final input frame while preserving frames 2–30, the 18-feature mel compression, 99-unit recurrent state, temporal summary, classifier, and training procedure.
mechanism: Opposite-boundary frame omission
evidence_used: Omitting a third leading frame narrowly failed at 84.66%, while the current frames 2–31 schedule achieved 85.03%; testing the opposite temporal boundary is the smallest distinct 29-step probe and preserves the earliest frame known to be necessary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4287354172806033113, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 821292615, "recurrent_steps": 23635, "total_inference_macs": 821938095, "training_seconds": 138.0448544579558, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4676607939363257}

RECENT RESULT
hypothesis: Starting exact classifier checks at recurrent step 20 and exiting only when the current class is mathematically invariant to every possible remaining bounded GRU output will preserve validation accuracy at or above 85% while reducing dense MACs and mean recurrent steps below the verified 30-step model.
change: Keep the verified model and training unchanged, defer learned readout until step 20, then use classifier-weight bounds to skip remaining frames only when they cannot alter the predicted class.
mechanism: Bounded-state certified early exit
evidence_used: The 30-step model achieved 85.03%, while both fixed 29-step boundary schedules achieved 84.66%; this motivates retaining the final step for ambiguous recordings while safely omitting it—and potentially additional late steps—for recordings whose decision is already provably fixed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Processing frames 2–29 and 31 will retain at least 85% validation accuracy while reducing execution from 30 to 29 recurrent steps and recurrent MACs by approximately 28.3 million.
change: Omit frame 30 while preserving the earliest necessary frame and the final recurrent update.
mechanism: Penultimate-frame omission with endpoint preservation
evidence_used: Both 29-step boundary omissions narrowly failed at 84.66%, whereas frames 2–31 reached 85.03%; removing the penultimate frame tests temporal redundancy without losing either endpoint or the final context update.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4287354172806033113, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 821292615, "recurrent_steps": 23635, "total_inference_macs": 821938095, "training_seconds": 85.98116954113357, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4678003978144172}

RECENT RESULT
hypothesis: Replacing the 8-output classifier with seven learned logits plus one fixed reference logit will retain at least 85% validation accuracy because softmax is invariant to a shared logit offset, while reducing exact classifier MACs by 12.5% and learned parameters by 100.
change: Change the 99-to-8 classifier to 99-to-7 and append a zero reference-class logit during classification.
mechanism: Softmax reference-class parameterization
evidence_used: The current 18-feature, 30-step model meets the requirement at 85.03%; unlike prior hidden-width, readout-pooling, and frame-removal changes, this preserves the full softmax decision family and recurrent computation while structurally reducing dense inference cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4434657005876133578, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36083, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850177845, "training_seconds": 84.13995191711001, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.5107662083912481}

RECENT RESULT
hypothesis: Restricting classification to 98 of the 99 recurrent channels will retain at least 85% validation accuracy while reducing classifier inference by 6,520 MACs; the unobserved channel remains available as recurrent memory that can influence later observed outputs.
change: Preserve the verified 18-feature, 99-unit, 30-step GRU, but reduce the classifier input from 99 to 98 channels and exclude one summary channel from the final readout.
mechanism: Single-channel latent recurrent memory
evidence_used: Grouping the readout down to 33 features failed at 82.45%, while reducing the recurrent state itself to 98 units also failed; this conservative change preserves all 99 recurrent units and removes only one of 99 direct classifier inputs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4435043861801154235, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36175, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850252010, "training_seconds": 78.52013216703199, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.46358979582055215}

RECENT RESULT
hypothesis: Factoring the 99-to-8 classifier through seven learned logits will retain at least 85% validation accuracy while saving 43 MACs and 43 parameters per classifier, because eight-class softmax needs only seven independent logit-difference dimensions.
change: Replace the direct classifier with a bias-free 99-to-7 projection followed by a learned 7-to-8 decoder, preserving access to all 99 recurrent channels and trainable logits for every class.
mechanism: Symmetric rank-seven classifier factorization
evidence_used: The fixed-reference seven-logit classifier reached only 83.56%, showing that asymmetric treatment of one class harms learning; the proposed symmetric decoder retains the same structural dimensionality reduction while avoiding a fixed class, and unlike the failed 98-channel readout it discards no recurrent-state information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4434895071060761675, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36140, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850223485, "training_seconds": 77.3180928749498, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4670647088735381}

RECENT RESULT
hypothesis: A classifier-weight bound at step 29 will preserve every exited example’s final predicted class, maintaining at least 85% accuracy while reducing mean recurrent steps and exact MACs.
change: Defer the learned readout until the penultimate step, then exit examples whose class cannot change under any possible final bounded GRU output.
mechanism: Penultimate-step certified early exit
evidence_used: The 30-step model meets the target at 85.03%, while fixed 29-step schedules narrowly fail; the prior certified-exit attempt timed out without accuracy evidence, motivating this cheaper single-check variant that retains the final step only for uncertified examples.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4308117546380092537, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 29.137423312883435, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 825184503, "recurrent_steps": 23747, "total_inference_macs": 825918687, "training_seconds": 149.85506487521343, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4649688159029908}



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
