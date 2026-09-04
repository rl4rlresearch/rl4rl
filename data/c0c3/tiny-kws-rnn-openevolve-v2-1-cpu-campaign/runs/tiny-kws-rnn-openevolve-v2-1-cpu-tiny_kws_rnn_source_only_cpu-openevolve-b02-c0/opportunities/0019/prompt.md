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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4435077871113243963, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850258530, "training_seconds": 151.97869416698813, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45974077096014665}
prior_hypothesis: The verified 18-feature, 99-unit GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and recurrent MACs by approximately 28.3 million.

## Recent verification evidence

RECENT RESULT
hypothesis: A 99-unit full-resolution GRU will retain at least 85% validation accuracy while reducing exact recurrent MACs below the verified 100-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 99 while preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width boundary reduction
evidence_used: The 100-unit model achieved 85.40% accuracy while the 96-unit model achieved 84.29%; a one-unit reduction is the smallest available probe of the accuracy boundary and preserves full temporal resolution, unlike the unsuccessful 24- and 16-step designs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 111.973967500031, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45291225690783166}

RECENT RESULT
hypothesis: A 98-unit full-resolution GRU will retain at least 85% validation accuracy while reducing exact recurrent and classifier MACs below the verified 99-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 99 to 98 while preserving all 32 causal frames and the established training procedure.
mechanism: One-unit recurrent-width boundary reduction
evidence_used: The 99-unit model achieved 85.40% accuracy, while the 96-unit model achieved 84.29%; testing 98 units is the most informative next probe of the minimum viable width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 92.19683279120363, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4501637090203221}

RECENT RESULT
hypothesis: Processing 31 frames with the verified 99-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 3.125%.
change: Omit only the earliest input frame while retaining the remaining 31 causal frames, including the final frame.
mechanism: Single leading-frame omission
evidence_used: The 99-unit 32-step model achieved 85.40% accuracy; prior 24- and 16-step schedules were too aggressive, so removing one likely low-information boundary frame is the smallest temporal-cost probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4661082503522968417, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893586375, "training_seconds": 107.63260129187256, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4587176457504553}

RECENT RESULT
hypothesis: Pooling the verified 99-unit recurrent summary into 33 groups before classification will retain at least 85% accuracy while reducing exact classifier MACs and learned parameters.
change: Preserve the 99-unit GRU and 31-step schedule, but average each consecutive group of three recurrent features and replace the 99-to-8 classifier with a 33-to-8 classifier.
mechanism: Parameter-free grouped recurrent readout
evidence_used: The 99-unit, 31-step model achieved 85.03% accuracy, while even a 98-unit model using all 32 frames achieved only 84.54%; this motivates preserving recurrent capacity and structurally reducing the readout instead.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4658837888925046369, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36249, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893156055, "training_seconds": 110.998170417035, "validation_accuracy": 0.8245398773006135, "validation_cross_entropy": 0.5709395519794862}

RECENT RESULT
hypothesis: Merging only the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by 7,503,705 versus the verified 99-unit, 31-step design.
change: Preserve the 99-unit state, 31-frame schedule, and full classifier while replacing the GRU’s 20-dimensional input with 19 features formed by retaining bands 0–17 and averaging bands 18–19.
mechanism: Adjacent high-frequency mel-band pooling
evidence_used: The 99-unit, 31-step model met the threshold at 85.03%, whereas reducing hidden width to 98 or compressing the recurrent readout failed; this motivates preserving recurrent capacity and readout information while conservatively exploiting locality in the ordered mel inputs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4621942036471711615, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36480, "peak_hidden_elements": 101888, "recurrent_macs": 885437190, "recurrent_steps": 25265, "total_inference_macs": 886082670, "training_seconds": 181.31405062507838, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4569374412115366}

RECENT RESULT
hypothesis: Independently averaging mel bands 16–17 and 18–19 will retain at least 85% validation accuracy while reducing recurrent MACs by another 7,503,705 versus the verified 19-input design.
change: Reduce the GRU input from 19 to 18 features by retaining bands 0–15 and averaging each of the two highest adjacent band pairs; preserve the 99-unit state, 31-step schedule, and classifier.
mechanism: Second adjacent high-frequency mel-band pooling
evidence_used: The current 19-feature model achieved 85.52% accuracy after pooling bands 18–19, improving on the uncompressed 31-step model’s 85.03%; this supports another conservative, locality-preserving input reduction rather than reducing recurrent width or readout capacity, both of which previously failed.
result: training did not finish within the verification time limit

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
