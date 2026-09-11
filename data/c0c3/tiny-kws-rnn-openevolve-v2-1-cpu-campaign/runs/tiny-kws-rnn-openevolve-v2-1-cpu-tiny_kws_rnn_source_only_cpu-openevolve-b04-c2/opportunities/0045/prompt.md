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
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2231146911964034612, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 27432, "peak_hidden_elements": 86528, "recurrent_macs": 427190400, "recurrent_steps": 16300, "total_inference_macs": 427738080, "training_seconds": 131.98087516706437, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.42247594353611484}
prior_hypothesis: An 84-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 427,738,080.

REFERENCE DESIGN 1
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2279389121163303013, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28013, "peak_hidden_elements": 87552, "recurrent_macs": 436432500, "recurrent_steps": 16300, "total_inference_macs": 436986700, "training_seconds": 112.64498849981464, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.41102057146879795}
prior_hypothesis: An 85-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 436,986,700.

REFERENCE DESIGN 2
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 113.07924199989066, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.43155082281381807}
prior_hypothesis: An 86-unit GRU processing frames 4–23 with a parameter-free 25% final-state residual will achieve at least 85% validation accuracy while retaining approximately 446,333,120 total MACs.

REFERENCE DESIGN 3
verified_results: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2444402303504125855, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 468061020, "recurrent_steps": 17115, "total_inference_macs": 468621740, "training_seconds": 136.71907783392817, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.45798510217959165}
prior_hypothesis: The 86-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 468,621,740.

## Recent verification evidence

RECENT RESULT
hypothesis: The 97-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing standard inference from 22 to 21 recurrent steps and total MACs to approximately 583.3 million.
change: Remove frame 25 from the current schedule, processing 21 contiguous frames for standard recordings while preserving valid schedules for shorter inputs.
mechanism: Incremental trailing-frame trim at reduced recurrent width
evidence_used: The current 97-unit, 22-step model achieved 86.50% accuracy, a 1.50-point margin; the analogous 103-unit batch-size-32 trim from 22 to 21 steps lost only 0.74 points and still qualified.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 3042825907870834613, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 35453, "peak_hidden_elements": 99840, "recurrent_macs": 582714405, "recurrent_steps": 17115, "total_inference_macs": 583346845, "training_seconds": 114.14717325009406, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.44280826123945555}

RECENT RESULT
hypothesis: A 91-unit GRU trained at batch size 16 over frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs to approximately 543,926,110.
change: Narrow the GRU and classifier from 103 to 91 units while expanding the standard schedule from 20 to 22 contiguous frames.
mechanism: Recurrent width-for-context exchange
evidence_used: The 97-unit, 22-step design achieved 86.50% accuracy and 611,095,150 MACs; another six-unit width reduction preserves its temporal context and tests a structurally cheaper model against a 1.50-point accuracy margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2837201355893873265, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 31625, "peak_hidden_elements": 93696, "recurrent_macs": 543332790, "recurrent_steps": 17930, "total_inference_macs": 543926110, "training_seconds": 111.14460374996997, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4392411776115558}

RECENT RESULT
hypothesis: An 85-unit GRU trained at batch size 16 on frames 4–25 will retain at least 85% validation accuracy while reducing total inference MACs from 543,926,110 to approximately 480,629,950.
change: Replace the current 103-unit, 21-step model with an 85-unit GRU processing 22 contiguous frames, and use batch size 16.
mechanism: Incremental recurrent-width reduction
evidence_used: Reducing the 22-step GRU from 97 to 91 units lowered accuracy by only 0.37 points, from 86.50% to 86.13%, leaving 1.13 points of margin; repeating that six-unit reduction is the most direct test of a substantially cheaper frontier.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2507038954126867893, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 28013, "peak_hidden_elements": 87552, "recurrent_macs": 480075750, "recurrent_steps": 17930, "total_inference_macs": 480629950, "training_seconds": 114.5374007918872, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4446986379798936}

RECENT RESULT
hypothesis: An 86-unit GRU trained at batch size 16 on frames 4–25 will achieve at least 85% validation accuracy while reducing total inference MACs to approximately 490.9 million.
change: Reduce the GRU and classifier from 103 to 86 units, retain the qualified 22-frame schedule, and use batch size 16.
mechanism: Minimal recurrent-width rebound
evidence_used: The 85-unit design missed qualification by only 0.092 percentage points at 480.6 million MACs, while the 91-unit design achieved 86.13%; testing the adjacent 86-unit width is the smallest capacity increase and cheapest unresolved frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2560663136964334490, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 3127, "p95_recurrent_steps": 22, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 490349640, "recurrent_steps": 17930, "total_inference_macs": 490910360, "training_seconds": 148.6919713751413, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4445762213022431}

RECENT RESULT
hypothesis: The 86-unit GRU trained at batch size 16 on frames 4–24 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs to approximately 468,621,740.
change: Remove frame 25 from the standard schedule while preserving valid schedules for shorter inputs.
mechanism: Incremental trailing-frame trim at the 86-unit frontier
evidence_used: The current 86-unit, 22-step model achieved 86.75% accuracy, leaving 1.75 points of margin; the analogous 97-unit trim from 22 to 21 steps retained 86.01% accuracy and reduced MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2444402303504125855, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 3127, "p95_recurrent_steps": 21, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 468061020, "recurrent_steps": 17115, "total_inference_macs": 468621740, "training_seconds": 136.71907783392817, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.45798510217959165}

RECENT RESULT
hypothesis: The qualified 86-unit GRU will retain at least 85% validation accuracy on frames 4–23 while reducing execution from 21 to 20 recurrent steps and total inference MACs to approximately 446 million.
change: Change the current model to an 86-unit GRU and classifier, and shorten the standard schedule to 20 contiguous frames while retaining batch size 16 and the established training procedure.
mechanism: Incremental trailing-frame trim at fixed recurrent width
evidence_used: The 86-unit, 21-step design achieved 85.15% accuracy at 468,621,740 MACs; analogously, trimming the 103-unit model from 21 to 20 steps reduced accuracy by only 0.12 percentage points and still qualified.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 131.9196069170721, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4744566279686302}

RECENT RESULT
hypothesis: An 87-unit GRU trained on frames 4–23 will recover at least one validation example over the 86-unit, 20-step design, reaching at least 85% accuracy while reducing total inference MACs from the qualified 468,621,740 to approximately 455,777,340.
change: Replace the 97-unit GRU and classifier with 87 units and shorten the standard schedule from 21 to 20 contiguous frames, retaining batch size 16 and the established training procedure.
mechanism: Minimal width rebound at the 20-step frontier
evidence_used: The 86-unit, 20-step design missed qualification by only one validation example at 84.91%, while the adjacent 86-unit, 21-step design reached 85.15%; adding one hidden unit is the smallest capacity increase available while preserving the cheaper 20-step execution.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2377403958605877233, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 29193, "peak_hidden_elements": 89600, "recurrent_macs": 455210100, "recurrent_steps": 16300, "total_inference_macs": 455777340, "training_seconds": 116.49121791590005, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.5012824590952119}

RECENT RESULT
hypothesis: An 86-unit GRU processing frames 5–24 will reach at least 85% validation accuracy while retaining the 20-step cost of approximately 446,333,120 MACs.
change: Use batch size 16, reduce the GRU to 86 units, and replace the unsuccessful frames 4–23 schedule with 20 contiguous frames shifted one step later.
mechanism: Late-context-preserving leading-frame trim
evidence_used: The 86-unit frames 4–24 model qualified at 85.15%, while removing frame 24 missed qualification by only one example; the failed 87-unit rebound suggests preserving the final frame is more informative than adding capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 109.89118841709569, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4928817468186829}

RECENT RESULT
hypothesis: Processing frames 4–22 and 24 with the qualified 86-unit GRU will preserve at least 85% accuracy while reducing inference to 20 recurrent steps and approximately 446,333,120 total MACs.
change: Keep both endpoints of the qualified frames 4–24 window while omitting the redundant penultimate frame, replacing the unsuccessful contiguous 20-frame schedules.
mechanism: Full-span late-frame substitution
evidence_used: Removing frame 24 missed qualification by only one validation example, while removing frame 4 performed worse; this suggests retaining the final frame and dropping an adjacent interior frame is the most targeted unresolved 20-step schedule.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 104.73653479106724, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.47286453715131327}

RECENT RESULT
hypothesis: An 86-unit GRU processing frames 4–23 with a parameter-free 25% final-state residual will achieve at least 85% validation accuracy while retaining approximately 446,333,120 total MACs.
change: Reduce the GRU to 86 units, use the best-performing tested 20-frame schedule, and bias classification toward the context-complete final recurrent state without adding learned operations.
mechanism: Late-state-biased temporal pooling
evidence_used: The 86-unit frames 4–23 model missed qualification by only one validation example at 84.91%, while alternative 20-frame schedules performed worse; changing its zero-MAC temporal aggregation is therefore more informative than further schedule changes.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2328141470043917220, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28600, "peak_hidden_elements": 88576, "recurrent_macs": 445772400, "recurrent_steps": 16300, "total_inference_macs": 446333120, "training_seconds": 113.07924199989066, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.43155082281381807}

RECENT RESULT
hypothesis: An 85-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 436,986,700.
change: Reduce the qualified residual-pooled 20-step architecture from 86 to 85 hidden units while preserving batch size 16 and the established training procedure.
mechanism: One-unit recurrent-width reduction with late-state-biased pooling
evidence_used: The 86-unit residual-pooled 20-step model achieved 86.38% at 446,333,120 MACs, improving the same-width mean-pooled model from 84.91%; this margin motivates the smallest unresolved structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2279389121163303013, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 28013, "peak_hidden_elements": 87552, "recurrent_macs": 436432500, "recurrent_steps": 16300, "total_inference_macs": 436986700, "training_seconds": 112.64498849981464, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.41102057146879795}

RECENT RESULT
hypothesis: An 84-unit GRU processing frames 4–23 with the qualified 25% final-state residual will retain at least 85% validation accuracy while reducing total inference MACs to approximately 427,738,080.
change: Reduce the qualified 85-unit architecture to 84 hidden units, retain its 20-frame schedule and residual pooling, and use batch size 16.
mechanism: One-unit recurrent-width reduction on the residual-pooled frontier
evidence_used: The 85-unit residual-pooled model qualified at 85.40% and 436,986,700 MACs; reducing the qualified 86-unit model by one unit also preserved qualification, making the adjacent 84-unit width the cheapest unresolved frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 16, "examples_processed": 50000, "inference_cost": 2231146911964034612, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 3127, "p95_recurrent_steps": 20, "parameters": 27432, "peak_hidden_elements": 86528, "recurrent_macs": 427190400, "recurrent_steps": 16300, "total_inference_macs": 427738080, "training_seconds": 131.98087516706437, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.42247594353611484}



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
