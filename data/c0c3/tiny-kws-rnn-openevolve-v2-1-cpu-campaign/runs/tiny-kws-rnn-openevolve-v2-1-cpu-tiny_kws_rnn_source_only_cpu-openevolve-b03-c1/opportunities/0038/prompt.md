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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1758132646498671669, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18189, "peak_hidden_elements": 149504, "recurrent_macs": 335158155, "recurrent_steps": 22005, "total_inference_macs": 337055475, "training_seconds": 88.30740637495182, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4447639886586944}
prior_hypothesis: One 33-unit GRU head plus two 32-unit heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.6% versus the verified 33/33/32 model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing frames 2–28 while augmenting the endpoint-and-mean classifier with an online temporal maximum will retain at least 85% validation accuracy and reduce total inference MACs below the verified 28-step model.
change: Remove one additional terminal frame, track the elementwise maximum recurrent output, and classify from the final state, temporal mean, and temporal maximum.
mechanism: Terminal trimming with online temporal-max readout
evidence_used: Frames 2–29 achieved 85.28% at 28 steps, while removing leading frame 2 previously caused failure; preserving the early boundary and trimming the terminal boundary is therefore the strongest 27-step candidate, with max pooling adding complementary trajectory evidence at only a small classifier cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2762916516369401253, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 528120000, "recurrent_steps": 22005, "total_inference_macs": 529684800, "training_seconds": 38.68669312517159, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.40201921521520323}

RECENT RESULT
hypothesis: A 76-unit GRU retaining the successful 27-step endpoint, mean, and temporal-maximum readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 8.8%.
change: Reduce the GRU hidden width from 80 to 76 and resize all recurrent state summaries and the classifier input from 240 to 228 features.
mechanism: Temporal-max-assisted recurrent width reduction
evidence_used: The 80-unit 27-step model with temporal maximum achieved 87.61%, substantially above the threshold, while the 76-unit mean-only model narrowly missed at 84.79%; the complementary maximum readout provides a plausible margin for revisiting that width at materially lower cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2520090028048798221, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 481645440, "recurrent_steps": 22005, "total_inference_macs": 483132000, "training_seconds": 31.469110749894753, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.42248674638432226}

RECENT RESULT
hypothesis: A 78-unit GRU retaining the successful 27-step endpoint, mean, and temporal-maximum readout will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 4.4%.
change: Reduce the GRU hidden width from 80 to 78 and resize its recurrent states and classifier input from 240 to 234 features.
mechanism: Temporal-max-assisted width interpolation
evidence_used: The 80-unit temporal-maximum model achieved 87.61%, while the 76-unit version narrowly missed at 84.79%; testing their midpoint locates the capacity boundary for this specific readout at lower cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2640125895069466065, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 25320, "peak_hidden_elements": 120320, "recurrent_macs": 504618660, "recurrent_steps": 22005, "total_inference_macs": 506144340, "training_seconds": 51.69099379098043, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4260737694114264}

RECENT RESULT
hypothesis: A 77-unit GRU with the successful 27-step endpoint, mean, and temporal-maximum readout will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the verified 78-unit model.
change: Reduce the GRU hidden width from 78 to 77 and resize its recurrent summaries and classifier input from 234 to 231 features, preserving the frame schedule and training procedure.
mechanism: Single-unit recurrent width boundary search
evidence_used: The 78-unit temporal-maximum model achieved 85.64%, while the 76-unit version narrowly missed at 84.79%; the untested 77-unit midpoint is the smallest structural cost reduction and directly resolves the remaining width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2579763617274223725, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 24765, "peak_hidden_elements": 118784, "recurrent_macs": 493066035, "recurrent_steps": 22005, "total_inference_macs": 494572155, "training_seconds": 52.10705554182641, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.409486248741852}

RECENT RESULT
hypothesis: Two independent 48-unit GRU heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the verified 77-unit GRU.
change: Replace the single densely coupled recurrent state with two continuous 48-unit GRU states that independently encode every scheduled frame, then fuse their endpoints, temporal means, and temporal maxima for classification.
mechanism: Parallel block-diagonal recurrent ensemble
evidence_used: The 77-unit model achieved 86.38%, providing accuracy margin, while its dense hidden-to-hidden matrix dominates cost. Unlike the failed chunk hierarchy, this design never resets local memory or compresses chunks through an intermittently updated bottleneck; both heads preserve the proven 27-step schedule and readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2257674175964516997, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 22512, "peak_hidden_elements": 147968, "recurrent_macs": 430945920, "recurrent_steps": 22005, "total_inference_macs": 432823680, "training_seconds": 64.13959562499076, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4356593313392686}

RECENT RESULT
hypothesis: Three independent 36-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.4% versus two 48-unit heads, because the aggregate recurrent representation grows from 96 to 108 features while each costly hidden-to-hidden matrix becomes smaller.
change: Replace the two 48-unit GRU heads with three 36-unit heads and resize the state summaries and endpoint/mean/maximum classifier input accordingly.
mechanism: Three-way block-diagonal recurrent ensemble
evidence_used: Two independent 48-unit heads achieved 86.13% accuracy and substantially outperformed the denser 77-unit GRU on MACs, showing that partitioned recurrent state preserves accuracy; a third smaller block tests whether further structural factorization can exploit the same effect with more aggregate features and fewer MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2093613254443706637, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 21432, "peak_hidden_elements": 166400, "recurrent_macs": 399258720, "recurrent_steps": 22005, "total_inference_macs": 401371200, "training_seconds": 82.76438416587189, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4320945224878978}

RECENT RESULT
hypothesis: Four independent 29-unit GRU heads will maintain at least 85% validation accuracy while reducing total inference MACs by approximately 5.9% versus three 36-unit heads, because aggregate recurrent width increases from 108 to 116 while each hidden-to-hidden block becomes smaller.
change: Replace the three 36-unit GRU heads with four 29-unit heads and resize the recurrent state summaries and endpoint/mean/maximum classifier input accordingly.
mechanism: Four-way block-diagonal recurrent ensemble
evidence_used: Moving from two 48-unit heads to three 36-unit heads retained 86.01% accuracy while reducing recurrent MACs by 7.4%; this continues the successful factorization pattern with greater aggregate representation and lower structural cost.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1969088158227196005, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 20580, "peak_hidden_elements": 178688, "recurrent_macs": 375229260, "recurrent_steps": 22005, "total_inference_macs": 377498220, "training_seconds": 105.52508333418518, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.45010555711991945}

RECENT RESULT
hypothesis: Three independent 35-unit GRU heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 4.5% versus the verified three-by-36 model.
change: Reduce each GRU head from 36 to 35 units and resize the hidden state, trajectory summaries, and classifier input from 108/324 to 105/315 features.
mechanism: Three-head recurrent width boundary reduction
evidence_used: Three 36-unit heads achieved 86.01%, while four 29-unit heads fell to 84.29% despite greater aggregate width; preserving the successful three-head topology and removing only one unit per head is the most direct capacity-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1999301180854900953, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 20523, "peak_hidden_elements": 161792, "recurrent_macs": 381236625, "recurrent_steps": 22005, "total_inference_macs": 383290425, "training_seconds": 79.32638879097067, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.43087680559216834}

RECENT RESULT
hypothesis: Three independent 34-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.6% versus the verified three-by-35 model.
change: Reduce every GRU head from 35 to 34 units and resize the hidden state, trajectory summaries, and classifier input from 105/315 to 102/306 features.
mechanism: Three-head recurrent width boundary reduction
evidence_used: Three 35-unit heads achieved 86.26% accuracy at 383.29M total MACs, leaving 1.26 percentage points of margin; a one-unit-per-head reduction is the smallest structural change that tests the remaining capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1907055172975545777, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 19632, "peak_hidden_elements": 157184, "recurrent_macs": 363610620, "recurrent_steps": 22005, "total_inference_macs": 365605740, "training_seconds": 73.72847845917568, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.42577869438686256}

RECENT RESULT
hypothesis: Three independent 33-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.7% versus the verified three-by-34 model.
change: Reduce each GRU head from 34 to 33 units and resize the hidden state, trajectory summaries, and classifier input from 102/306 to 99/297 features.
mechanism: Three-head recurrent width boundary reduction
evidence_used: Three-by-34 achieved 85.64% accuracy at 365.61M total MACs after three-by-35 achieved 86.26%; another one-unit-per-head reduction is the smallest symmetric structural change that directly tests the remaining capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1816875230805641109, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18759, "peak_hidden_elements": 152576, "recurrent_macs": 346380705, "recurrent_steps": 22005, "total_inference_macs": 348317145, "training_seconds": 75.28419691603631, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4422609294119057}

RECENT RESULT
hypothesis: Two 33-unit GRU heads plus one 32-unit head will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.6% versus the verified three-by-33 model.
change: Reduce only the third GRU head to 32 units and resize the separate hidden states, trajectory summaries, and classifier input from 99/297 to 98/294 features.
mechanism: Asymmetric single-head width reduction
evidence_used: Three 33-unit heads achieved 85.15% accuracy; reducing all three heads by one unit still passed in the preceding 34-to-33 trial, so removing a single unit from only one head is the smallest available structural capacity reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1787503938652156389, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18474, "peak_hidden_elements": 151040, "recurrent_macs": 340769430, "recurrent_steps": 22005, "total_inference_macs": 342686310, "training_seconds": 96.50850574998185, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4324811876917178}

RECENT RESULT
hypothesis: One 33-unit GRU head plus two 32-unit heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.6% versus the verified 33/33/32 model.
change: Reduce the second GRU head from 33 to 32 units and resize its hidden state, aggregate summaries, and classifier input from 98/294 to 97/291 features.
mechanism: Asymmetric single-head width boundary reduction
evidence_used: The verified 33/33/32 model achieved 86.13% accuracy, leaving 1.13 percentage points of margin; reducing one additional head by a single unit is the smallest structural change that continues the successful asymmetric width search.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1758132646498671669, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 18189, "peak_hidden_elements": 149504, "recurrent_macs": 335158155, "recurrent_steps": 22005, "total_inference_macs": 337055475, "training_seconds": 88.30740637495182, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4447639886586944}



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
