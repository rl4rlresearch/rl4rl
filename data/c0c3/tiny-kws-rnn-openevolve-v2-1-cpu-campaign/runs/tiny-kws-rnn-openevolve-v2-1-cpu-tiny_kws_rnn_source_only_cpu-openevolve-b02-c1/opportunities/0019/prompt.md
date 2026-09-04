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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3503209966281375417, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671608085, "training_seconds": 93.30611424986273, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45579669226898006}
prior_hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 3–31, while reducing recurrent execution from 30 to 29 steps and lowering total inference MACs below 694,740,230.

## Recent verification evidence

RECENT RESULT
hypothesis: A 98-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing exact inference MACs below the verified 100-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 98 while preserving the successful training procedure and full-frame schedule.
mechanism: Boundary-search hidden-width reduction at full temporal resolution
evidence_used: The 100-unit model reached 85.40% accuracy while the 96-unit model reached 84.29%; testing the midpoint is the most informative probe of the minimum viable width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 99.56158966687508, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4501637090203221}

RECENT RESULT
hypothesis: A 99-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing exact inference MACs below the verified 100-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 99 while preserving the full-frame schedule and training procedure.
mechanism: One-unit boundary-search width reduction
evidence_used: The 100-unit model met the target at 85.40% accuracy, while the 98-unit model missed it at 84.54%; width 99 is the only untested integer at the observed feasibility boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 87.23605174990371, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45291225690783166}

RECENT RESULT
hypothesis: Skipping only the first frame will preserve at least 85% validation accuracy while reducing recurrent steps from 32 to 31 and lowering exact inference MACs below the verified 99-unit full-frame model.
change: Keep the verified 99-unit GRU and training procedure unchanged, but process frames 1–31 instead of all 32 frames.
mechanism: Single-frame boundary trimming
evidence_used: The 99-unit 32-step model achieved 85.40% accuracy; unlike the failed 16- and 24-step schedules, this conservative change removes only one boundary frame and retains every remaining consecutive frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4661082503522968417, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893586375, "training_seconds": 94.2386459580157, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4587176457504553}

RECENT RESULT
hypothesis: Restoring all 32 frames but exiting high-confidence examples after 24–31 steps will retain at least 85% accuracy and reduce total MACs below 893,586,375 by reserving late recurrent updates for ambiguous recordings.
change: Replace fixed first-frame trimming with full-frame processing and a progressively relaxed confidence-based exit policy beginning after 24 recurrent steps.
mechanism: Confidence-adaptive recurrent-depth control
evidence_used: The 99-unit full-frame model reached 85.40%, while fixed 16- and 24-step schedules failed; this challenges the load-bearing assumption that every recording needs the same temporal budget by selectively shortening only confident examples.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Two independent 64-unit GRUs processing all 32 frames will retain at least 85% accuracy by preserving 128 aggregate recurrent features, while reducing exact recurrent MACs below the verified 99-unit, 31-step model.
change: Replace the single 99-unit GRU with two parallel 64-unit GRUs, concatenate their temporal summaries for classification, and restore the full-frame schedule.
mechanism: Parallel block-diagonal gated recurrence
evidence_used: The full-frame 128-unit GRU achieved 87.24% accuracy, while temporal subsampling repeatedly harmed accuracy; splitting 128 features into two structurally cheaper recurrent blocks preserves temporal resolution and aggregate width while eliminating costly cross-block recurrent connections.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4392370677619577616, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34096, "peak_hidden_elements": 131584, "recurrent_macs": 841236480, "recurrent_steps": 26080, "total_inference_macs": 842071040, "training_seconds": 145.33630258403718, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.43617060374628547}

RECENT RESULT
hypothesis: Three independent 43-unit GRUs processing all 32 frames will retain at least 85% validation accuracy by preserving 129 aggregate recurrent features, while reducing recurrent MACs per example-step from 32,256 to 24,381.
change: Replace the two parallel 64-unit GRUs with three parallel 43-unit GRUs, concatenate their outputs into a 129-feature temporal summary, and retain the full-frame schedule and training procedure.
mechanism: Three-way block-diagonal gated recurrence
evidence_used: Two parallel 64-unit GRUs achieved 85.89% accuracy with lower MACs than the single 99-unit model, showing that block-diagonal recurrence can preserve accuracy; adding a third smaller block preserves aggregate width while further reducing quadratic recurrent cost.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Two independent 60-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.7% versus the verified two-by-64 model.
change: Reduce both parallel GRUs from 64 to 60 units, resize their states and concatenated classifier input from 128 to 120, and preserve the full-frame schedule and training procedure.
mechanism: Narrower two-block gated recurrence
evidence_used: The two-by-64 full-frame model achieved 85.89% accuracy at 842,071,040 total MACs, while the three-by-43 attempt timed out; narrowing the successful two-module design tests its capacity boundary without adding recurrent-module overhead or discarding temporal frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3921953872794567008, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30528, "peak_hidden_elements": 123392, "recurrent_macs": 751104000, "recurrent_steps": 26080, "total_inference_macs": 751886400, "training_seconds": 66.71308599994518, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46019525820492235}

RECENT RESULT
hypothesis: A 60-unit GRU paired with a 59-unit GRU will retain at least 85% validation accuracy while reducing exact recurrent and classifier MACs below the verified two-by-60 model.
change: Reduce one parallel GRU by one unit and resize its state, temporal summary, and classifier input from 120 aggregate features to 119.
mechanism: Asymmetric two-block gated recurrence
evidence_used: The two-by-60 model achieved 85.28% accuracy at 751,886,400 total MACs after successfully narrowing the two-by-64 design; reducing only one block is the smallest untested structural cost reduction at this accuracy boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3865192330916823897, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 740228640, "recurrent_steps": 26080, "total_inference_macs": 741004520, "training_seconds": 91.05980095919222, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.44478286087878643}

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when skipping only the first frame, while reducing recurrent steps from 32 to 31 and total inference MACs below 741,004,520.
change: Preserve the successful asymmetric parallel-GRU model and training procedure, but process frames 1–31 instead of all 32 frames.
mechanism: Single-frame boundary trimming on asymmetric block-diagonal recurrence
evidence_used: The current 60+59 model achieved 86.50% accuracy, providing a 1.50-point margin, and prior first-frame trimming preserved 85.03% accuracy in the 99-unit GRU while reducing its recurrent execution to 31 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3744531542705007737, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 717096495, "recurrent_steps": 25265, "total_inference_macs": 717872375, "training_seconds": 98.94699670816772, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.44520491149527897}

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 2–31, while reducing recurrent execution from 31 to 30 steps and total inference MACs below 717,872,375.
change: Skip the first two input frames instead of only the first frame, preserving the model width and training procedure.
mechanism: Second boundary-frame trimming
evidence_used: Skipping one initial frame reduced the 60+59 model by 23,132,145 recurrent MACs while retaining 86.13% accuracy, leaving a 1.13-point margin above the requirement; removing one additional boundary frame is the largest conservative cost reduction directly supported by that result.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3623870754493191577, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 693964350, "recurrent_steps": 24450, "total_inference_macs": 694740230, "training_seconds": 115.44413508311845, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.45253378277176}

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 3–31, while reducing recurrent execution from 30 to 29 steps and lowering total inference MACs below 694,740,230.
change: Skip the first three input frames instead of the first two, preserving the model architecture and training procedure.
mechanism: Third boundary-frame trimming
evidence_used: Processing frames 2–31 achieved 85.64% accuracy at 694,740,230 MACs; the preceding one-frame trim also remained feasible, making one additional boundary trim the most direct test of the minimum viable temporal budget.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3503209966281375417, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671608085, "training_seconds": 93.30611424986273, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45579669226898006}

RECENT RESULT
hypothesis: Representing eight-class softmax probabilities with seven learned logits and one fixed-zero reference logit will retain at least 85% validation accuracy while reducing classifier MACs by 119 per example and learned parameters by 120.
change: Change the classifier from eight outputs to seven and append a fixed-zero eighth logit, preserving the full softmax function class and the verified 29-step recurrent architecture.
mechanism: Reference-class logit gauge fixing
evidence_used: The 60+59 GRU over frames 3–31 met the target at 85.03% accuracy; softmax is invariant to subtracting one class logit from every logit, so fixing that reference logit to zero removes redundant learned computation without reducing representational capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 69.34918704093434, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.481317831402176}



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
