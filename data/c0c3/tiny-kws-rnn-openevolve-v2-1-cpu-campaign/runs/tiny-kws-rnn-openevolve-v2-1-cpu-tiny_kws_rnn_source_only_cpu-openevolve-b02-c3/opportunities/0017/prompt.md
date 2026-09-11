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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4431107283926769866, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 848219400, "recurrent_steps": 24450, "total_inference_macs": 849497320, "training_seconds": 133.96698258304968, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.41693727458181556}
prior_hypothesis: A 98-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 31-frame design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4578588665885341461, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 876493380, "recurrent_steps": 25265, "total_inference_macs": 877771300, "training_seconds": 133.51997066708282, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40440371460709834}
prior_hypothesis: Omitting only the first frame will retain at least 85% accuracy from the current 87.12% baseline while reducing recurrent MACs and executed steps by 3.125%.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4267216408966407858, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 816825600, "recurrent_steps": 24450, "total_inference_macs": 818077440, "training_seconds": 120.89504654193297, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.3885523509394172}
prior_hypothesis: A 96-unit dual-readout GRU processing 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the current 98-unit, 29-frame model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 112-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 104 while retaining full temporal coverage and the existing training procedure.
mechanism: Full-resolution recurrent width bisection
evidence_used: The 96-unit model narrowly failed at 84.29%, while the 112-unit model passed at 86.13%; testing their 104-unit midpoint directly refines the smallest viable full-resolution width after reduced-frame designs failed.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 65.88476891699247, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}

RECENT RESULT
hypothesis: A 100-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 104-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 128 to 100 while retaining full temporal coverage and the existing training procedure.
mechanism: Full-resolution recurrent width bisection
evidence_used: The 96-unit model narrowly failed at 84.29%, while the 104-unit model passed at 85.89%; testing their 100-unit midpoint most directly refines the minimum viable recurrent width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 61.4100241670385, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4471566978407784}

RECENT RESULT
hypothesis: A 98-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 100-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 100 to 98 while preserving full temporal coverage and the existing training procedure.
mechanism: Full-resolution recurrent width bisection
evidence_used: The 100-unit model passed at 85.40% while the 96-unit model failed at 84.29%; testing the 98-unit midpoint most directly refines the minimum viable recurrent width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4722737135259119712, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 905406320, "training_seconds": 94.01462516700849, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4501637090203221}

RECENT RESULT
hypothesis: A 99-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while using fewer MACs and parameters than the qualified 100-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 99 while preserving full temporal coverage and training.
mechanism: Full-resolution one-unit recurrent width refinement
evidence_used: The 100-unit model passed at 85.40%, while the 98-unit model failed at 84.54%; testing 99 directly resolves the remaining width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4811331393252840977, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 921745440, "recurrent_steps": 26080, "total_inference_macs": 922390920, "training_seconds": 111.02356379199773, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45291225690783166}

RECENT RESULT
hypothesis: A 98-unit GRU classifying the concatenation of its final hidden state and mean temporal output will reach at least 85% accuracy while reducing total inference MACs below the qualified 99-unit mean-only model.
change: Reduce recurrent width to 98 and expand the classifier input to combine final-state and sequence-average representations.
mechanism: Dual-timescale recurrent readout
evidence_used: The 98-unit mean-only model narrowly missed at 84.54%, while 99 units passed at 85.40%; a richer readout may recover the small accuracy gap while retaining lower recurrent MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4726070047843913056, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 904767360, "recurrent_steps": 26080, "total_inference_macs": 906045280, "training_seconds": 104.06286133313552, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.39721636625886697}

RECENT RESULT
hypothesis: A 128-dimensional GRU-like state with rank-32 learned recurrent mixing and dual-timescale readout will retain at least 85% accuracy while reducing total inference MACs below the current 906,045,280.
change: Replace the dense 98-unit standard GRU with a custom GRU update whose recurrent gates share a 32-dimensional bottleneck, while retaining all 32 frames and concatenating mean and final states for classification.
mechanism: Shared-bottleneck low-rank GRU recurrence
evidence_used: Dropping frames failed even at 24 steps, showing full temporal coverage is load-bearing, while the 98-unit mean-plus-final readout reached 87.12%. This challenges the remaining assumption that every gate needs full-rank hidden-to-hidden mixing: a wider state preserves representation capacity while low-rank recurrent matrices structurally reduce MACs.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 90-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 906,045,280 to approximately 775,749,600.
change: Reduce the qualified dual-readout GRU width from 98 to 90 while preserving all 32 causal frames and expand the classifier input to the concatenated 180-dimensional mean/final representation.
mechanism: Dual-timescale readout with narrower recurrent state
evidence_used: The 98-unit dual-timescale model achieved 87.12% accuracy versus 84.54% for the 98-unit mean-only model, showing a 2.58-point gain and enough margin to test a meaningful eight-unit structural reduction without revisiting failed temporal subsampling.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 96-unit GRU classifying concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 98-unit model’s 906,045,280.
change: Reduce GRU width from 100 to 96 and use the qualified mean-plus-final recurrent readout with a 192-dimensional classifier input.
mechanism: Conservative dual-timescale width reduction
evidence_used: The 98-unit dual-timescale model achieved 87.12%, outperforming the 98-unit mean-only model by 2.58 points; a two-unit reduction is a conservative test of how much of that accuracy margin can be converted into structural MAC savings.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Omitting only the first frame will retain at least 85% accuracy from the current 87.12% baseline while reducing recurrent MACs and executed steps by 3.125%.
change: Process the most recent 31 of 32 frames, preserving uniform spacing and the final input frame.
mechanism: Single-edge-frame causal trimming
evidence_used: Reductions to 24 and 16 frames were too aggressive, but the qualified 98-unit dual-readout model has 2.12 percentage points of accuracy margin; a one-frame edge reduction is the most conservative untested temporal optimization.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4578588665885341461, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 876493380, "recurrent_steps": 25265, "total_inference_macs": 877771300, "training_seconds": 133.51997066708282, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.40440371460709834}

RECENT RESULT
hypothesis: A 98-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 31-frame design.
change: Replace the 99-unit mean-only model with the qualified 98-unit mean-plus-final readout and omit the first two input frames.
mechanism: Incremental edge-frame trimming with dual-timescale readout
evidence_used: The 98-unit dual-readout model achieved 87.12% over 32 frames and 86.50% over the most recent 31 frames; its remaining 1.50-point margin motivates testing one additional edge-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4431107283926769866, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 848219400, "recurrent_steps": 24450, "total_inference_macs": 849497320, "training_seconds": 133.96698258304968, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.41693727458181556}

RECENT RESULT
hypothesis: A 98-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs below the qualified 30-frame design.
change: Replace the current 100-unit mean-only GRU with the qualified 98-unit mean-plus-final architecture and omit the first three input frames.
mechanism: Incremental edge-frame trimming with dual-timescale readout
evidence_used: The same 98-unit dual-readout architecture passed at 87.12% with 32 frames, 86.50% with 31 frames, and 85.77% with 30 frames; its remaining 0.77-point margin supports testing one further edge-frame reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4283625901968198271, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 819945420, "recurrent_steps": 23635, "total_inference_macs": 821223340, "training_seconds": 151.8656526671257, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.42601396057503355}

RECENT RESULT
hypothesis: A 96-unit dual-readout GRU processing 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the current 98-unit, 29-frame model.
change: Reduce recurrent width from 98 to 96, resize the mean-plus-final classifier, and restore one input frame for a predicted total of approximately 818.1M inference MACs.
mechanism: Width–time tradeoff with dual-timescale readout
evidence_used: The 98-unit dual-readout model achieved 85.77% at 30 frames, providing 0.77 percentage points of margin; exchanging two hidden units for that additional frame is the smallest width–time tradeoff that improves on the current 821.2M-MAC design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4267216408966407858, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 816825600, "recurrent_steps": 24450, "total_inference_macs": 818077440, "training_seconds": 120.89504654193297, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.3885523509394172}



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
