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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167504046643746721, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27521, "peak_hidden_elements": 85504, "recurrent_macs": 606166845, "recurrent_steps": 23635, "total_inference_macs": 607249165, "training_seconds": 89.7734805829823, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4487377236957199}
prior_hypothesis: An 83-unit GRU processing 29 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.2% versus the verified 82-unit, 30-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 91-unit GRU using both the temporal mean and final recurrent output will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the verified 92-unit model.
change: Reduce recurrent width from 92 to 91, preserve all 32 causal steps, and compensate for the smaller state by classifying the concatenated mean and terminal GRU outputs.
mechanism: Dual-timescale mean-and-terminal recurrent readout
evidence_used: The 92-unit full-frame model narrowly passed at 85.03%, while temporal subsampling failed at 16 and 20 steps; preserving every frame and adding a richer, low-cost readout is therefore safer than removing temporal observations.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4128526434427528113, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32353, "peak_hidden_elements": 93696, "recurrent_macs": 790302240, "recurrent_steps": 26080, "total_inference_macs": 791488880, "training_seconds": 100.87191074993461, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.42103065537528755}

RECENT RESULT
hypothesis: A 90-unit GRU retaining the mean-and-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.0% versus the verified 91-unit model.
change: Reduce the GRU state and temporal-summary width from 91 to 90 and resize the concatenated classifier input from 182 to 180, preserving all 32 causal steps and the verified training procedure.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The 91-unit dual-readout model achieved 86.01% accuracy at 791.49M total MACs, giving a 1.01-point margin above the requirement and motivating the next incremental structural reduction without sacrificing temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4046427955042943408, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31728, "peak_hidden_elements": 92672, "recurrent_macs": 774576000, "recurrent_steps": 26080, "total_inference_macs": 775749600, "training_seconds": 65.46588225010782, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.38939220217839343}

RECENT RESULT
hypothesis: An 88-unit GRU retaining the full 32-frame schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 4% versus the verified 90-unit model.
change: Reduce the GRU state and summary width from 90 to 88 and resize the concatenated classifier input from 180 to 176, leaving training and temporal coverage unchanged.
mechanism: Two-unit dual-timescale GRU width compression
evidence_used: The verified 90-unit model achieved 87.12% accuracy, 2.12 points above the requirement, at 775.75M total MACs; this unusually strong margin supports testing a two-unit structural reduction while preserving the readout and all 32 frames.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 92.70229770801961, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.40472788430430406}

RECENT RESULT
hypothesis: An 86-unit GRU preserving all 32 frames and the mean-plus-terminal readout will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.1% versus the verified 88-unit model.
change: Reduce the GRU state and summary width from 88 to 86 and resize the concatenated classifier input from 176 to 172, leaving temporal coverage and training unchanged.
mechanism: Two-unit dual-timescale GRU width compression
evidence_used: The 88-unit full-frame model achieved 86.87% accuracy—1.87 points above the requirement—at 744.74M total MACs, supporting another cautious two-unit structural reduction without revisiting temporal subsampling, which failed at 16 and 20 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3726196272406137448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 713235840, "recurrent_steps": 26080, "total_inference_macs": 714357280, "training_seconds": 114.65576104214415, "validation_accuracy": 0.8662576687116564, "validation_cross_entropy": 0.42406827774515915}

RECENT RESULT
hypothesis: Encoding every frame with a 64-unit RNN reset every four frames, then updating a 64-unit GRU from each ordered block descriptor, will retain at least 85% accuracy while reducing recurrent MACs from 713.2M to approximately 300.4M.
change: Replace the monolithic 86-unit full-rate GRU with a two-timescale recurrent hierarchy: a short-horizon local RNN processes all 32 frames, and a persistent GRU processes eight terminal-plus-mean block summaries.
mechanism: Hierarchical four-frame acoustic recurrence with a slower lexical GRU
evidence_used: Skipping frames at 16 and 20 steps failed, while 24 and 32 steps passed, indicating that retaining acoustic observations is load-bearing. This patch preserves every frame but challenges the assumption that the entire gated state must update at full rate.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1571502295649839760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 31920, "peak_hidden_elements": 131584, "recurrent_macs": 300441600, "recurrent_steps": 26080, "total_inference_macs": 301276160, "training_seconds": 97.52072033309378, "validation_accuracy": 0.8294478527607362, "validation_cross_entropy": 0.4815550821690472}

RECENT RESULT
hypothesis: An 84-unit GRU preserving all 32 frames and the mean-plus-terminal readout will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.2% versus the verified 86-unit model.
change: Reduce the GRU and summary width from 86 to 84 and resize the concatenated classifier input from 172 to 168, leaving temporal coverage and training unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 86-unit model achieved 86.63% accuracy, 1.63 points above the requirement, while every tested dual-readout width from 86 through 91 passed; this supports another cautious two-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3570977772028654184, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28104, "peak_hidden_elements": 86528, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 684600000, "training_seconds": 124.01448708283715, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.40037742567939993}

RECENT RESULT
hypothesis: An 82-unit GRU with the full 32-frame schedule and mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.3% versus the verified 84-unit model.
change: Reduce the GRU and summary width from 84 to 82 and resize the concatenated classifier input from 168 to 164, leaving training and temporal coverage unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 84-unit model achieved 87.73% accuracy, 2.73 points above the requirement, and all tested dual-readout widths from 84 through 91 passed; this supports another cautious two-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3419024165611784064, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26944, "peak_hidden_elements": 84480, "recurrent_macs": 654399360, "recurrent_steps": 26080, "total_inference_macs": 655468640, "training_seconds": 181.46486658300273, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4223508308270226}

RECENT RESULT
hypothesis: An 80-unit GRU with all 32 frames and the mean-plus-terminal readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 4.5% versus the verified 82-unit model.
change: Reduce the GRU and summary width from 82 to 80 and resize the concatenated classifier input from 164 to 160, leaving training and temporal coverage unchanged.
mechanism: Incremental dual-timescale GRU width compression
evidence_used: The verified 82-unit model achieved 86.01% accuracy at 655.47M total MACs, and every tested dual-readout width from 82 through 91 passed, making the next two-unit reduction the most direct probe of the minimum viable width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3270335453155527088, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25808, "peak_hidden_elements": 82432, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 626963200, "training_seconds": 68.48943950003013, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4433465594894316}

RECENT RESULT
hypothesis: An 81-unit GRU using all 32 frames and the mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.2% versus the verified 82-unit model.
change: Reduce the GRU state and summary width from 82 to 81 and resize the concatenated classifier input from 164 to 162, preserving temporal coverage and training.
mechanism: One-unit dual-timescale GRU width compression
evidence_used: The 82-unit model passed at 86.01% while the 80-unit model narrowly failed at 84.29%; testing 81 units directly resolves the remaining minimum-width boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3344271697638578933, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 640081440, "recurrent_steps": 26080, "total_inference_macs": 641137680, "training_seconds": 111.35484620905481, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.43995596294754125}

RECENT RESULT
hypothesis: The 81-unit GRU processing 31 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.1% versus the verified 32-step model.
change: Reduce the causal frame schedule from 32 to 31 unique increasing indices while preserving both recording endpoints, the mean-plus-terminal readout, and training procedure.
mechanism: Single-frame endpoint-preserving temporal compression
evidence_used: The 81-unit 32-step model passed at 85.64% accuracy, while 24-frame processing has previously passed at higher width; with the recurrent-width boundary already isolated between 80 and 81 units, removing one frame is the most cautious probe of the remaining temporal redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3239935379229828373, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 26373, "peak_hidden_elements": 83456, "recurrent_macs": 620078895, "recurrent_steps": 25265, "total_inference_macs": 621135135, "training_seconds": 71.20170233398676, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4404342464142782}

RECENT RESULT
hypothesis: An 82-unit GRU processing 30 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.1% versus the verified 81-unit, 31-step model.
change: Increase the GRU and readout width from 81 to 82 while reducing the causal schedule from 31 to 30 frames.
mechanism: Width-for-time Pareto trade
evidence_used: The 82-unit, 32-step model reached 86.01% accuracy, while the 81-unit, 31-step model narrowly passed at 85.15%; trading one additional hidden unit for one fewer recurrent step is the smallest unexplored diagonal reduction and still lowers exact dense MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3205683750709968874, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 26944, "peak_hidden_elements": 84480, "recurrent_macs": 613499400, "recurrent_steps": 24450, "total_inference_macs": 614568680, "training_seconds": 98.14459629217163, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44018247639474695}

RECENT RESULT
hypothesis: An 83-unit GRU processing 29 endpoint-preserving frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.2% versus the verified 82-unit, 30-step model.
change: Increase the GRU and dual-timescale readout width from 82 to 83 while reducing the causal schedule from 30 to 29 frames.
mechanism: Width-for-time Pareto continuation
evidence_used: The adjacent diagonal designs—81 units at 31 steps and 82 units at 30 steps—both passed at 85.15% and 85.52%, respectively; continuing that width-for-time trade is the smallest evidence-supported reduction in exact dense MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167504046643746721, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27521, "peak_hidden_elements": 85504, "recurrent_macs": 606166845, "recurrent_steps": 23635, "total_inference_macs": 607249165, "training_seconds": 89.7734805829823, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.4487377236957199}



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
