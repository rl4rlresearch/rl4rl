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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4661082503522968417, "maximum_recurrent_steps": 31, "mean_recurrent_steps": 31.0, "median_recurrent_steps": 31, "optimizer_steps": 397, "p95_recurrent_steps": 31, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 892940895, "recurrent_steps": 25265, "total_inference_macs": 893586375, "training_seconds": 94.2386459580157, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4587176457504553}
prior_hypothesis: Skipping only the first frame will preserve at least 85% validation accuracy while reducing recurrent steps from 32 to 31 and lowering exact inference MACs below the verified 99-unit full-frame model.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing every other frame will preserve at least 85% validation accuracy because the baseline’s 32-frame representation is temporally redundant, while halving recurrent steps and recurrent MACs.
change: Change the frame schedule from all 32 frames to the 16 even-indexed frames; model capacity and training remain unchanged.
mechanism: Uniform 2× causal frame subsampling
evidence_used: The current 128-unit GRU achieves 87.24% accuracy, providing a 2.24-point margin above the requirement, while its 32 recurrent steps account for virtually all 1.483B inference MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 68.51940975012258, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.5077147314153566}

RECENT RESULT
hypothesis: Using 24 evenly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and MACs by 25% versus the verified 32-step baseline.
change: Replace the full 32-frame schedule with 24 uniformly spaced indices spanning the complete recording.
mechanism: Uniform 24-step causal frame subsampling
evidence_used: The 32-step model reached 87.24% accuracy, while aggressive 16-step subsampling reached 84.17%; testing the midpoint directly brackets the accuracy-efficiency boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 89.01151354215108, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.4806923895525786}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 22% versus the verified 128-unit baseline.
change: Reduce the GRU state and classifier width from 128 to 112 without changing the successful full-frame schedule or training procedure.
mechanism: Hidden-width reduction with full temporal resolution
evidence_used: Temporal subsampling failed at both 16 steps (84.17%) and 24 steps (83.19%), while the full 32-step model achieved 87.24%; preserving every frame and reducing recurrent width tests an orthogonal efficiency axis with less temporal-information loss.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 126.90054870815948, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4414740345960746}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13% versus the verified 112-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 112 to 104 while preserving the successful full-frame training procedure.
mechanism: Moderate hidden-width reduction at full temporal resolution
evidence_used: The 112-unit full-frame model achieved 86.13% accuracy, while temporal subsampling failed; this motivates preserving all 32 frames and testing the next meaningful reduction along the successful width-efficiency axis.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 121.98937408393249, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will maintain at least 85% validation accuracy while reducing recurrent MACs by approximately 13% versus the verified 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 96 while preserving the successful full-frame schedule and training procedure.
mechanism: Hidden-width reduction at full temporal resolution
evidence_used: Reducing width from 112 to 104 lowered accuracy by only 0.25 percentage points, from 86.13% to 85.89%, while temporal subsampling failed; another eight-unit reduction directly extends the successful width-efficiency trend without discarding frames.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 95.08381520793773, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.44714848805058954}

RECENT RESULT
hypothesis: A 100-unit GRU processing all 32 frames will achieve at least 85% validation accuracy while reducing exact inference MACs versus the verified 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier input width from 104 to 100, preserving the successful full-frame schedule and training procedure.
mechanism: Boundary-search hidden-width reduction at full temporal resolution
evidence_used: The 104-unit model achieved 85.89% accuracy while the 96-unit model achieved 84.29%; testing the midpoint is the most informative next probe of the minimum viable width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4900741874736715528, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 939532000, "training_seconds": 97.08712229179218, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4471566978407784}

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
