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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4198364556070272933, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 44488, "peak_hidden_elements": 113152, "recurrent_macs": 804160500, "recurrent_steps": 18745, "total_inference_macs": 804877700, "training_seconds": 80.2774366659578, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.36332314146076977}
prior_hypothesis: A 110-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 111-unit model.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Doubling the optimizer updates with batch size 64 will let the narrowly failing 117-unit, 23-step GRU reach at least 85% accuracy while reducing inference MACs by approximately 1.6% versus the passing 118-unit model.
change: Reduce recurrent/readout width from 118 to 117 and training batch size from 128 to 64, preserving the verified schedule, loss, augmentation, optimizer, and learning-rate schedule.
mechanism: Smaller-batch optimization rescue
evidence_used: The identical 117-unit, 23-step architecture missed the threshold by only one validation example at 84.908%, while it passed at 24 and 25 steps; smaller batches provide twice as many parameter updates within the fixed 50,000-example exposure without adding inference cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4705770738956866973, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 49773, "peak_hidden_elements": 120320, "recurrent_macs": 901390815, "recurrent_steps": 18745, "total_inference_macs": 902153655, "training_seconds": 78.10909666703083, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.37536532162157304}

RECENT RESULT
hypothesis: A 116-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact inference MACs by approximately 1.6% versus the passing 117-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 117 to 116 while preserving the successful 23-frame schedule and batch-64 training procedure.
mechanism: One-channel recurrent-width trim under smaller-batch training
evidence_used: Batch-64 training raised the 117-unit, 23-step model from 84.908% to 86.503%; this margin motivates retesting the adjacent lower width under the improved optimization regime.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4631524159500996225, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 49000, "peak_hidden_elements": 119296, "recurrent_macs": 887163360, "recurrent_steps": 18745, "total_inference_macs": 887919680, "training_seconds": 74.78334024990909, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.3722479510161043}

RECENT RESULT
hypothesis: A 115-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 116-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 116 to 115 while preserving the successful schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: The 116-unit model achieved 86.380% accuracy at batch size 64, leaving a 1.38-point margin; the preceding 117-to-116 trim also retained accuracy while lowering MACs, motivating the next isolated width-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4557864240678673153, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 48233, "peak_hidden_elements": 118272, "recurrent_macs": 873048375, "recurrent_steps": 18745, "total_inference_macs": 873798175, "training_seconds": 84.63310170802288, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.3789182791680646}

RECENT RESULT
hypothesis: A 114-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 115-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 115 to 114 while preserving the successful schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: The adjacent width reductions from 117 to 116 and from 116 to 115 both passed; the 115-unit model retained 86.380% accuracy, leaving a 1.38-point margin for another isolated width-boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4484790982489897757, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 47472, "peak_hidden_elements": 117248, "recurrent_macs": 859045860, "recurrent_steps": 18745, "total_inference_macs": 859789140, "training_seconds": 95.49982433393598, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3690659224621357}

RECENT RESULT
hypothesis: A 113-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 114-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 114 to 113 while preserving the successful schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Successive batch-64 width reductions from 117 through 114 units all passed, and the 114-unit model achieved 86.871% accuracy, leaving a 1.87-point margin that motivates testing the adjacent lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4412304384934670037, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 46717, "peak_hidden_elements": 116224, "recurrent_macs": 845155815, "recurrent_steps": 18745, "total_inference_macs": 845892575, "training_seconds": 95.07249741605483, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.38322134193467217}

RECENT RESULT
hypothesis: A 112-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 113-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 113 to 112 while preserving the successful 23-frame schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 113 units passed, and the current 113-unit model achieved 86.012% accuracy, leaving a 1.012-point margin for testing the next lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4340404448012989993, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 831378240, "recurrent_steps": 18745, "total_inference_macs": 832108480, "training_seconds": 97.42092504212633, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.3917799405525067}

RECENT RESULT
hypothesis: A 111-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 112-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 112 to 111 while preserving the verified 23-frame schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 112 units passed, and the 112-unit model achieved 85.890% accuracy, leaving a 0.890-point margin that motivates testing the next lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4269091171724857625, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 45225, "peak_hidden_elements": 114176, "recurrent_macs": 817713135, "recurrent_steps": 18745, "total_inference_macs": 818436855, "training_seconds": 99.56368112494238, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.36047079636275403}

RECENT RESULT
hypothesis: A 110-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 111-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 111 to 110 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 111 units passed, and the 111-unit model achieved 86.748% accuracy, leaving a 1.748-point margin for testing the next lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4198364556070272933, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 44488, "peak_hidden_elements": 113152, "recurrent_macs": 804160500, "recurrent_steps": 18745, "total_inference_macs": 804877700, "training_seconds": 80.2774366659578, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.36332314146076977}



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
