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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3719704744227515017, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 712441215, "recurrent_steps": 18745, "total_inference_macs": 713112775, "training_seconds": 64.17969787493348, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.3853984996585027}
prior_hypothesis: A 103-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 104-unit model.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 109-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 110-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 110 to 109 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 110 units passed, and the 110-unit model achieved 86.994% accuracy, leaving a 1.994-point margin that motivates testing the next lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4128224601049235917, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 43757, "peak_hidden_elements": 112128, "recurrent_macs": 790720335, "recurrent_steps": 18745, "total_inference_macs": 791431015, "training_seconds": 98.48433554219082, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.37416679522742524}

RECENT RESULT
hypothesis: A 108-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 109-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 109 to 108 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: The 109-unit model achieved 87.607% accuracy, and every adjacent batch-64 reduction from 117 through 109 units passed, making the next one-channel trim the most informative lower-cost boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 4058671306661746577, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 43032, "peak_hidden_elements": 111104, "recurrent_macs": 777392640, "recurrent_steps": 18745, "total_inference_macs": 778096800, "training_seconds": 88.73811449995264, "validation_accuracy": 0.8785276073619632, "validation_cross_entropy": 0.3791235520064465}

RECENT RESULT
hypothesis: A 107-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 108-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 108 to 107 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: The 108-unit model achieved 87.853% accuracy, and every adjacent batch-64 width reduction from 117 through 108 units passed, making the next one-channel trim the clearest lower-cost boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3989704672907804913, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 42313, "peak_hidden_elements": 110080, "recurrent_macs": 764177415, "recurrent_steps": 18745, "total_inference_macs": 764875055, "training_seconds": 81.7119608328212, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.3903882266553633}

RECENT RESULT
hypothesis: A 106-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 107-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 107 to 106 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 107 units passed; the 107-unit model achieved 85.767% accuracy, motivating the next isolated lower-cost boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3921324699787410925, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 41600, "peak_hidden_elements": 109056, "recurrent_macs": 751074660, "recurrent_steps": 18745, "total_inference_macs": 751765780, "training_seconds": 72.24660720909014, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.3877136979366373}

RECENT RESULT
hypothesis: A 105-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 106-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 106 to 105 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 106 units passed, and the 106-unit model achieved 86.748% accuracy, leaving a 1.748-point margin that motivates testing the next lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3853531387300564613, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 40893, "peak_hidden_elements": 108032, "recurrent_macs": 738084375, "recurrent_steps": 18745, "total_inference_macs": 738768975, "training_seconds": 78.93231716682203, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.3742036784353432}

RECENT RESULT
hypothesis: A 104-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 105-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 105 to 104 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 105 units passed, and the 105-unit model achieved 86.748% accuracy, leaving a 1.748-point margin that motivates testing the next lower-cost width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3786324735447265977, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 725206560, "recurrent_steps": 18745, "total_inference_macs": 725884640, "training_seconds": 55.493518540868536, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.3855093271454419}

RECENT RESULT
hypothesis: A 103-unit, 23-step GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 104-unit model.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 103 while preserving the verified schedule and training procedure.
mechanism: One-channel recurrent-width trim under batch-64 training
evidence_used: Every adjacent batch-64 width reduction from 117 through 104 units passed; the 104-unit model achieved 85.521% accuracy, making the next one-channel trim the most direct lower-cost boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3719704744227515017, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 786, "p95_recurrent_steps": 23, "parameters": 39497, "peak_hidden_elements": 105984, "recurrent_macs": 712441215, "recurrent_steps": 18745, "total_inference_macs": 713112775, "training_seconds": 64.17969787493348, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.3853984996585027}



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
