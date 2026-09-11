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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3017510226754546524, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 26984, "peak_hidden_elements": 86528, "recurrent_macs": 577528560, "recurrent_steps": 22820, "total_inference_macs": 578493520, "training_seconds": 48.24753158399835, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.44222420534473256}
prior_hypothesis: A 74-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will achieve at least 85% validation accuracy while reducing predicted total inference MACs from 591,445,500 to approximately 578,493,520.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3157424536528656486, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 604169280, "recurrent_steps": 21190, "total_inference_macs": 605316800, "training_seconds": 70.86741162510589, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44337790085494155}
prior_hypothesis: An 88-unit dual-readout GRU using 26 evenly thinned frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 628,554,080 to approximately 605,316,800.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3085069725220775893, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27573, "peak_hidden_elements": 87552, "recurrent_macs": 590467500, "recurrent_steps": 22820, "total_inference_macs": 591445500, "training_seconds": 46.008248457917944, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4075592041015625}
prior_hypothesis: A 75-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will retain at least 85% accuracy while reducing total inference MACs from 604,534,400 to approximately 591,445,500.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3153343419240889388, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28168, "peak_hidden_elements": 88576, "recurrent_macs": 603543360, "recurrent_steps": 22820, "total_inference_macs": 604534400, "training_seconds": 48.35405945801176, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4143990545916411}
prior_hypothesis: Explicit frame-to-frame spectral motion will let a 76-unit GRU over 28 frames achieve at least 85% accuracy with approximately 604,534,400 total inference MACs, below the qualified 26-step model’s 605,316,800.

## Recent verification evidence

RECENT RESULT
hypothesis: An 89-unit dual-readout GRU processing the most recent 30 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 90-unit model’s 727,338,600.
change: Reduce the GRU hidden width from 96 to 89 and resize its classifier, hidden state, and temporal summary while preserving the qualified 30-frame schedule and training procedure.
mechanism: Single-unit recurrent width refinement
evidence_used: The 90-unit, 30-frame dual-readout GRU achieved 86.01% accuracy; its 1.01-point margin motivates a conservative one-unit reduction after the successful sequence of width refinements.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3717702446548730069, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 31109, "peak_hidden_elements": 91648, "recurrent_macs": 711568350, "recurrent_steps": 24450, "total_inference_macs": 712728910, "training_seconds": 95.21129420911893, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4104718729030867}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU over the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 712,728,910 to 698,265,920.
change: Reduce the GRU hidden width from 89 to 88 and resize its state, temporal summary, and mean-plus-final classifier.
mechanism: Single-unit recurrent width refinement
evidence_used: The 89-unit model achieved 86.26% accuracy, while every tested dense width from 90 through 96 also qualified; this 1.26-point margin supports the next conservative one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3642261290005708066, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 697118400, "recurrent_steps": 24450, "total_inference_macs": 698265920, "training_seconds": 84.52542316704057, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4217787853779237}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU processing the most recent 29 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 30-frame model’s 698,265,920.
change: Reduce hidden width from 90 to 88, resize the classifier and recurrent state, and omit the first three input frames.
mechanism: Incremental temporal trimming of a qualified narrow GRU
evidence_used: The 88-unit, 30-frame model achieved 86.38% accuracy, while trimming the 98-unit model from 30 to 29 frames reduced accuracy by only 0.74 points and still qualified; applying the same single-frame trim to the narrower qualified design tests a larger structural cost reduction with 1.38 points of available margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3521052101636445171, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 673881120, "recurrent_steps": 23635, "total_inference_macs": 675028640, "training_seconds": 74.68331041699275, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.42446389227557035}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU processing the most recent 28 frames will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 88-unit, 29-frame model’s 675,028,640.
change: Reduce the GRU width from 94 to 88, resize its state and classifier, and omit the first four input frames.
mechanism: Incremental temporal trimming of the qualified narrow GRU
evidence_used: The 88-unit model qualified at 86.38% over 30 frames and 86.50% over 29 frames; the latter leaves 1.50 percentage points of margin and motivates testing the next single-frame structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3399842913267182276, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 650643840, "recurrent_steps": 22820, "total_inference_macs": 651791360, "training_seconds": 66.04627275001258, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.43144288794394653}

RECENT RESULT
hypothesis: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy on the most recent 27 frames while reducing total inference MACs below 651,791,360 and recurrent steps from 28 to 27 per example.
change: Omit one additional leading input frame, changing the causal schedule from the most recent 28 frames to the most recent 27 while preserving the model and training procedure.
mechanism: Incremental trailing-window temporal trimming
evidence_used: The same 88-unit model qualified at 86.50% with 29 frames and 85.89% with 28 frames; the current 0.89-point margin supports testing the next single-frame structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3278633724897919381, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 627406560, "recurrent_steps": 22005, "total_inference_macs": 628554080, "training_seconds": 64.4013512081001, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45655335970451494}

RECENT RESULT
hypothesis: An 86-unit dual-readout GRU over the most recent 28 frames will achieve at least 85% validation accuracy while reducing predicted total inference MACs below the qualified 88-unit, 27-frame model’s 628,554,080 to approximately 625,202,800.
change: Reduce GRU width from 88 to 86, resize its state and classifier, and expand the trailing window from 27 qualified frames to 28 frames.
mechanism: Width–time exchange with dual-timescale readout
evidence_used: The 88-unit model achieved 85.89% over 28 frames versus 85.03% over 27; additionally, exchanging two units for one frame previously improved accuracy when moving from the 98-unit, 29-frame model to the 96-unit, 30-frame model.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3261152938565302908, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 624081360, "recurrent_steps": 22820, "total_inference_macs": 625202800, "training_seconds": 59.73556770803407, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.45403945080341734}

RECENT RESULT
hypothesis: An 86-unit GRU over 28 frames with mean–final multiplicative features will achieve at least 85% validation accuracy while keeping total inference MACs below the qualified 88-unit, 27-frame model.
change: Use the near-qualified 86-unit, 28-frame architecture and augment its linear classifier with the elementwise product of mean and final recurrent outputs.
mechanism: Parameter-free cross-timescale interaction readout
evidence_used: The 86-unit, 28-frame model missed qualification by one validation example while achieving lower cross-entropy than the qualified 88-unit, 27-frame model; adding an inexpensive interaction feature targets this narrow accuracy gap with a predicted 625,763,520 total MACs, still below 628,554,080.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3264077739405019516, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29976, "peak_hidden_elements": 88576, "recurrent_macs": 624081360, "recurrent_steps": 22820, "total_inference_macs": 625763520, "training_seconds": 52.25696949986741, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.46472425402307804}

RECENT RESULT
hypothesis: An 88-unit dual-readout GRU using 26 evenly thinned frames across the latest 28-frame window will retain at least 85% validation accuracy while reducing total inference MACs from 628,554,080 to approximately 605,316,800.
change: Replace the 128-unit baseline with the qualified 88-unit mean-plus-final GRU and process 26 samples spanning the latest 28 frames instead of truncating the window to 26 contiguous frames.
mechanism: Span-preserving temporal thinning
evidence_used: The 88-unit model achieved 85.89% on 28 frames and 85.03% on 27, whereas reducing width to 86 missed qualification; retaining width and removing two distributed, temporally redundant frames tests a lower-cost alternative while preserving the stronger 28-frame temporal span.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3157424536528656486, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 604169280, "recurrent_steps": 21190, "total_inference_macs": 605316800, "training_seconds": 70.86741162510589, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44337790085494155}

RECENT RESULT
hypothesis: The qualified 88-unit dual-readout GRU will retain at least 85% validation accuracy using 25 evenly distributed frames across the latest 28-frame window, reducing predicted total inference MACs from 605,316,800 to approximately 582,079,520 and recurrent steps from 26 to 25 per example.
change: Reduce the frame-schedule target from 26 to 25 while preserving the qualified GRU, training procedure, and 28-frame temporal span.
mechanism: Incremental span-preserving temporal thinning
evidence_used: The 26-frame span-preserving schedule qualified at 85.03% and achieved lower cross-entropy than the qualified 27-frame contiguous schedule; this indicates that distributed sampling preserves useful temporal coverage and motivates testing the next single-step cost boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3036215348159393591, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 580932000, "recurrent_steps": 20375, "total_inference_macs": 582079520, "training_seconds": 70.22041741595604, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4608848805807851}

RECENT RESULT
hypothesis: Explicit frame-to-frame spectral motion will let a 76-unit GRU over 28 frames achieve at least 85% accuracy with approximately 604,534,400 total inference MACs, below the qualified 26-step model’s 605,316,800.
change: Replace the absolute-frame-only 88-unit GRU with a 76-unit GRU receiving each normalized frame and its causal first difference; retain mean-plus-final prediction and restore the qualified 28-frame window.
mechanism: Causal delta-augmented compact GRU
evidence_used: The 88-unit 28-frame model reached 85.89%, while the 86-unit version missed by only one example. This tests the load-bearing assumption that dense recurrent width must discover local acoustic motion internally, reallocating capacity from hidden mixing to explicit parameter-free delta features.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3153343419240889388, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 28168, "peak_hidden_elements": 88576, "recurrent_macs": 603543360, "recurrent_steps": 22820, "total_inference_macs": 604534400, "training_seconds": 48.35405945801176, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4143990545916411}

RECENT RESULT
hypothesis: A 75-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will retain at least 85% accuracy while reducing total inference MACs from 604,534,400 to approximately 591,445,500.
change: Replace the 88-unit absolute-frame GRU with a 75-unit delta-augmented GRU, resize its state and dual readout, and use the most recent 28 frames.
mechanism: Single-unit refinement of delta-augmented recurrence
evidence_used: The 76-unit delta-augmented model achieved 85.52% accuracy and 0.4144 cross-entropy at 604,534,400 MACs; its qualification and stronger cross-entropy motivate a conservative one-unit width reduction while preserving its proven temporal span and features.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3085069725220775893, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27573, "peak_hidden_elements": 87552, "recurrent_macs": 590467500, "recurrent_steps": 22820, "total_inference_macs": 591445500, "training_seconds": 46.008248457917944, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4075592041015625}

RECENT RESULT
hypothesis: A 74-unit GRU receiving normalized frames and causal spectral deltas over 28 steps will achieve at least 85% validation accuracy while reducing predicted total inference MACs from 591,445,500 to approximately 578,493,520.
change: Replace the current 88-unit absolute-frame GRU with a 74-unit delta-augmented GRU, resize its recurrent state and dual readout, and process the most recent 28 frames.
mechanism: Single-unit refinement of delta-augmented recurrence
evidence_used: The 75-unit delta-augmented model achieved 86.13% accuracy and 0.4076 cross-entropy at 591,445,500 MACs, improving upon the qualified 76-unit version; its 1.13-point margin motivates the next conservative one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3017510226754546524, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 26984, "peak_hidden_elements": 86528, "recurrent_macs": 577528560, "recurrent_steps": 22820, "total_inference_macs": 578493520, "training_seconds": 48.24753158399835, "validation_accuracy": 0.8601226993865031, "validation_cross_entropy": 0.44222420534473256}



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
