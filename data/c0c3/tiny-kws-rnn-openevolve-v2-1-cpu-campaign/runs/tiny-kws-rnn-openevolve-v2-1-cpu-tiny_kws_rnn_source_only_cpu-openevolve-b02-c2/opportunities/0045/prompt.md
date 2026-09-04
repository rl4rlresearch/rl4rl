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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3035331106289560162, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 580814640, "recurrent_steps": 22820, "total_inference_macs": 581910000, "training_seconds": 40.22126704105176, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4087623455773102}
prior_hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 29 to 28 causal steps, lowering total inference MACs from 602,653,380 to approximately 581,910,000.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3205666745972424105, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27850, "peak_hidden_elements": 86528, "recurrent_macs": 613470060, "recurrent_steps": 23635, "total_inference_macs": 614565420, "training_seconds": 46.13132729195058, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4174555959877061}
prior_hypothesis: An 84-unit GRU using the lowest 19 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 620,521,440 to approximately 614,565,420.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3143531732784505157, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 601558020, "recurrent_steps": 23635, "total_inference_macs": 602653380, "training_seconds": 37.70312804193236, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4105138579760592}
prior_hypothesis: An 84-unit GRU using mel bands 1–17 over the qualified 29-frame schedule will recover at least 85% validation accuracy while matching the 17-band model’s approximately 602,653,380 total inference MACs.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3306704161026971853, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28693, "peak_hidden_elements": 87552, "recurrent_macs": 632827125, "recurrent_steps": 23635, "total_inference_macs": 633935525, "training_seconds": 58.149112458806485, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.4017870850358273}
prior_hypothesis: An 85-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 647,491,420 to approximately 633,935,525.

## Recent verification evidence

RECENT RESULT
hypothesis: An 83-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 620,521,440 to approximately 607,249,165.
change: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 84 to 83 while preserving the qualified schedule and training procedure.
mechanism: Adjacent recurrent-width refinement at the qualified 29-step schedule
evidence_used: The 84-unit design achieved 85.64% validation accuracy with 620,521,440 MACs; its positive 0.64-point margin makes width 83 the closest unresolved structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3167504046643746721, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27521, "peak_hidden_elements": 85504, "recurrent_macs": 606166845, "recurrent_steps": 23635, "total_inference_macs": 607249165, "training_seconds": 52.139635833213106, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4415310350663823}

RECENT RESULT
hypothesis: A learned 20-to-16 input projection feeding the qualified 85-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 633,935,525 to approximately 617,391,025.
change: Preserve the 85-unit recurrent state, dual-view classifier, 29-frame schedule, and training procedure, but factor the GRU’s gate inputs through a shared 16-dimensional learned projection.
mechanism: Shared low-rank mel projection before gated recurrence
evidence_used: The 85-unit model achieved 86.99% accuracy, while reducing recurrent width to 83 fell below threshold; preserving width 85 while compressing the 20-band input targets input-side GRU MACs without sacrificing the recurrent capacity implicated by that boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3220405531599306669, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28009, "peak_hidden_elements": 87552, "recurrent_macs": 616282625, "recurrent_steps": 23635, "total_inference_macs": 617391025, "training_seconds": 52.27909362502396, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4564116401906394}

RECENT RESULT
hypothesis: An 84-unit GRU using the lowest 19 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 620,521,440 to approximately 614,565,420.
change: Preserve the qualified dual-view recurrent architecture and training procedure, reduce hidden width from 86 to 84, and structurally remove the highest-frequency mel band from every recurrent input.
mechanism: Edge-band structural pruning while preserving recurrent capacity
evidence_used: The 84-unit, 20-band model achieved 85.64% accuracy, whereas reducing recurrent width to 83 failed at 84.66%; preserving width 84 while pruning only one edge band targets input-side MACs without crossing the observed recurrent-capacity boundary. The failed 20-to-16 projection further favors a conservative one-band reduction over aggressive spectral compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3205666745972424105, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27850, "peak_hidden_elements": 86528, "recurrent_macs": 613470060, "recurrent_steps": 23635, "total_inference_macs": 614565420, "training_seconds": 46.13132729195058, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.4174555959877061}

RECENT RESULT
hypothesis: An 84-unit GRU using the lowest 18 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 614,565,420 to approximately 608,609,400.
change: Move to the qualified 84-unit architecture and structurally discard the two highest-frequency mel bands in both recurrent execution paths.
mechanism: Conservative high-frequency edge-band pruning
evidence_used: The 84-unit, 19-band model achieved 86.75% accuracy at 614,565,420 MACs, providing 1.75 percentage points of margin; removing one additional edge band preserves the demonstrated recurrent capacity while saving exactly 5,956,020 GRU MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3174599239378464631, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27596, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608609400, "training_seconds": 45.013332958100364, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4245996580533455}

RECENT RESULT
hypothesis: Averaging the mean recurrent output with the final hidden state before classification will retain at least 85% validation accuracy while halving classifier MACs, reducing total inference MACs from 608,609,400 to approximately 608,061,720.
change: Replace the 168-to-8 classifier over concatenated recurrent views with an 84-to-8 classifier over their elementwise average.
mechanism: Parameter-free fusion of temporal and final-state views
evidence_used: The current 84-unit, 18-band model already meets the threshold at 85.03%; unlike another spectral or recurrent-width reduction, this preserves both qualified recurrent representations and all 29 recurrent steps while removing only redundant classifier-side dimensionality.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3171742457162927479, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 26924, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608061720, "training_seconds": 43.2407687921077, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.45419027878462903}

RECENT RESULT
hypothesis: The qualified 84-unit, 18-band GRU with a linear 168→7→8 classifier will retain at least 85% validation accuracy while reducing total inference MACs from 608,609,400 to approximately 608,518,120.
change: Use the qualified lowest 18 mel bands and replace the 168→8 classifier with two linear layers having a seven-dimensional bottleneck and no intervening nonlinearity.
mechanism: Softmax-equivalent rank-7 classifier factorization
evidence_used: The 18-band trunk achieved 85.03% accuracy at 608,609,400 MACs. For eight-class softmax, seven independent logit contrasts suffice, so a rank-7 linear head preserves the necessary classifier output dimension while structurally saving 112 MACs per example; unlike the failed averaging head, it retains both recurrent views independently.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3174123109009208446, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27491, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608518120, "training_seconds": 47.02099200012162, "validation_accuracy": 0.8368098159509203, "validation_cross_entropy": 0.47702409241097105}

RECENT RESULT
hypothesis: The qualified 84-unit, 18-band GRU with seven learned class-contrast logits and one fixed reference logit will retain at least 85% accuracy while reducing total inference MACs from 608,609,400 to approximately 608,472,480.
change: Adopt the qualified 18-band recurrent trunk and replace the 168→8 classifier with one 168→7 linear layer whose outputs are concatenated with a zero reference-class logit.
mechanism: Single-layer reference-class logit parameterization
evidence_used: The 84-unit, 18-band trunk achieved 85.03% accuracy. The failed 168→7→8 head used a non-convex two-matrix factorization; directly learning seven logits preserves the full eight-class softmax function class without that factorization and saves 168 MACs per example.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3173885043824580342, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27427, "peak_hidden_elements": 86528, "recurrent_macs": 607514040, "recurrent_steps": 23635, "total_inference_macs": 608472480, "training_seconds": 42.87987033301033, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4565411245896041}

RECENT RESULT
hypothesis: An 84-unit GRU using the lowest 17 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 608,609,400 to approximately 602,653,380.
change: Adopt the qualified 84-unit architecture and structurally discard the three highest-frequency mel bands in both recurrent execution paths.
mechanism: Adjacent high-frequency edge-band pruning
evidence_used: The 84-unit, 19-band model achieved 86.75% accuracy and the 18-band model remained qualified at 85.03%; preserving recurrent width while removing the next edge band is the closest unresolved structural reduction and saves exactly 5,956,020 recurrent MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3143531732784505157, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 601558020, "recurrent_steps": 23635, "total_inference_macs": 602653380, "training_seconds": 37.495506417006254, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46444571091353526}

RECENT RESULT
hypothesis: An 84-unit GRU using mel bands 1–17 over the qualified 29-frame schedule will recover at least 85% validation accuracy while matching the 17-band model’s approximately 602,653,380 total inference MACs.
change: Reduce the recurrent input to 17 bands, but preserve band 17 by discarding the lowest-frequency band instead of the third-highest band.
mechanism: Opposite-edge spectral pruning
evidence_used: Bands 0–17 achieved 85.03%, while bands 0–16 narrowly missed at 84.79%; testing bands 1–17 distinguishes a dimensional-capacity limit from the loss of informative band 17.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3143531732784505157, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 601558020, "recurrent_steps": 23635, "total_inference_macs": 602653380, "training_seconds": 37.70312804193236, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4105138579760592}

RECENT RESULT
hypothesis: An 84-unit GRU using mel bands 2–17 over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 602,653,380 to approximately 596,697,360.
change: Structurally reduce the recurrent input from 19 to 16 bands, selecting bands 2–17 in both recurrent execution paths while preserving the qualified recurrent width, dual-view classifier, schedule, and training procedure.
mechanism: Opposite-edge spectral pruning preserving the informative upper band
evidence_used: The 17-band selection 1–17 achieved 86.26% accuracy, whereas 0–16 missed at 84.79%; this indicates retaining upper band 17 and pruning another low-frequency edge band is the best-supported next spectral reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3112464226190545683, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 27088, "peak_hidden_elements": 86528, "recurrent_macs": 595602000, "recurrent_steps": 23635, "total_inference_macs": 596697360, "training_seconds": 39.87663216609508, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.44875549129181846}

RECENT RESULT
hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 29 to 28 causal steps, lowering total inference MACs from 602,653,380 to approximately 581,910,000.
change: Adopt the qualified 17-band input selection and omit one additional trailing frame while preserving recurrent width, dual-view classification, and training procedure.
mechanism: Adjacent temporal edge pruning with the qualified spectral trunk
evidence_used: The 17-band bands-1–17 design achieved 86.26% accuracy at 602,653,380 MACs, while reducing to 16 bands failed; preserving its spectral and recurrent capacity while testing the nearest temporal reduction isolates a new cost axis with substantially larger potential savings.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3035331106289560162, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 580814640, "recurrent_steps": 22820, "total_inference_macs": 581910000, "training_seconds": 40.22126704105176, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.4087623455773102}

RECENT RESULT
hypothesis: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 28 to 27 causal steps, lowering total inference MACs from 581,910,000 to approximately 561,166,620.
change: Adopt the qualified 84-unit, 17-band architecture and omit one additional trailing frame while preserving its dual-view classifier and training procedure.
mechanism: Adjacent trailing-frame pruning on the qualified spectral trunk
evidence_used: The 28-step, 84-unit, bands-1–17 design achieved 86.87% validation accuracy at 581,910,000 MACs, leaving a 1.87-point margin; the closest unresolved reduction is one trailing recurrent step, saving approximately 20,743,380 MACs without reducing spectral or hidden capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2927130479794615167, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 27342, "peak_hidden_elements": 86528, "recurrent_macs": 560071260, "recurrent_steps": 22005, "total_inference_macs": 561166620, "training_seconds": 49.81973470794037, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4304316140391344}



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
