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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4003175120164524319, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.9840490797546, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764200008, "recurrent_steps": 21992, "total_inference_macs": 767457504, "training_seconds": 159.3860187078826, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4650421891475748}
prior_hypothesis: Restricting the step-22 classifier check to the highest-energy sixteenth will retain at least 13 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,462,355.

## Recent verification evidence

RECENT RESULT
hypothesis: Supervising classifier predictions at recurrent steps 26–29 will preserve final validation accuracy at or above 85% while enabling certified checks from step 26 to reduce MACs below the verified step-27 design.
change: Cache training-only logits from the four late prefixes, add a lightly weighted auxiliary cross-entropy loss, and begin learned classification and certified exit checks at step 26.
mechanism: Late-prefix deep supervision with four-step certified exit
evidence_used: Step-26 certified exit missed the threshold by one validation example at 84.91% while lowering total MACs to 778,595,301; step-27 met the target at 85.03%, so targeted late-prefix supervision is a minimal accuracy-oriented change to recover that narrow deficit.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Supervising the four summaries immediately preceding the final state will recover the one-example accuracy deficit of step-26 certified exit, achieving at least 85% accuracy while reducing MACs below the verified step-27 design.
change: Cache steps 26–29 during the standard training sequence, apply a lightly weighted auxiliary classification loss to them, and enable certified exits beginning at step 26.
mechanism: Verifiable late-prefix deep supervision
evidence_used: Step-26 certified exit reached 84.91%, only one validation example below the requirement, while reducing total inference MACs to 778,595,301; targeted prefix supervision directly trains the representations used by those earlier checks without changing inference architecture or parameter count.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing label smoothing from 0.03 to 0.02 will recover the one-example accuracy deficit of step-26 certified exit while increasing late-prefix margins, achieving at least 85% accuracy and fewer than 789,962,679 total inference MACs.
change: Begin learned classification and conservative exit certification at recurrent step 26, and slightly sharpen the training targets without changing model size or inference architecture.
mechanism: Margin-sharpened four-step certified early exit
evidence_used: Step-26 certified exit reduced total MACs to 778,595,301 but missed the accuracy threshold by only one validation example at 84.91%; step 27 reached 85.03%, making a small training-only margin adjustment the most targeted verifiable follow-up after the auxiliary-loss implementations could not be verified.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4059973853243375291, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.428220858895706, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 776779146, "recurrent_steps": 22354, "total_inference_macs": 778346514, "training_seconds": 160.25353908305988, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4649836721595811}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 25 will retain validation accuracy at or above 85% while reducing total inference MACs below 778,346,514 and mean recurrent steps below 27.4283.
change: Enable learned classification and the existing bounded-output exit certificate one recurrent step earlier, during the final five steps instead of the final four.
mechanism: Five-step certified early exit
evidence_used: Extending certified checks from step 27 to step 26 reduced total MACs from 789,962,679 to 778,346,514, and label smoothing 0.02 restored accuracy to 85.15%; moving the unchanged certificate back one more step is the smallest supported cost-reduction test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4024965611481058457, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.174233128834356, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 769586103, "recurrent_steps": 22147, "total_inference_macs": 771635007, "training_seconds": 147.53195574996062, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4650554165518357}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 24 will retain validation accuracy at or above 85% while reducing total inference MACs below 771,635,007 and mean recurrent steps below 27.1743.
change: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final six steps instead of the final five.
mechanism: Six-step certified early exit
evidence_used: Moving the certificate from step 26 to step 25 with label smoothing 0.02 preserved 85.15% accuracy and reduced total MACs from 778,346,514 to 771,635,007; extending the unchanged certificate by one step is the smallest supported cost-reduction probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4008866870652260027, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.045398773006134, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 765937458, "recurrent_steps": 22042, "total_inference_macs": 768548682, "training_seconds": 154.15326458308846, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4650977479899588}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 23 will retain validation accuracy at or above 85% while reducing total inference MACs below 768,548,682 and mean recurrent steps below 27.0454.
change: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final seven steps instead of the final six.
mechanism: Seven-step certified early exit
evidence_used: Moving the certificate successively from step 26 through step 24 preserved 85.15% accuracy while reducing total MACs at every extension; step 24 reached 768,548,682 MACs and 27.0454 mean steps, motivating the smallest supported next probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4005374463779140533, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764651745, "recurrent_steps": 22005, "total_inference_macs": 767879145, "training_seconds": 188.3290833751671, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46505390003414976}

RECENT RESULT
hypothesis: Beginning conservative classifier-bound checks at recurrent step 22 will retain validation accuracy at or above 85% while reducing total inference MACs below 767,879,145 and mean recurrent steps below 27.0.
change: Enable the learned classifier and existing bounded-output exit certificate one recurrent step earlier, during the final eight steps instead of the final seven.
mechanism: Eight-step certified early exit
evidence_used: Extending the unchanged certificate successively through steps 26, 25, 24, and 23 preserved 85.15% accuracy while consistently reducing inference cost; step 23 achieved 767,879,145 MACs and exactly 27.0 mean steps.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4006145963979265385, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98282208588957, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764165259, "recurrent_steps": 21991, "total_inference_macs": 768027051, "training_seconds": 193.41725833294913, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46504105643992044}

RECENT RESULT
hypothesis: Evaluating the full classifier only for the highest-energy quartile at step 22 will retain at least five of the 14 previously observed step-22 exits, preserving accuracy at or above 85% while reducing total inference MACs below 767,879,145.
change: Add a zero-MAC recurrent-activation energy screen at step 22, run the learned classifier only on the selected quartile, and retain the existing full certified checks from step 23 onward.
mechanism: Activation-energy-screened certified early exit
evidence_used: Unscreened step-22 checks saved only 14 recurrent steps and increased total MACs because all examples incurred an extra 99-to-8 classification; screening to roughly one quarter of examples breaks even if it retains five certified exits, while the conservative certificate protects predictions.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4003621805354731553, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98282208588957, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764165259, "recurrent_steps": 21991, "total_inference_macs": 767543139, "training_seconds": 184.6520685828291, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46504103771747984}

RECENT RESULT
hypothesis: Screening the highest-energy quartile at step 21 will produce at least five additional certified step savings, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,543,139.
change: Start the existing energy-screened classifier and conservative exit certificate at step 21 instead of step 22, while retaining all later checks unchanged.
mechanism: Energy-screened ninth-step certified exit
evidence_used: Moving the certificate earlier produced diminishing marginal savings of 105, 37, and 14 recurrent steps at steps 24, 23, and 22; the step-22 energy screen retained all 14 savings while evaluating only one quarter of examples, and five step-21 savings are sufficient to offset the added classifier calls.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4004283312555537411, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98159509202454, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764130510, "recurrent_steps": 21990, "total_inference_macs": 767669958, "training_seconds": 151.47394229192287, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.465038238712615}

RECENT RESULT
hypothesis: Restricting the step-22 classifier check to the highest-energy eighth will retain at least 12 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,543,139.
change: Reduce the zero-MAC step-22 candidate screen from one quarter to one eighth of each active batch while leaving all certified checks from step 23 onward unchanged.
mechanism: Tighter activation-energy screening
evidence_used: The highest-energy quartile retained all 14 exits produced by the unscreened step-22 check and reduced total MACs to 767,543,139; halving its classifier calls remains a strict improvement even if up to two of those exits are deferred.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4003200423718557329, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.98282208588957, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764165259, "recurrent_steps": 21991, "total_inference_macs": 767462355, "training_seconds": 159.27410729206167, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.465041271747987}

RECENT RESULT
hypothesis: Restricting the step-22 classifier check to the highest-energy sixteenth will retain at least 13 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,462,355.
change: Halve the zero-MAC step-22 candidate pool from one eighth to one sixteenth of each active batch while leaving certified checks from step 23 onward unchanged.
mechanism: Sixteenth-batch activation-energy screening
evidence_used: Highest-energy eighth screening retained all 14 exits from the unscreened step-22 check and reduced total MACs to 767,462,355; halving classifier calls remains a strict improvement if at most one exit is deferred.
result: the patch search text matched more than once

RECENT RESULT
hypothesis: Restricting the step-22 classifier check to the highest-energy sixteenth will retain at least 13 of the 14 observed early exits, preserve validation accuracy at or above 85%, and reduce total inference MACs below 767,462,355.
change: Halve the zero-MAC step-22 candidate pool from one eighth to one sixteenth while leaving all certified checks from step 23 onward unchanged.
mechanism: Sixteenth-batch activation-energy screening
evidence_used: Highest-energy eighth screening retained all 14 exits from the unscreened step-22 check and achieved 767,462,355 MACs; the previous sixteenth-screening proposal failed only because its search text was ambiguous, not because the design was disproven.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4003175120164524319, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 26.9840490797546, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764200008, "recurrent_steps": 21992, "total_inference_macs": 767457504, "training_seconds": 159.3860187078826, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4650421891475748}



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
