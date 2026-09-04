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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4005374463779140533, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 764651745, "recurrent_steps": 22005, "total_inference_macs": 767879145, "training_seconds": 188.3290833751671, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.46505390003414976}
prior_hypothesis: Beginning conservative classifier-bound checks at recurrent step 23 will retain validation accuracy at or above 85% while reducing total inference MACs below 768,548,682 and mean recurrent steps below 27.0454.

## Recent verification evidence

RECENT RESULT
hypothesis: Restricting classification to 98 of the 99 recurrent channels will retain at least 85% validation accuracy while reducing classifier inference by 6,520 MACs; the unobserved channel remains available as recurrent memory that can influence later observed outputs.
change: Preserve the verified 18-feature, 99-unit, 30-step GRU, but reduce the classifier input from 99 to 98 channels and exclude one summary channel from the final readout.
mechanism: Single-channel latent recurrent memory
evidence_used: Grouping the readout down to 33 features failed at 82.45%, while reducing the recurrent state itself to 98 units also failed; this conservative change preserves all 99 recurrent units and removes only one of 99 direct classifier inputs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4435043861801154235, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36175, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850252010, "training_seconds": 78.52013216703199, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.46358979582055215}

RECENT RESULT
hypothesis: Factoring the 99-to-8 classifier through seven learned logits will retain at least 85% validation accuracy while saving 43 MACs and 43 parameters per classifier, because eight-class softmax needs only seven independent logit-difference dimensions.
change: Replace the direct classifier with a bias-free 99-to-7 projection followed by a learned 7-to-8 decoder, preserving access to all 99 recurrent channels and trainable logits for every class.
mechanism: Symmetric rank-seven classifier factorization
evidence_used: The fixed-reference seven-logit classifier reached only 83.56%, showing that asymmetric treatment of one class harms learning; the proposed symmetric decoder retains the same structural dimensionality reduction while avoiding a fixed class, and unlike the failed 98-channel readout it discards no recurrent-state information.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4434895071060761675, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 30.0, "median_recurrent_steps": 30, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36140, "peak_hidden_elements": 101888, "recurrent_macs": 849613050, "recurrent_steps": 24450, "total_inference_macs": 850223485, "training_seconds": 77.3180928749498, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.4670647088735381}

RECENT RESULT
hypothesis: A classifier-weight bound at step 29 will preserve every exited example’s final predicted class, maintaining at least 85% accuracy while reducing mean recurrent steps and exact MACs.
change: Defer the learned readout until the penultimate step, then exit examples whose class cannot change under any possible final bounded GRU output.
mechanism: Penultimate-step certified early exit
evidence_used: The 30-step model meets the target at 85.03%, while fixed 29-step schedules narrowly fail; the prior certified-exit attempt timed out without accuracy evidence, motivating this cheaper single-check variant that retains the final step only for uncertified examples.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4308117546380092537, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 29.137423312883435, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 825184503, "recurrent_steps": 23747, "total_inference_macs": 825918687, "training_seconds": 149.85506487521343, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4649688159029908}

RECENT RESULT
hypothesis: Extending exact classifier-bound checks from step 29 to step 28 will preserve validation accuracy at or above 85% while reducing total inference MACs below 825,918,687 by allowing confident examples to skip both remaining recurrent steps.
change: Run the learned classifier beginning at recurrent step 28 and apply the existing mathematically conservative bounded-output certificate at both steps 28 and 29.
mechanism: Two-step certified early exit
evidence_used: The step-29 certificate preserved 85.28% accuracy while 86.3% of examples exited early; the broader step-20 attempt provided no contrary accuracy evidence because training timed out. Adding only the immediately preceding check is a lower-overhead probe whose certified exits cannot change the final predicted class.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4203589080597487453, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 28.42331288343558, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 804960585, "recurrent_steps": 23165, "total_inference_macs": 805879305, "training_seconds": 196.69990370911546, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46787701004121934}

RECENT RESULT
hypothesis: Beginning the exact classifier-bound checks at recurrent step 27 will preserve validation accuracy at or above 85% while reducing total inference MACs and mean recurrent steps below the verified step-28 design.
change: Enable the learned classifier and conservative bounded-output exit certificate during the final three recurrent steps instead of the final two.
mechanism: Three-step certified early exit
evidence_used: Step-28 certified exit achieved 85.03% accuracy and reduced mean execution to 28.42 steps; extending the same mathematically conservative certificate by one step is the smallest next cost-reduction probe, while the much earlier step-20 attempt timed out without contrary accuracy evidence.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4120565537445158201, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.851533742331288, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 788767551, "recurrent_steps": 22699, "total_inference_macs": 789962679, "training_seconds": 178.55719587486237, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4701891191166603}

RECENT RESULT
hypothesis: Beginning the exact classifier-bound checks at recurrent step 26 will preserve validation accuracy at or above 85% while reducing total inference MACs below 789,962,679 and mean recurrent steps below 27.8515.
change: Enable the learned classifier and conservative bounded-output exit certificate during the final four recurrent steps instead of the final three.
mechanism: Four-step certified early exit
evidence_used: Extending certified checks from step 28 to step 27 preserved 85.03% accuracy while reducing total MACs from 805,879,305 to 789,962,679; moving the same mathematically conservative certificate back one additional step is the smallest supported cost-reduction probe.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4061271564091754005, "maximum_recurrent_steps": 30, "mean_recurrent_steps": 27.43680981595092, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 30, "parameters": 36183, "peak_hidden_elements": 101888, "recurrent_macs": 777022389, "recurrent_steps": 22361, "total_inference_macs": 778595301, "training_seconds": 171.58000745810568, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4703606073110381}

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
