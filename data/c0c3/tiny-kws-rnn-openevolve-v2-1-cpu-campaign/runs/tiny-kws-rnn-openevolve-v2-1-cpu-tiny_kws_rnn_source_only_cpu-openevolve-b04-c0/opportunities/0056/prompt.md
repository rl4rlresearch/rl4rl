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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1764539150337571809, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20584, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338283680, "training_seconds": 35.3120228338521, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.438861083984375}
prior_hypothesis: A 68-unit GRU using the qualifying 24-frame schedule with only the earliest boundary frame removed will reach at least 85% accuracy in 23 steps and use fewer MACs than the verified 67-unit, 24-step model.

## Recent verification evidence

RECENT RESULT
hypothesis: Projecting each 20-band frame to 16 learned features while retaining the verified 65-unit, 26-step GRU will preserve at least 85% validation accuracy and reduce total dense inference MACs by approximately 2.8%.
change: Add a 20-to-16 linear input projection and change the GRU input width from 20 to 16, preserving recurrent capacity, temporal summaries, schedule, and training procedure.
mechanism: Learned spectral bottleneck before recurrence
evidence_used: The 65-unit, 26-step deviation-readout model achieved 86.13% accuracy, whereas reducing recurrent width to 64 at 26 steps failed with 84.05%; this motivates preserving the successful hidden width while reducing the more redundant spectral input dimension.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1790037632321339889, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 18649, "peak_hidden_elements": 133632, "recurrent_macs": 341476850, "recurrent_steps": 21190, "total_inference_macs": 343172050, "training_seconds": 36.306638916023076, "validation_accuracy": 0.8196319018404908, "validation_cross_entropy": 0.4820497290488401}

RECENT RESULT
hypothesis: A 66-unit GRU with deviation readout processing 25 uniformly spaced frames will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 1.2% versus the verified 65-unit, 26-step model.
change: Increase recurrent and summary width from 65 to 66 units, resize the classifier input from 260 to 264 features, and reduce the uniform schedule from 26 to 25 steps.
mechanism: Width–time rebalancing with deviation-assisted readout
evidence_used: The 65-unit deviation-readout model achieved 86.13% at 26 steps, providing accuracy margin, while the same width–time exchange previously allowed 66 units at 27 steps to outperform the qualifying 65-unit, 28-step design in MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1818698980003451539, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 19584, "peak_hidden_elements": 135680, "recurrent_macs": 346945500, "recurrent_steps": 20375, "total_inference_macs": 348666780, "training_seconds": 42.63805266702548, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4410978539589724}

RECENT RESULT
hypothesis: A 67-unit GRU processing 24 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.4% versus the verified 66-unit, 25-step model.
change: Increase recurrent and summary width from 66 to 67 units, resize the classifier input from 264 to 268 features, and reduce the uniform schedule from 25 to 24 steps.
mechanism: Width–time rebalancing with deviation-assisted readout
evidence_used: Exchanging one step for one hidden unit previously produced qualifying 65-unit/26-step and 66-unit/25-step deviation-readout models; the latest achieved 85.64% accuracy, motivating the next incremental exchange.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1793277019134885521, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 20081, "peak_hidden_elements": 137728, "recurrent_macs": 342045720, "recurrent_steps": 19560, "total_inference_macs": 343793080, "training_seconds": 35.397358583053574, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4548308109213238}

RECENT RESULT
hypothesis: A 68-unit GRU processing 23 uniformly spaced frames will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.6% versus the verified 67-unit, 24-step model.
change: Increase recurrent and summary width from 67 to 68 units, resize the four-statistic classifier input from 268 to 272 features, and reduce the uniform schedule from 24 to 23 steps.
mechanism: Deviation-assisted width–time rebalancing
evidence_used: Successive deviation-readout exchanges qualified at 65 units/26 steps, 66/25, and 67/24; the latest reached 85.03%, motivating one more incremental exchange while preserving a net MAC reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1764539150337571809, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20584, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338283680, "training_seconds": 37.355420750100166, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.4483058250754889}

RECENT RESULT
hypothesis: Adding a zero-MAC temporal trend statistic to the 68-unit, 23-step GRU will raise validation accuracy from 83.93% to at least 85% while keeping total dense inference MACs below the verified 67-unit, 24-step model.
change: Use 68 hidden units over 23 uniformly spaced frames and augment the classifier with a fifth summary encoding the normalized linear temporal trend of each recurrent unit.
mechanism: Temporal-trend-augmented recurrent readout
evidence_used: The 68-unit, 23-step model reduced MACs to 338,283,680 but missed the threshold by 1.07 points; its 0.4483 cross-entropy was better than the qualifying 67-unit, 24-step model’s 0.4548, suggesting a compact timing-sensitive readout may correct decisions without restoring a recurrent step.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1766851783559673313, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 21128, "peak_hidden_elements": 174592, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338727040, "training_seconds": 41.42003950010985, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.44180428908646474}

RECENT RESULT
hypothesis: Four contiguous temporal-bin means will capture nonlinear phoneme timing omitted by the trend statistic, raising the 68-unit, 23-step GRU from 84.29% to at least 85% while remaining below the verified 67-unit, 24-step model’s dense MAC cost.
change: Use 68 hidden units over 23 frames and replace the global mean readout with four causal temporal-bin means, retaining final, maximum, and deviation summaries.
mechanism: Coarse temporal-pyramid recurrent readout
evidence_used: The 68-unit, 23-step model had lower cross-entropy than the qualifying 67-unit, 24-step model, and adding a linear trend improved both accuracy and cross-entropy but remained short of 85%; a four-bin summary provides richer timing information while preserving the one-step MAC saving.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1771477050003876321, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 22216, "peak_hidden_elements": 281088, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 339613760, "training_seconds": 50.24004374979995, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.44104288487346627}

RECENT RESULT
hypothesis: A 68-unit, 23-step GRU augmented with four temporal-bin deviations will achieve at least 85% validation accuracy while using fewer dense inference MACs than the verified 67-unit, 24-step model.
change: Use 23 frames and summarize recurrent outputs with final, maximum, global deviation, four temporal-bin means, and four temporal-bin deviations.
mechanism: Temporal-bin deviation readout
evidence_used: The 68-unit, 23-step four-bin-mean model reached 84.66% accuracy at 339,613,760 MACs; adding within-bin deviations supplies localized activation-duration information while keeping the estimated total near 341.4M MACs, below the qualifying 343,793,080-MAC design.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Four temporal-bin means and deviations will raise the 68-unit, 23-step GRU from 84.66% to at least 85% accuracy while keeping total inference below 343,793,080 MACs.
change: Use a 68-unit GRU over 23 frames and accumulate causal first and second moments in four contiguous temporal bins for an 11-summary classifier.
mechanism: Localized temporal-deviation readout
evidence_used: The 68-unit, 23-step four-bin-mean model reached 84.66%; localized deviations add complementary duration information with an estimated total cost of 341,387,200 MACs.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Four temporal-bin means and deviations will raise the 68-unit, 23-step GRU from 84.66% to at least 85% accuracy while keeping total inference below the verified 343,793,080-MAC model.
change: Use a 68-unit GRU over 23 frames and accumulate first and second moments in four temporal bins, classifying from final, maximum, global deviation, bin-mean, and bin-deviation summaries.
mechanism: Localized temporal-deviation readout
evidence_used: The 68-unit, 23-step four-bin-mean model reached 84.66% at 339,613,760 MACs; four localized deviation summaries add activation-duration information while keeping estimated cost near 341,387,200 MACs.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding four temporal-bin deviation summaries to the 68-unit, 23-step GRU will raise validation accuracy from 84.66% to at least 85% while keeping total inference below 343,793,080 MACs.
change: Use a 68-unit GRU over 23 frames, accumulate first and second moments in four contiguous temporal bins, and classify from final, maximum, global deviation, bin-mean, and bin-deviation summaries.
mechanism: Localized temporal-deviation readout
evidence_used: The verified 68-unit, 23-step four-bin-mean model reached 84.66% at 339,613,760 MACs; localized deviations add complementary activation-duration information with an estimated total of 341,387,200 MACs.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 68-unit GRU using the qualifying 24-frame schedule with only the earliest boundary frame removed will reach at least 85% accuracy in 23 steps and use fewer MACs than the verified 67-unit, 24-step model.
change: Increase hidden width from 67 to 68 and replace uniform 23-frame resampling with the verified 24-frame sampling pattern minus frame zero.
mechanism: Boundary-aware temporal subsampling
evidence_used: Uniform 68-unit/23-step sampling achieved 83.93% accuracy but better cross-entropy than the qualifying 67-unit/24-step model; preserving 23 of that qualifying model’s exact sample locations tests whether schedule displacement, rather than insufficient capacity, caused the regression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1764539150337571809, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20584, "peak_hidden_elements": 139776, "recurrent_macs": 336510240, "recurrent_steps": 18745, "total_inference_macs": 338283680, "training_seconds": 35.3120228338521, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.438861083984375}

RECENT RESULT
hypothesis: Reducing the qualifying 68-unit, 23-step GRU to 67 units will retain at least 85% validation accuracy while lowering total dense inference MACs by approximately 2.6%.
change: Keep the successful 23-frame boundary-aware schedule and four-statistic readout, but reduce recurrent width from 68 to 67 and resize the classifier and states accordingly.
mechanism: Redundant-width pruning on the boundary-aware schedule
evidence_used: The 67-unit GRU already achieved 85.03% accuracy with the full qualifying 24-frame schedule, while removing its earliest boundary frame and using 68 units improved accuracy to 85.52%; this tests whether that added unit was unnecessary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 1718936913989268001, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 397, "p95_recurrent_steps": 23, "parameters": 20081, "peak_hidden_elements": 137728, "recurrent_macs": 327793815, "recurrent_steps": 18745, "total_inference_macs": 329541175, "training_seconds": 35.89221137482673, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.4525481616061158}



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
