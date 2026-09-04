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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3002091254641370120, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 29830, "peak_hidden_elements": 91648, "recurrent_macs": 574636125, "recurrent_steps": 20375, "total_inference_macs": 575537515, "training_seconds": 43.83593229088001, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.38937989217372027}
prior_hypothesis: Processing frames 3–27 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and total inference MACs from 598,522,960 to approximately 575,537,515.

## Recent verification evidence

RECENT RESULT
hypothesis: Merging the two highest adjacent normalized mel bands into one feature will retain at least 85% accuracy while reducing total inference MACs from 671,608,085 to approximately 663,170,390.
change: Preserve the verified 60+59 hidden state and 29-frame schedule, but reduce each GRU’s input width from 20 to 19 by averaging the final two normalized mel bands.
mechanism: Adjacent high-mel band coalescing
evidence_used: The 20-to-16 projection reached 84.17%, while reducing recurrent width to 59+59 reached 84.29%; this suggests hidden capacity is more load-bearing and motivates a conservative one-feature input reduction that preserves information from every band.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3459197665273266165, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29740, "peak_hidden_elements": 122368, "recurrent_macs": 662394510, "recurrent_steps": 23635, "total_inference_macs": 663170390, "training_seconds": 49.59928729198873, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.4746463892650019}

RECENT RESULT
hypothesis: An orthonormal seven-coordinate parameterization of the eight centered logits will retain at least 85% accuracy while matching the prior seven-logit MAC and parameter reductions.
change: Replace the eight-output classifier with seven learned contrast coordinates and expand them through a fixed orthonormal basis into eight zero-sum logits.
mechanism: Orthonormal softmax gauge fixing
evidence_used: The fixed-zero seven-logit model reached 84.42% despite preserving softmax expressivity, suggesting its asymmetric parameterization hurt optimization; balanced orthonormal contrasts preserve the same full function class with better-conditioned, nonconstant logits for every class.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 54.18845779192634, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45900973454574867}

RECENT RESULT
hypothesis: A 61+60 GRU processing frames 4–31 with the verified orthonormal seven-coordinate classifier will recover the 0.46-point accuracy shortfall of the prior 28-step model, reaching at least 85% while reducing total inference MACs below the current 671,511,100.
change: Skip one additional leading frame, widen the parallel GRUs from 60+59 to 61+60 units, resize the recurrent state and summary to 121 features, and retain the seven-coordinate orthonormal classifier.
mechanism: Step-to-width reallocation with orthonormal logit contrasts
evidence_used: The prior 61+60, 28-step model reached 84.54%, narrowly missing the threshold, while the orthonormal seven-coordinate classifier subsequently produced 85.40% on the verified 29-step architecture and reduced classifier MACs; this directly tests whether its balanced parameterization can close the small 28-step accuracy gap.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482090183392159228, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30843, "peak_hidden_elements": 124416, "recurrent_macs": 666868860, "recurrent_steps": 22820, "total_inference_macs": 667559165, "training_seconds": 54.79871074995026, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.4892535437835506}

RECENT RESULT
hypothesis: A 60+58 GRU with the verified orthonormal seven-coordinate classifier will retain at least 85% validation accuracy while reducing total inference MACs from 671,511,100 to approximately 661,791,410.
change: Remove one hidden unit from the smaller GRU branch and resize its state, temporal summary, and classifier input from 119 to 118 features.
mechanism: Preserve-dominant-branch asymmetric width reduction
evidence_used: The 60+59 orthonormal model achieved 85.40% accuracy, whereas reducing the larger branch to form a symmetric 59+59 model failed at 84.29%; preserving the 60-unit branch tests whether the successful asymmetry is load-bearing while making the smallest remaining recurrent reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3452004695766290198, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29553, "peak_hidden_elements": 121344, "recurrent_macs": 661118220, "recurrent_steps": 23635, "total_inference_macs": 661791410, "training_seconds": 72.39296129206195, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.46837381000167755}

RECENT RESULT
hypothesis: Processing frames 3–30 will retain at least 85% validation accuracy while reducing recurrent execution from 29 to 28 steps and total inference MACs from 671,511,100 to approximately 648,378,955.
change: Preserve the verified 60+59 GRUs and orthonormal seven-coordinate classifier, but omit the final input frame.
mechanism: Opposite-boundary frame trimming
evidence_used: The current frames 3–31 design achieved 85.40% accuracy. A 28-step design that instead removed frame 3 failed, indicating that testing the opposite boundary is the cleanest way to distinguish temporal-position importance from a fundamental 29-step requirement.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3382043289552224552, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 647700060, "recurrent_steps": 22820, "total_inference_macs": 648378955, "training_seconds": 77.13366170902736, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4665845344403039}

RECENT RESULT
hypothesis: Replacing each three-gate GRU with a width-matched coupled forget cell will retain at least 85% validation accuracy while reducing recurrent MACs by exactly one third, from 670,832,205 to 447,221,470, with the same 29 recurrent steps.
change: Preserve the verified 60+59 hidden widths, causal mean summary, frame schedule, and orthonormal classifier, but replace each GRU with a two-output-per-hidden-unit `nn.Linear` recurrent cell.
mechanism: Coupled single-gate recurrent cells
evidence_used: The 100+100 ungated tanh RNN fell to 71.78% while the 60+59 GRU reached 85.40%, indicating gating is load-bearing. A coupled forget cell retains learned elementwise gating and recurrent dependence while removing one third of the GRU’s dense gate computation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2336316459170982440, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 20040, "peak_hidden_elements": 122368, "recurrent_macs": 447221470, "recurrent_steps": 23635, "total_inference_macs": 447900365, "training_seconds": 105.49562495900318, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46414298777199964}

RECENT RESULT
hypothesis: Retaining the dominant 60-unit GRU while replacing only the 59-unit branch with a coupled forget cell will preserve at least 85% validation accuracy and reduce total inference MACs from 671,511,100 to approximately 561,348,365.
change: Replace the smaller GRU with a two-output `nn.Linear` recurrent cell while preserving the 60-unit GRU, 29-frame schedule, temporal mean summary, classifier, and training procedure.
mechanism: Single-GRU anchor with coupled-gate auxiliary recurrence
evidence_used: Replacing both GRUs with coupled forget cells narrowly missed the target at 84.79% while cutting recurrent MACs by one third; retaining a full GRU in the larger branch tests whether one reset-gated recurrent path can recover that small accuracy deficit while still removing 110,162,735 recurrent MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2928078489532115480, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 25080, "peak_hidden_elements": 122368, "recurrent_macs": 560669470, "recurrent_steps": 23635, "total_inference_macs": 561348365, "training_seconds": 63.70274870796129, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.49833491974813077}

RECENT RESULT
hypothesis: A single cross-coupled 79-unit GRU receiving both normalized spectra and explicit frame-to-frame deltas will retain at least 85% accuracy while reducing total inference MACs from 671,511,100 to approximately 667,479,295.
change: Replace the two isolated GRU branches with one 79-unit GRU over 40-dimensional spectrum-plus-delta features, preserve all 29 frames, and classify from concatenated mean and final recurrent outputs.
mechanism: Delta-augmented unified GRU with dual-timescale readout
evidence_used: The 100+100 tanh RNN failed at 71.78%, showing learned gating is load-bearing, while reductions in hidden width or temporal evidence also missed 85%. This patch retains full GRU gating and the successful schedule but challenges the assumption that two independent memories of absolute spectra are preferable to a unified recurrent state with explicit local dynamics; its per-step recurrent cost is slightly lower (28,203 versus 28,383 MACs).
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3481673569400559960, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29830, "peak_hidden_elements": 91648, "recurrent_macs": 666577905, "recurrent_steps": 23635, "total_inference_macs": 667479295, "training_seconds": 49.95399195794016, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3698424544071127}

RECENT RESULT
hypothesis: Processing frames 3–30 with the current 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 29 to 28 recurrent steps and total inference MACs to approximately 644,493,850.
change: Omit the final input frame while preserving the successful recurrent architecture, readout, and training procedure.
mechanism: Opposite-boundary frame trimming on the delta-augmented unified GRU
evidence_used: The current unified GRU achieved 87.24% accuracy and substantially lower cross-entropy than the prior dual-GRU design. Removing the final frame from that weaker design reduced accuracy by only 0.74 points, so the current 2.24-point margin provides a grounded basis for retesting the larger step-level MAC reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3361777990710762500, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29830, "peak_hidden_elements": 91648, "recurrent_macs": 643592460, "recurrent_steps": 22820, "total_inference_macs": 644493850, "training_seconds": 41.33691154210828, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.37098437350220476}

RECENT RESULT
hypothesis: Processing frames 3–29 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 28 to 27 recurrent steps and total inference MACs from 644,493,850 to approximately 621,508,405.
change: Omit one additional trailing input frame while preserving the successful recurrent architecture, dual-timescale readout, classifier, and training procedure.
mechanism: Successive trailing-frame trimming
evidence_used: The unified GRU improved to 87.61% accuracy after the previous trailing-frame removal, leaving a 2.61-point margin above the requirement; that edit also reduced one full recurrent step without harming accuracy, directly motivating another one-step trim.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3241882412020965040, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 29830, "peak_hidden_elements": 91648, "recurrent_macs": 620607015, "recurrent_steps": 22005, "total_inference_macs": 621508405, "training_seconds": 48.18057529209182, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.36794182713046397}

RECENT RESULT
hypothesis: Processing frames 3–28 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 27 to 26 recurrent steps and total inference MACs from 621,508,405 to approximately 598,522,960.
change: Omit one additional trailing input frame while preserving the recurrent architecture, dual-timescale readout, classifier, and training procedure.
mechanism: Successive trailing-frame trimming
evidence_used: The same unified GRU remained above 87% accuracy after two consecutive trailing-frame removals, and the current 27-step model retains a 2.12-point margin above the requirement, directly motivating another one-step reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3121986833331167580, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 29830, "peak_hidden_elements": 91648, "recurrent_macs": 597621570, "recurrent_steps": 21190, "total_inference_macs": 598522960, "training_seconds": 40.93015079200268, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.37314739578340683}

RECENT RESULT
hypothesis: Processing frames 3–27 with the 79-unit spectrum-plus-delta GRU will retain at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps and total inference MACs from 598,522,960 to approximately 575,537,515.
change: Omit one additional trailing input frame while preserving the recurrent architecture, dual-timescale readout, classifier, and training procedure.
mechanism: Successive trailing-frame trimming
evidence_used: Three consecutive trailing-frame removals retained at least 86.87% accuracy; the current 26-step model remains 1.87 points above the requirement, directly motivating another one-step reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3002091254641370120, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 29830, "peak_hidden_elements": 91648, "recurrent_macs": 574636125, "recurrent_steps": 20375, "total_inference_macs": 575537515, "training_seconds": 43.83593229088001, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.38937989217372027}



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
