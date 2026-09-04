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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2006345359458497350, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35295, "peak_hidden_elements": 92672, "recurrent_macs": 383728080, "recurrent_steps": 20375, "total_inference_macs": 384640880, "training_seconds": 97.75180570804514, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4321110637641392}
prior_hypothesis: Updating the 64-unit slow GRU every third frame instead of every second frame will retain at least 85% validation accuracy while reducing total inference MACs from 454,743,920 to approximately 384,640,880.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Compressing only the two highest-mel delta channels will retain at least 85% accuracy while reducing total inference MACs from 575,537,515 to approximately 570,708,640.
change: Preserve all 20 normalized spectral bands, the 79-unit GRU, 25-frame schedule, and readout, but average the final two delta bands into one feature and reduce GRU input width from 40 to 39.
mechanism: Delta-only high-band coalescing
evidence_used: The current spectrum-plus-delta model achieves 85.77% at 25 steps. The prior 20-to-19 spectral merge failed, motivating the more conservative reduction of only redundant local-dynamics detail while retaining every absolute spectral band.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2976903107874921008, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 29593, "peak_hidden_elements": 91648, "recurrent_macs": 569807250, "recurrent_steps": 20375, "total_inference_macs": 570708640, "training_seconds": 45.72943054209463, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.3934491397413008}

RECENT RESULT
hypothesis: Pairwise averaging the four highest-mel delta channels will retain at least 85% validation accuracy while reducing total inference MACs from 570,708,640 to approximately 565,879,765.
change: Preserve all spectral bands, 25 recurrent steps, 79-unit GRU, and readout while representing delta bands 16–19 as two pairwise averages and reducing GRU input width from 39 to 38.
mechanism: Progressive high-band delta coalescing
evidence_used: Coalescing only delta bands 18–19 improved accuracy from 85.77% to 86.75% while reducing MACs, whereas merging absolute spectral bands failed; this motivates another conservative reduction confined to adjacent high-band dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2951714961108471896, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 29356, "peak_hidden_elements": 91648, "recurrent_macs": 564978375, "recurrent_steps": 20375, "total_inference_macs": 565879765, "training_seconds": 46.499366333009675, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.39710100794131037}

RECENT RESULT
hypothesis: Pairwise averaging the six highest-mel delta channels will retain at least 85% validation accuracy while reducing GRU input width from 38 to 37 and total inference MACs by approximately 4.83 million.
change: Preserve all 20 spectral bands, 25 recurrent steps, 79-unit GRU, and dual-timescale readout, while representing delta bands 14–19 as three adjacent pairwise averages.
mechanism: Progressive high-band delta coalescing
evidence_used: Compressing delta bands 16–19 into two pairwise averages retained 86.38% accuracy and reduced MACs, while prior absolute-spectrum compression failed; this supports testing one further local reduction confined to adjacent high-band dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2926526814342022784, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 29119, "peak_hidden_elements": 91648, "recurrent_macs": 560149500, "recurrent_steps": 20375, "total_inference_macs": 561050890, "training_seconds": 63.54470487497747, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.4091079524689657}

RECENT RESULT
hypothesis: Pairwise averaging the eight highest-mel delta channels will retain at least 85% validation accuracy while reducing GRU input width from 37 to 36 and total inference MACs by exactly 4,828,875.
change: Preserve all 20 spectral bands, 25 recurrent steps, the 79-unit GRU, and dual-timescale readout, while representing delta bands 12–19 as four adjacent pairwise averages.
mechanism: Progressive high-band delta coalescing
evidence_used: Pairwise averaging delta bands 14–19 retained 86.38% accuracy and reduced total inference MACs to 561,050,890; the remaining 1.38-point margin motivates one further conservative compression confined to adjacent high-band dynamics.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2901338667575573672, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 28882, "peak_hidden_elements": 91648, "recurrent_macs": 555320625, "recurrent_steps": 20375, "total_inference_macs": 556222015, "training_seconds": 62.80980566702783, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4299747303219661}

RECENT RESULT
hypothesis: Pairwise averaging the ten highest-mel delta channels will retain at least 85% validation accuracy while reducing GRU input width from 36 to 35 and total inference MACs by exactly 4,828,875.
change: Preserve all 20 spectral bands, 25 recurrent steps, the 79-unit GRU, and dual-timescale readout, while representing delta bands 10–19 as five adjacent pairwise averages.
mechanism: Progressive high-band delta coalescing
evidence_used: Four successive high-band delta coalescing designs retained the accuracy target while reducing MACs; the current 36-input model achieves 85.64%, motivating one further adjacent-pair compression.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2876150520809124560, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 28645, "peak_hidden_elements": 91648, "recurrent_macs": 550491750, "recurrent_steps": 20375, "total_inference_macs": 551393140, "training_seconds": 60.38359549990855, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.3924435761808618}

RECENT RESULT
hypothesis: A 48-unit GRU updated every frame plus a 64-unit GRU updated after each frame pair will retain at least 85% validation accuracy while reducing expected total inference MACs from 551,393,140 to approximately 454,743,920.
change: Replace the single same-rate 79-unit GRU with fully gated fast and slow GRU cells; the fast state integrates all 25 frames, while the slow state receives contextualized fast states only at 12 pair boundaries. Predict from mean-fast, final-fast, and slow memories.
mechanism: Hierarchical two-rate gated memory
evidence_used: Full GRU gating is load-bearing—the unified 79-unit GRU reached 86.75%, whereas coupled-gate alternatives missed 85%. This challenges the shared assumption that all memory capacity must execute at every frame while preserving full GRU gates, the verified input features, temporal coverage, and comparable learned capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2372013483047166790, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35295, "peak_hidden_elements": 92672, "recurrent_macs": 453831120, "recurrent_steps": 20375, "total_inference_macs": 454743920, "training_seconds": 134.10849058395252, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.41974893581647815}

RECENT RESULT
hypothesis: Updating the 64-unit slow GRU every third frame instead of every second frame will retain at least 85% validation accuracy while reducing total inference MACs from 454,743,920 to approximately 384,640,880.
change: Preserve the successful 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, features, readout, and training procedure, but reduce slow-memory execution from 12 to 8 updates per example.
mechanism: Three-frame hierarchical gated memory
evidence_used: The hierarchical two-rate model reached 86.38% accuracy while showing that the slow recurrent capacity need not execute at every frame; its 1.38-point margin supports testing a lower slow-state update frequency that removes four costly 64-unit GRU executions per example.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2006345359458497350, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 397, "p95_recurrent_steps": 25, "parameters": 35295, "peak_hidden_elements": 92672, "recurrent_macs": 383728080, "recurrent_steps": 20375, "total_inference_macs": 384640880, "training_seconds": 97.75180570804514, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4321110637641392}



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
