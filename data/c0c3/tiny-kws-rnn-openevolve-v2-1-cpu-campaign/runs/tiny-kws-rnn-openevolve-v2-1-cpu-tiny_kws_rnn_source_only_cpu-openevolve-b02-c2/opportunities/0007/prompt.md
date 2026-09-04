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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 120.51731466688216, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4414740345960746}
prior_hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by 12% versus the qualified 120-unit design.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6860358437346396608, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1314432000, "recurrent_steps": 26080, "total_inference_macs": 1315214400, "training_seconds": 144.4831611670088, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4347592968150882}
prior_hypothesis: A 120-unit GRU using all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 11.3% versus the qualified 128-unit baseline.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 96.61382112489082, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}
prior_hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the qualified 112-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing execution from 32 to 24 recurrent steps will cut recurrent MACs by exactly 25% while retaining at least 85% accuracy because the 0.8724 baseline provides margin and adjacent log-mel frames are temporally redundant.
change: Skip one frame from each four-frame group while preserving the first and final frames.
mechanism: Causal three-of-four temporal frame sampling
evidence_used: The current 128-unit GRU reaches 0.872392638 validation accuracy but spends 1,482,178,560 recurrent MACs across 32 steps per example, making temporal thinning the highest-leverage isolated reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 101.92465950013138, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4628353469942245}

RECENT RESULT
hypothesis: Processing 28 evenly spaced frames will retain at least 85% validation accuracy while reducing recurrent MACs by exactly 12.5% versus the qualified 32-step baseline.
change: Skip four interior frames distributed across the recording while preserving the first and final frames.
mechanism: Evenly distributed seven-of-eight temporal sampling
evidence_used: The 32-step model achieved 87.24% accuracy, while the more aggressive 24-step schedule achieved 84.66%; testing the midpoint directly targets the cost–accuracy boundary with substantially more margin than the failed design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 120.53622858412564, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.46104526636790644}

RECENT RESULT
hypothesis: A 120-unit GRU using all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 11.3% versus the qualified 128-unit baseline.
change: Reduce GRU state and classifier width from 128 to 120 without changing temporal sampling or training.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 128-unit, 32-step model achieved 87.24% accuracy, while both 24- and 28-step variants failed; preserving every frame tests whether temporal information, rather than modest excess hidden capacity, caused those regressions.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6860358437346396608, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 52128, "peak_hidden_elements": 123392, "recurrent_macs": 1314432000, "recurrent_steps": 26080, "total_inference_macs": 1315214400, "training_seconds": 144.4831611670088, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4347592968150882}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by 12% versus the qualified 120-unit design.
change: Reduce the GRU state, temporal summary, and classifier width from 128 to 112 while preserving every input frame and the established training procedure.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 120-unit full-resolution GRU achieved 86.50% accuracy, whereas reducing temporal steps caused both tested models to fail; this motivates retaining all 32 frames and testing the remaining capacity margin through a moderate width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 120.51731466688216, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4414740345960746}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the qualified 112-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 104 while preserving the proven full-frame schedule and training procedure.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 112-unit full-resolution GRU achieved 86.13% accuracy and improved cost over the qualified 120-unit model, while reduced-frame designs failed; another eight-unit width reduction is the most direct test of remaining capacity margin.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 96.61382112489082, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4562299014600508}

RECENT RESULT
hypothesis: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 13% versus the qualified 104-unit design.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 96 while preserving the full-frame schedule and established training procedure.
mechanism: Full-resolution recurrent width reduction
evidence_used: The 104-unit full-resolution GRU achieved 85.89% accuracy, while prior 112- and 120-unit reductions also remained qualified; reduced-frame designs failed, so testing the next eight-unit width reduction is the most informative cost-boundary experiment.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4547997289742137040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 34800, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 871906560, "training_seconds": 100.25730083300732, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.44714848805058954}



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
