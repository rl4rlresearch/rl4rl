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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1775558166884143231, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28431, "peak_hidden_elements": 107008, "recurrent_macs": 339040000, "recurrent_steps": 13040, "total_inference_macs": 340396160, "training_seconds": 120.8442945831921, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4407977613203365}
prior_hypothesis: Retaining 32 feedback coordinates for the update and candidate paths while reducing only the reset path to 31 will preserve at least 85% validation accuracy and reduce total inference MACs from 343,108,480 to approximately 340,396,160.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2150748897857939461, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 33621, "peak_hidden_elements": 95744, "recurrent_macs": 411112080, "recurrent_steps": 13040, "total_inference_macs": 412324800, "training_seconds": 46.76749200001359, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4099570151487011}
prior_hypothesis: A 93-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 420,435,680 to approximately 412,324,800.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1789706040713466960, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28640, "peak_hidden_elements": 107008, "recurrent_macs": 341752320, "recurrent_steps": 13040, "total_inference_macs": 343108480, "training_seconds": 133.4293375830166, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4200194938051189}
prior_hypothesis: A 104-unit gated state using independent 32-unit recurrent bottlenecks will retain at least 85% accuracy while reducing estimated total inference MACs from 412,324,800 to approximately 343,108,480.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2278895985812006368, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 435640320, "recurrent_steps": 13040, "total_inference_macs": 436892160, "training_seconds": 48.260580625152215, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4099158445018932}
prior_hypothesis: A 96-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 445,237,760 to approximately 436,892,160.

## Recent verification evidence

RECENT RESULT
hypothesis: A 100-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 478,744,040 to approximately 470,092,000.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 100 units while preserving the qualified batch-64 training procedure and 16-frame even-index schedule.
mechanism: One-unit recurrent width boundary trim
evidence_used: The adjacent 101-unit design qualified at 85.276% accuracy and 478,744,040 MACs; 100 units is the smallest untested structural reduction and isolates the next cost boundary without changing temporal coverage or optimization.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2452071402972862488, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 37448, "peak_hidden_elements": 102912, "recurrent_macs": 469440000, "recurrent_steps": 13040, "total_inference_macs": 470092000, "training_seconds": 63.406882541952655, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.44861407133699194}

RECENT RESULT
hypothesis: A 99-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 470,092,000 to approximately 461,518,200.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 100 to 99 units while preserving the qualified training procedure and 16-frame schedule.
mechanism: One-unit recurrent width boundary trim
evidence_used: The adjacent 100-unit design achieved 85.521% accuracy at 470,092,000 MACs; its 0.521-point margin makes 99 units the smallest untested lower-cost boundary, while the failed 15-frame experiment supports keeping temporal coverage unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2407349157574880017, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36777, "peak_hidden_elements": 101888, "recurrent_macs": 460872720, "recurrent_steps": 13040, "total_inference_macs": 461518200, "training_seconds": 64.19769616704434, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4651131003912241}

RECENT RESULT
hypothesis: A 98-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 461,518,200 to approximately 453,022,640.
change: Reduce the GRU hidden state, temporal summary, and classifier width from 101 to 98 units while preserving the qualified batch-64 training procedure and 16-frame even-index schedule.
mechanism: One-unit recurrent width boundary trim
evidence_used: The adjacent 99-unit design achieved 85.521% accuracy at 461,518,200 MACs; 98 units is the next untested structural reduction, while the failed 15-frame design indicates temporal coverage should remain unchanged.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2363035023921974192, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36112, "peak_hidden_elements": 100864, "recurrent_macs": 452383680, "recurrent_steps": 13040, "total_inference_macs": 453022640, "training_seconds": 57.88504658290185, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.4804356393638564}

RECENT RESULT
hypothesis: A 98-unit GRU classified from both its mean output and final hidden state will recover at least 85% validation accuracy while requiring approximately 453,661,600 total inference MACs.
change: Reduce the GRU to 98 units and replace the mean-only classifier with a single linear classifier over the concatenated temporal mean and final hidden state.
mechanism: Endpoint-augmented temporal readout
evidence_used: The mean-only 98-unit model narrowly missed at 84.785% accuracy, while 99 units qualified at 85.521%. Adding the final-state feature costs only 638,960 MACs over the failed 98-unit design and remains 7,856,600 MACs below the qualified 99-unit design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2366367936506767536, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36896, "peak_hidden_elements": 100864, "recurrent_macs": 452383680, "recurrent_steps": 13040, "total_inference_macs": 453661600, "training_seconds": 53.559765374986455, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.3973136574212759}

RECENT RESULT
hypothesis: A 97-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 453,661,600 to approximately 445,237,760.
change: Reduce the recurrent state and temporal summary from 98 to 97 units and resize the endpoint-augmented classifier from 196 to 194 inputs.
mechanism: Endpoint-augmented one-unit width trim
evidence_used: The 98-unit mean-only model narrowly failed at 84.785%, while adding the final hidden state raised accuracy to 85.644%; this motivates testing whether the stronger readout supports the next one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2322427905286848629, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 36229, "peak_hidden_elements": 99840, "recurrent_macs": 443972880, "recurrent_steps": 13040, "total_inference_macs": 445237760, "training_seconds": 48.569967083167285, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.3981554089879697}

RECENT RESULT
hypothesis: A 96-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 445,237,760 to approximately 436,892,160.
change: Reduce the recurrent state and temporal summary to 96 units and classify from the concatenated temporal mean and final hidden state.
mechanism: Endpoint-augmented one-unit width trim
evidence_used: The qualified 97-unit endpoint-augmented design achieved 86.135% accuracy at 445,237,760 MACs, providing a 1.135-point margin for testing the next one-unit structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2278895985812006368, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 435640320, "recurrent_steps": 13040, "total_inference_macs": 436892160, "training_seconds": 48.260580625152215, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4099158445018932}

RECENT RESULT
hypothesis: A 95-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 436,892,160 to approximately 428,624,800.
change: Reduce the recurrent state and temporal summary from 101 to 95 units and replace the mean-only classifier with a single linear classifier over the 190-feature concatenation of temporal mean and final hidden state.
mechanism: Endpoint-augmented one-unit width trim
evidence_used: The qualified 96-unit endpoint-augmented design achieved 85.276% accuracy at 436,892,160 MACs; 95 units is the next untested structural reduction and preserves the 16-frame coverage shown to be important.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2235772178082240753, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 34913, "peak_hidden_elements": 97792, "recurrent_macs": 427386000, "recurrent_steps": 13040, "total_inference_macs": 428624800, "training_seconds": 49.897612374974415, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.40936705232397913}

RECENT RESULT
hypothesis: A 94-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 428,624,800 to approximately 420,435,680.
change: Use batch size 64, reduce the GRU and state width to 94, classify from the 188-feature mean/final-state concatenation, and process the established 16-frame even-index schedule.
mechanism: Endpoint-augmented one-unit width trim
evidence_used: The adjacent 95-unit endpoint-augmented design achieved 85.521% accuracy at 428,624,800 MACs; 94 units is the next untested structural reduction, while the failed 15-frame experiment supports preserving 16-frame temporal coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2193056482097551784, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 34264, "peak_hidden_elements": 96768, "recurrent_macs": 419209920, "recurrent_steps": 13040, "total_inference_macs": 420435680, "training_seconds": 43.08981949998997, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.39504253176823717}

RECENT RESULT
hypothesis: A 93-unit GRU using concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs from 420,435,680 to approximately 412,324,800.
change: Reduce the GRU, recurrent state, temporal summary, and classifier input width from 94 to 93 units while preserving the qualified 16-frame schedule and training procedure.
mechanism: Endpoint-augmented one-unit width trim
evidence_used: The adjacent 94-unit design achieved 87.239% accuracy at 420,435,680 MACs, leaving a 2.239-point margin and making 93 units the most informative next structural reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2150748897857939461, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 33621, "peak_hidden_elements": 95744, "recurrent_macs": 411112080, "recurrent_steps": 13040, "total_inference_macs": 412324800, "training_seconds": 46.76749200001359, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4099570151487011}

RECENT RESULT
hypothesis: A 104-unit gated state using independent 32-unit recurrent bottlenecks will retain at least 85% accuracy while reducing estimated total inference MACs from 412,324,800 to approximately 343,108,480.
change: Replace the dense GRU with a custom GRU-style update whose reset, update, and candidate feedback each pass through a structurally reduced nonlinear bottleneck; retain the proven 16-frame schedule and mean/final-state readout.
mechanism: Gate-specific nonlinear bottleneck recurrence
evidence_used: The 93-unit dense GRU reached 86.258% accuracy, whereas the ungated stacked tanh RNN reached only 80.368%. This suggests gating is load-bearing, but does not establish that every gate needs full-rank hidden-to-hidden feedback; the patch preserves gating while directly testing that shared assumption.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1789706040713466960, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28640, "peak_hidden_elements": 107008, "recurrent_macs": 341752320, "recurrent_steps": 13040, "total_inference_macs": 343108480, "training_seconds": 133.4293375830166, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4200194938051189}

RECENT RESULT
hypothesis: Reducing each recurrent feedback bottleneck from 32 to 31 units while preserving the 104-unit state and 16-frame schedule will retain at least 85% accuracy and reduce total inference MACs from 343,108,480 to approximately 334,971,520.
change: Replace the dense 96-unit GRU with the qualified gated bottleneck recurrence, using a 93-unit shared projection split into three independent 31-unit feedback paths.
mechanism: Gate-bottleneck rank trim
evidence_used: The 104-state, 32-unit-per-gate bottleneck design achieved 85.521% accuracy at 343,108,480 MACs; trimming one coordinate per gate is the nearest untested structural reduction and preserves its demonstrated gating, state capacity, temporal coverage, and readout.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1747262419225495773, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28013, "peak_hidden_elements": 107008, "recurrent_macs": 333615360, "recurrent_steps": 13040, "total_inference_macs": 334971520, "training_seconds": 130.84056287491694, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.42294873196654525}

RECENT RESULT
hypothesis: Retaining 32 feedback coordinates for the update and candidate paths while reducing only the reset path to 31 will preserve at least 85% validation accuracy and reduce total inference MACs from 343,108,480 to approximately 340,396,160.
change: Replace the dense 97-unit GRU with the qualified 104-unit gated bottleneck recurrence, using feedback widths of 31, 32, and 32 for the reset, update, and candidate gates respectively.
mechanism: Selective reset-gate bottleneck trim
evidence_used: The 104-state design with three 32-unit gate bottlenecks achieved 85.521% accuracy at 343,108,480 MACs, while trimming all three paths to 31 narrowly failed at 84.663%; isolating the reduction to the reset path tests a smaller structural cut while preserving full update and candidate capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1775558166884143231, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28431, "peak_hidden_elements": 107008, "recurrent_macs": 339040000, "recurrent_steps": 13040, "total_inference_macs": 340396160, "training_seconds": 120.8442945831921, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4407977613203365}



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
