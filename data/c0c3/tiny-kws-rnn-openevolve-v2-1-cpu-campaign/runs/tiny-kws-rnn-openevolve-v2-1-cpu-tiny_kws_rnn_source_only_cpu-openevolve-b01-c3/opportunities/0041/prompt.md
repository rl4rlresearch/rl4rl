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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1703662481126473990, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27270, "peak_hidden_elements": 101888, "recurrent_macs": 325321920, "recurrent_steps": 13040, "total_inference_macs": 326612880, "training_seconds": 131.3961366249714, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4360813538721003}
prior_hypothesis: A 99-unit state with 31/32/33 reset-update-candidate feedback widths will retain at least 85% validation accuracy while reducing total inference MACs to approximately 326,612,880.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1720871193043872584, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27544, "peak_hidden_elements": 102912, "recurrent_macs": 328608000, "recurrent_steps": 13040, "total_inference_macs": 329912000, "training_seconds": 106.52642687503248, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.42420507325716544}
prior_hypothesis: A 100-unit state with 31/32/33 reset-update-candidate feedback widths will achieve at least 85% validation accuracy at approximately 329,912,000 MACs by shifting one coordinate from the less capacity-sensitive reset path to the candidate path.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1775558166884143231, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28431, "peak_hidden_elements": 107008, "recurrent_macs": 339040000, "recurrent_steps": 13040, "total_inference_macs": 340396160, "training_seconds": 120.8442945831921, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4407977613203365}
prior_hypothesis: Retaining 32 feedback coordinates for the update and candidate paths while reducing only the reset path to 31 will preserve at least 85% validation accuracy and reduce total inference MACs from 343,108,480 to approximately 340,396,160.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1738079904961271178, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27818, "peak_hidden_elements": 103936, "recurrent_macs": 331894080, "recurrent_steps": 13040, "total_inference_macs": 333211120, "training_seconds": 108.37409804109484, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.428412754550302}
prior_hypothesis: A 101-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 336,510,240 to approximately 333,211,120.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Reducing only the reset feedback path from 31 to 30 coordinates will retain at least 85% validation accuracy while reducing total inference MACs from 340,396,160 to approximately 337,683,840.
change: Resize the shared recurrent projection from 95 to 94 outputs, split it into 30/32/32 reset-update-candidate codes, and resize only the reset expansion layer.
mechanism: Second selective reset-gate bottleneck trim
evidence_used: The selective 31/32/32 design qualified at 85.153% and outperformed the failed 31/31/31 design, indicating update and candidate capacity should remain intact while the reset path is the most promising target for another minimal structural reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1761410293054819502, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28222, "peak_hidden_elements": 107008, "recurrent_macs": 336327680, "recurrent_steps": 13040, "total_inference_macs": 337683840, "training_seconds": 116.85040212492459, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4404744434941766}

RECENT RESULT
hypothesis: Reducing the gated state from 104 to 103 units while retaining all three 32-unit feedback bottlenecks will preserve at least 85% validation accuracy and reduce total inference MACs from 343,108,480 to approximately 339,809,360.
change: Resize the input gate projections, recurrent state, gate expansions, temporal summary, and endpoint-augmented classifier from 104 to 103 units while preserving the qualified training procedure and 16-frame schedule.
mechanism: Gated-state width boundary trim
evidence_used: The qualified 104-state, 32/32/32 bottleneck design achieved 85.521% accuracy; unlike the failed bottleneck-rank trims, this nearest structural reduction preserves the full demonstrated feedback capacity of every gate and would improve upon the 340,396,160-MAC selective-reset design.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1772497328796068366, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28366, "peak_hidden_elements": 105984, "recurrent_macs": 338466240, "recurrent_steps": 13040, "total_inference_macs": 339809360, "training_seconds": 111.88948058406822, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4285078645483848}

RECENT RESULT
hypothesis: A 102-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 339,809,360 to approximately 336,510,240.
change: Replace the dense 93-unit GRU with a 102-unit bottleneck-gated recurrence, preserving the proven 16-frame schedule and mean/final-state readout.
mechanism: Full-rank gate feedback with a narrower recurrent state
evidence_used: The 103-state design with three 32-unit bottlenecks achieved 85.399% accuracy at 339,809,360 MACs, while bottleneck-rank reductions were less reliable; trimming state width by one is the closest untested structural reduction that preserves full gate-feedback capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1755288616878669772, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 28092, "peak_hidden_elements": 104960, "recurrent_macs": 335180160, "recurrent_steps": 13040, "total_inference_macs": 336510240, "training_seconds": 108.23541283304803, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4422338474016248}

RECENT RESULT
hypothesis: A 101-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 336,510,240 to approximately 333,211,120.
change: Replace the dense 96-unit GRU with a 101-unit bottleneck-gated recurrence, preserving the 16-frame schedule and mean/final-state readout.
mechanism: Full gate-feedback capacity with a narrower recurrent state
evidence_used: The adjacent 102-unit bottleneck-gated design qualified at 85.276% accuracy and 336,510,240 MACs; prior bottleneck-rank reductions were less reliable, so trimming state width by one is the closest supported structural reduction that preserves all demonstrated gate-feedback capacity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1738079904961271178, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27818, "peak_hidden_elements": 103936, "recurrent_macs": 331894080, "recurrent_steps": 13040, "total_inference_macs": 333211120, "training_seconds": 108.37409804109484, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.428412754550302}

RECENT RESULT
hypothesis: A 100-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 333,211,120 to approximately 329,912,000.
change: Reduce the recurrent state, input gate projections, gate expansions, temporal summary, and endpoint-augmented classifier from 101 to 100 units while preserving the qualified 16-frame schedule and training procedure.
mechanism: Full gate-feedback capacity with a one-unit state trim
evidence_used: The adjacent 101-unit design achieved 85.399% accuracy at 333,211,120 MACs, and each prior one-unit state trim from 103 through 101 qualified while saving exactly 3,299,120 MACs; keeping all three bottlenecks at 32 avoids the accuracy failures observed when feedback rank was reduced.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1720871193043872584, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27544, "peak_hidden_elements": 102912, "recurrent_macs": 328608000, "recurrent_steps": 13040, "total_inference_macs": 329912000, "training_seconds": 96.84299454209395, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.43103351241971816}

RECENT RESULT
hypothesis: A 100-unit state with 31/32/33 reset-update-candidate feedback widths will achieve at least 85% validation accuracy at approximately 329,912,000 MACs by shifting one coordinate from the less capacity-sensitive reset path to the candidate path.
change: Reduce the state to 100 units and reallocate the fixed 96-dimensional recurrent bottleneck from 32/32/32 to 31/32/33, preserving the 16-frame schedule and training procedure.
mechanism: Cost-neutral feedback-rank reallocation toward candidate generation
evidence_used: The 100-unit 32/32/32 model narrowly failed at 84.417%, while the qualified 104-unit 31/32/32 result shows the reset path can tolerate a one-coordinate reduction; the failure of 31/31/31 indicates update/candidate capacity is more valuable, motivating a cost-neutral transfer to the candidate path.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1720871193043872584, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27544, "peak_hidden_elements": 102912, "recurrent_macs": 328608000, "recurrent_steps": 13040, "total_inference_macs": 329912000, "training_seconds": 106.52642687503248, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.42420507325716544}

RECENT RESULT
hypothesis: A 99-unit state with 31/32/33 reset-update-candidate feedback widths will retain at least 85% validation accuracy while reducing total inference MACs to approximately 326,612,880.
change: Reduce the recurrent state and readout from 103 to 99 units, while reallocating the fixed 96-dimensional feedback bottleneck from 32/32/32 to 31/32/33.
mechanism: Candidate-biased feedback with a one-unit state trim
evidence_used: The qualified 100-unit 31/32/33 design achieved 85.890% accuracy at 329,912,000 MACs, providing the strongest margin among bottleneck-gated designs and making a one-unit state reduction the closest supported cost improvement.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1703662481126473990, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 27270, "peak_hidden_elements": 101888, "recurrent_macs": 325321920, "recurrent_steps": 13040, "total_inference_macs": 326612880, "training_seconds": 131.3961366249714, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4360813538721003}

RECENT RESULT
hypothesis: A 128-unit gated model that retains all 16 sampled frames but performs one learned transition per ordered two-frame block will reach at least 85% validation accuracy while reducing total inference MACs to approximately 262,886,400.
change: Replace per-frame learned transitions with alternating frame buffering and eight gated block transitions over local mean/difference features; widen the state to 128 units and classify from temporal mean, maximum, and final state.
mechanism: Ordered two-frame block recurrence with mean-max-endpoint readout
evidence_used: The 100-unit 31/32/33 model reached 85.890% accuracy, showing the candidate-biased gated mechanism has useful margin. Unlike the failed 15-frame reduction, this patch preserves all 16 observations and challenges the load-bearing assumption that every observation requires a separate expensive recurrent transition.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 1371255464761567360, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 786, "p95_recurrent_steps": 16, "parameters": 43920, "peak_hidden_elements": 207872, "recurrent_macs": 260382720, "recurrent_steps": 13040, "total_inference_macs": 262886400, "training_seconds": 85.1744798750151, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.47758421166542847}



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
