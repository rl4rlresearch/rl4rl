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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3894385047229083012, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.300613496932517, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744744000, "recurrent_steps": 17360, "total_inference_macs": 746601120, "training_seconds": 151.4811316251289, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46449375269603144}
prior_hypothesis: Lowering the qualified penultimate-step exit threshold from 0.80 to 0.75 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 747,806,724.

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3928390144682457070, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.479754601226993, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 751007400, "recurrent_steps": 17506, "total_inference_macs": 753120312, "training_seconds": 125.33483404200524, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.46321860471386117}
prior_hypothesis: Lowering the early-exit threshold from 0.95 to 0.90 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 757,808,772 by skipping the final recurrent step for more examples.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3913250888966913825, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.4, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 748218900, "recurrent_steps": 17441, "total_inference_macs": 750217932, "training_seconds": 111.10649529099464, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4642757111531825}
prior_hypothesis: Lowering the penultimate-step exit threshold from 0.90 to 0.85 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 753,120,312.

REFERENCE DESIGN 3
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3900673661141693283, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.333742331288345, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 745902300, "recurrent_steps": 17387, "total_inference_macs": 747806724, "training_seconds": 120.20683200005442, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46505647436972775}
prior_hypothesis: Lowering the qualified early-exit threshold from 0.85 to 0.80 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 750,217,932.

## Recent verification evidence

RECENT RESULT
hypothesis: Omitting frame 0 from the qualified 110-unit, 23-step schedule while retaining frame 3 will achieve at least 85% validation accuracy with 22 recurrent steps and approximately 4.35% fewer recurrent MACs.
change: Exclude frame 0 in addition to frames 1, 2, and 27–30, preserving the qualified dual-readout GRU and training procedure.
mechanism: Leading-silence frame omission
evidence_used: The 23-step model qualified at 85.52% after frames 1 and 2 were removed, whereas removing frame 3 reduced accuracy to 84.54%; this isolates whether the remaining earliest frame is less informative than frame 3 at the same 22-step cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019730644237518698, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45368, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770631400, "training_seconds": 52.34599075000733, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46311264506146954}

RECENT RESULT
hypothesis: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when one of its 110 final-state readout coordinates is removed, while reducing dense classifier MACs by 6,520 and learned parameters by 8.
change: Adopt the qualified 22-frame schedule and classify from the complete 110-coordinate mean output plus 109 coordinates of the final hidden state.
mechanism: Single-coordinate complementary-readout pruning
evidence_used: The 110-unit, 22-step design achieved 85.03% accuracy, while reducing recurrent width to 109 units and reducing outputs to seven logits both failed; pruning one of 220 readout coordinates preserves the qualified recurrent path and all eight logits.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019696634925428970, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45360, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770624880, "training_seconds": 47.418838291894644, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4611427821996022}

RECENT RESULT
hypothesis: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when a second final-state coordinate is removed, reducing dense validation MACs by another 6,520.
change: Restore the qualified 22-frame schedule and 110-unit GRU, then classify from all 110 mean coordinates and 108 final-state coordinates.
mechanism: Second-coordinate complementary-readout pruning
evidence_used: The first final-state coordinate removal improved accuracy from 85.03% to 85.15% while reducing MACs; larger reductions to 109 recurrent units or seven learned logits failed, motivating another minimal readout-only reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019662625613339242, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45352, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770618360, "training_seconds": 79.3798720841296, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.46926335995914015}

RECENT RESULT
hypothesis: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when LayerNorm’s 40 affine parameters are removed, because the following GRU input weights and biases can represent the same scaling and offset, while inference MACs and recurrent steps remain unchanged and learned parameters decrease from 45,360 to 45,320.
change: Restore the qualified 22-step, 110-unit, 219-coordinate dual-readout design and make its input LayerNorm non-affine.
mechanism: Affine-free input normalization with GRU absorption
evidence_used: Reference Design 2 achieved 85.15% accuracy at 770,624,880 MACs after one final-state coordinate was pruned; the failed second-coordinate pruning motivates preserving that qualified recurrent path and readout while reducing only functionally redundant normalization parameters.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019696634925428930, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45320, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770624880, "training_seconds": 74.7782356669195, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46733713910623564}

RECENT RESULT
hypothesis: Removing the classifier’s eight bias parameters will retain at least 85% validation accuracy with unchanged MACs and recurrent steps, while reducing learned parameters from 45,360 to 45,352.
change: Preserve the qualified 110-unit, 22-step, 219-feature architecture and disable only the final classifier bias.
mechanism: Bias-free eight-logit readout
evidence_used: The current design achieved 85.15% accuracy; removing 40 LayerNorm affine parameters missed by only 0.09 points, while further feature pruning failed substantially, motivating a smaller parameter-only reduction that leaves every recurrent and readout coordinate intact.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019696634925428962, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45352, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770624880, "training_seconds": 64.83698537503369, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.46190509445096817}

RECENT RESULT
hypothesis: Removing only LayerNorm’s 20 bias parameters from the qualified 110-unit, 22-step, 219-feature bias-free-readout model will retain at least 85% validation accuracy with unchanged MACs and recurrent steps, while reducing learned parameters from 45,352 to 45,332.
change: Restore the qualified 219-feature bias-free classifier and disable only the input LayerNorm bias while retaining its learned scale.
mechanism: LayerNorm offset absorption into GRU biases
evidence_used: The 219-feature bias-free design achieved 85.40% accuracy; fully non-affine LayerNorm missed by only 0.09 points after removing both scale and offset, motivating the narrower test of removing the offset, which can be represented by the following GRU biases.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4019696634925428942, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 769197000, "recurrent_steps": 17930, "total_inference_macs": 770624880, "training_seconds": 65.64059208310209, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4620154632381135}

RECENT RESULT
hypothesis: Exiting examples whose penultimate-step confidence is at least 0.98 will preserve validation_accuracy >= 0.85 and reduce total_inference_macs below 770,624,880 by avoiding the final recurrent step for enough stable predictions.
change: Restore the qualified 110-unit, 22-step, 219-feature bias-free design with LayerNorm bias removed, then add a penultimate-step early exit; intermediate classifier calls use recurrent hidden logits without executing the dense readout.
mechanism: Confidence-gated penultimate-frame exit
evidence_used: Reference Design 3 achieved 85.28% at 22 steps and 770,624,880 MACs, while an unconditional alternative 22-step frame removal scored 84.54%; selectively skipping the isolated final frame only for already-confident examples is a lower-risk route to fewer MACs than another universal frame deletion.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3985220581522342482, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.779141104294478, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 761475000, "recurrent_steps": 17750, "total_inference_macs": 764015400, "training_seconds": 124.60310674994253, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46186555265648965}

RECENT RESULT
hypothesis: Lowering the qualified early-exit threshold from 0.98 to 0.95 will preserve validation_accuracy >= 0.85 while reducing total_inference_macs below 764,015,400 by skipping the final recurrent step for more stable predictions.
change: Restore the qualified 110-unit, 22-frame, 219-feature bias-free model with LayerNorm bias removed, and broaden its penultimate-step early exit to confidence >= 0.95.
mechanism: Expanded penultimate-frame confidence exit
evidence_used: The 0.98 exit model matched the no-exit model’s 85.276% accuracy exactly while eliminating 180 recurrent steps, indicating headroom to admit additional slightly lower-confidence penultimate predictions.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3952845865453719235, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.608588957055215, "median_recurrent_steps": 22, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 755511900, "recurrent_steps": 17611, "total_inference_macs": 757808772, "training_seconds": 117.85435270797461, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4621456310061589}

RECENT RESULT
hypothesis: Lowering the early-exit threshold from 0.95 to 0.90 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 757,808,772 by skipping the final recurrent step for more examples.
change: Broaden the qualified penultimate-step early exit to examples with maximum softmax confidence of at least 0.90.
mechanism: Expanded penultimate-frame confidence exit
evidence_used: Lowering the threshold from 0.98 to 0.95 preserved validation accuracy exactly at 85.276% while reducing mean recurrent steps from 21.779 to 21.609 and total inference MACs by 6,206,628, supporting another incremental threshold reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3928390144682457070, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.479754601226993, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 751007400, "recurrent_steps": 17506, "total_inference_macs": 753120312, "training_seconds": 125.33483404200524, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.46321860471386117}

RECENT RESULT
hypothesis: Lowering the penultimate-step exit threshold from 0.90 to 0.85 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 753,120,312.
change: Broaden the qualified penultimate-step early exit to examples with maximum softmax confidence of at least 0.85.
mechanism: Expanded penultimate-frame confidence exit
evidence_used: Lowering the threshold from 0.95 to 0.90 increased validation accuracy from 85.276% to 85.399% while reducing mean recurrent steps from 21.609 to 21.480 and total inference MACs by 4,688,460, supporting another incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3913250888966913825, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.4, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 748218900, "recurrent_steps": 17441, "total_inference_macs": 750217932, "training_seconds": 111.10649529099464, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.4642757111531825}

RECENT RESULT
hypothesis: Lowering the qualified early-exit threshold from 0.85 to 0.80 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 750,217,932.
change: Add the qualified intermediate-classification shortcut and exit examples at the penultimate recurrent step when classifier confidence is at least 0.80.
mechanism: Expanded penultimate-frame confidence exit
evidence_used: Lowering the threshold from 0.90 to 0.85 preserved 85.399% accuracy while reducing mean recurrent steps from 21.480 to 21.400 and total inference MACs by 2,902,380, motivating another incremental threshold reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3900673661141693283, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.333742331288345, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 745902300, "recurrent_steps": 17387, "total_inference_macs": 747806724, "training_seconds": 120.20683200005442, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.46505647436972775}

RECENT RESULT
hypothesis: Lowering the qualified penultimate-step exit threshold from 0.80 to 0.75 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 747,806,724.
change: Restore the qualified affine-scale-only input normalization and intermediate-classification shortcut, then exit penultimate-step examples whose softmax confidence is at least 0.75.
mechanism: Expanded penultimate-frame confidence exit
evidence_used: Lowering the threshold from 0.85 to 0.80 retained 85.276% accuracy while reducing mean recurrent steps from 21.400 to 21.334 and total inference MACs by 2,411,208, motivating another incremental threshold reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3894385047229083012, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.300613496932517, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744744000, "recurrent_steps": 17360, "total_inference_macs": 746601120, "training_seconds": 151.4811316251289, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.46449375269603144}



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
