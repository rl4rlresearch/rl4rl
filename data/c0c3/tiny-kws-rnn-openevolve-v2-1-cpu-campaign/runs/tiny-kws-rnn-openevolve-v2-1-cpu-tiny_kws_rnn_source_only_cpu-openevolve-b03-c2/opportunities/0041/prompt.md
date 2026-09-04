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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3894385047229083012, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.300613496932517, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744744000, "recurrent_steps": 17360, "total_inference_macs": 746601120, "training_seconds": 124.43413041601889, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4645164817388804}
prior_hypothesis: A 0.7475 penultimate-step exit threshold will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120 by admitting safe examples between the qualified 0.75 and failed 0.745 boundaries.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Lowering the penultimate-step exit threshold from 0.750 to 0.745 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120.
change: Admit a small additional band of high-confidence examples to the qualified penultimate-step early exit.
mechanism: Fine-grained confidence-boundary expansion
evidence_used: The 0.75 threshold qualified at 85.03% and reduced recurrent execution by 27 steps versus 0.80; the narrow 0.005 change seeks further savings while respecting the now-small accuracy margin.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3893453400723511120, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.295705521472392, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744572400, "recurrent_steps": 17356, "total_inference_macs": 746422512, "training_seconds": 144.81303000007756, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4647272683360094}

RECENT RESULT
hypothesis: A 0.7475 penultimate-step exit threshold will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120 by admitting safe examples between the qualified 0.75 and failed 0.745 boundaries.
change: Lower the qualified early-exit threshold from 0.80 to the midpoint 0.7475.
mechanism: Confidence-boundary bisection
evidence_used: Threshold 0.75 qualified at 85.03% with 746,601,120 MACs, whereas 0.745 saved four additional recurrent steps but missed accuracy by one validation example; bisecting this narrow interval is the most informative next boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3894385047229083012, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.300613496932517, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744744000, "recurrent_steps": 17360, "total_inference_macs": 746601120, "training_seconds": 124.43413041601889, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4645164817388804}

RECENT RESULT
hypothesis: A 0.74625 penultimate-step exit threshold will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120 by safely admitting a subset of the four examples introduced at 0.745.
change: Lower the current penultimate-step confidence threshold from 0.85 to 0.74625.
mechanism: Confidence-boundary bisection
evidence_used: Thresholds 0.75 and 0.7475 both achieved 85.03% accuracy with 746,601,120 MACs, while 0.745 saved four more recurrent steps but lost exactly one validation example; bisecting the remaining interval is the most informative boundary test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3893686312349904093, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.296932515337424, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744615300, "recurrent_steps": 17357, "total_inference_macs": 746467164, "training_seconds": 134.03014808311127, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46476229802231117}

RECENT RESULT
hypothesis: A 0.746875 penultimate-step threshold will retain validation_accuracy >= 0.85 and may reduce total_inference_macs below 746,601,120 by admitting a safe subset of examples excluded at 0.7475.
change: Lower the current penultimate-step confidence threshold from 0.90 to 0.746875.
mechanism: Confidence-boundary bisection
evidence_used: The 0.7475 threshold qualified at 85.03% and 746,601,120 MACs, while 0.74625 admitted three additional exits but missed the accuracy requirement by one example; bisecting their interval directly probes the remaining boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3893919223976297066, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 21.298159509202453, "median_recurrent_steps": 21, "optimizer_steps": 397, "p95_recurrent_steps": 22, "parameters": 45332, "peak_hidden_elements": 113152, "recurrent_macs": 744658200, "recurrent_steps": 17358, "total_inference_macs": 746511816, "training_seconds": 118.09068270795979, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.4647686074847824}



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
