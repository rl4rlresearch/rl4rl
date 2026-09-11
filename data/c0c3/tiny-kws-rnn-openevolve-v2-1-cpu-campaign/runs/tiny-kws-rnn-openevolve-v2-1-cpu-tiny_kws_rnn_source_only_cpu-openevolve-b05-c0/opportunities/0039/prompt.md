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
verified_results: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1273274637120059906, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 522, "p95_recurrent_steps": 22, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 243345960, "recurrent_steps": 17930, "total_inference_macs": 244102280, "training_seconds": 47.512227582978085, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4581193608009011}
prior_hypothesis: Processing frames 3, 5–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 23 to 22 recurrent steps and total inference MACs by approximately 4.3%.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing frames 2–23, 25, 27, and 29 will achieve at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps.
change: Replace the contiguous 26-frame schedule with 25 frames spanning the stronger frames-2–29 window, omitting alternating late frames.
mechanism: Sparse tail sampling with preserved temporal coverage
evidence_used: Frames 2–29 achieved 86.50%, while contiguous frames 2–26 fell to 83.31%; preserving the later endpoint and broader temporal span directly tests whether coverage, rather than the 25-step count itself, caused that failure.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1446365031245192291, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 522, "p95_recurrent_steps": 25, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277285820, "training_seconds": 51.014770542038605, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.43589297803633054}

RECENT RESULT
hypothesis: Processing frames 2–22, 25, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 25 to 24 recurrent steps and total inference MACs by approximately 4%.
change: Remove frame 23 from the dense prefix while preserving the three late sampled frames and the full temporal endpoint.
mechanism: Dense-prefix trimming with preserved sparse tail coverage
evidence_used: The 25-step schedule spanning frames 2–29 achieved 85.52%, whereas the contiguous 25-step schedule ending at frame 26 achieved only 83.31%; this indicates late temporal coverage should be preserved while testing one fewer dense-prefix frame.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1388668233203481496, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 522, "p95_recurrent_steps": 24, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 265468320, "recurrent_steps": 19560, "total_inference_macs": 266224640, "training_seconds": 39.540001958142966, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.45352651209919}

RECENT RESULT
hypothesis: Processing frames 2–23, 25, and 29 will achieve at least 85% validation accuracy with 24 recurrent steps by preserving the successful dense prefix and temporal endpoint while omitting the likely redundant intermediate tail frame 27.
change: Change the 24-frame schedule to retain frame 23 and remove frame 27 from the sparse tail.
mechanism: Redundant mid-tail frame removal
evidence_used: The verified 25-step schedule using frames 2–23, 25, 27, and 29 achieved 85.52%, while removing frame 23 narrowly failed at 84.91%; this suggests preserving dense-prefix continuity and testing redundancy among the late samples.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1388668233203481496, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 522, "p95_recurrent_steps": 24, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 265468320, "recurrent_steps": 19560, "total_inference_macs": 266224640, "training_seconds": 46.14543274999596, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4569703927069354}

RECENT RESULT
hypothesis: Processing frames 2–23, 27, and 29 will achieve at least 85% validation accuracy with 24 recurrent steps by removing frame 25, which is temporally closest to the preserved dense prefix.
change: Remove frame 25 from the successful 25-step schedule while retaining the dense prefix and the later tail samples at frames 27 and 29.
mechanism: Near-prefix sparse-tail frame removal
evidence_used: The 25-step schedule using frames 2–23, 25, 27, and 29 achieved 85.52%; removing frame 23 failed at 84.91% and removing frame 27 failed at 84.66%, indicating that frame 25 is the remaining direct ablation candidate for reaching 24 steps.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1388668233203481496, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 522, "p95_recurrent_steps": 24, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 265468320, "recurrent_steps": 19560, "total_inference_macs": 266224640, "training_seconds": 40.900113959098235, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.44674931625647046}

RECENT RESULT
hypothesis: Processing frames 2–23 and 27 will retain at least 85% validation accuracy while reducing execution from 24 to 23 recurrent steps and total inference MACs by approximately 4.2%.
change: Remove frame 29 from the current schedule while preserving the dense prefix and frame 27.
mechanism: Tail-endpoint ablation with preserved informative mid-tail coverage
evidence_used: Removing frame 27 previously failed at 84.66% despite retaining frame 29, whereas removing frame 25 succeeded at 85.28%; this identifies frame 27 as the stronger late sample and makes frame 29 the most informative remaining ablation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1330971435161770701, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 522, "p95_recurrent_steps": 23, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 254407140, "recurrent_steps": 18745, "total_inference_macs": 255163460, "training_seconds": 35.89749474986456, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.4770843505859375}

RECENT RESULT
hypothesis: Replacing the eight-output classifier with seven learned log-odds and one fixed reference logit will retain at least 85% accuracy while removing 94,540 validation MACs and 117 parameters without reducing softmax expressivity.
change: Preserve the verified 24-step recurrent schedule and emit seven learned logits plus a zero reference-class logit.
mechanism: Reference-class logit parameterization
evidence_used: The current 24-step model achieved 85.28%, while reducing to 23 steps fell to 84.29%; preserving its recurrent computation and compressing the redundant eight-class logit representation is therefore the lower-risk cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1388175098178180439, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 522, "p95_recurrent_steps": 24, "parameters": 14779, "peak_hidden_elements": 59904, "recurrent_macs": 265468320, "recurrent_steps": 19560, "total_inference_macs": 266130100, "training_seconds": 34.319123042048886, "validation_accuracy": 0.8392638036809816, "validation_cross_entropy": 0.49036069530650883}

RECENT RESULT
hypothesis: Processing frames 3–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 24 to 23 recurrent steps and total inference MACs by approximately 4.2%.
change: Remove frame 2 from the verified 24-step schedule while preserving its dense command-bearing prefix and both informative late samples.
mechanism: Earliest-prefix frame ablation
evidence_used: A 29-step schedule beginning at frame 3 achieved 85.28%, showing frame 2 is not intrinsically required, while removing frame 29 from the current schedule failed at 84.29%; this makes the earliest prefix frame a better-supported 23-step ablation than removing late coverage.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1330971435161770701, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 522, "p95_recurrent_steps": 23, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 254407140, "recurrent_steps": 18745, "total_inference_macs": 255163460, "training_seconds": 32.027560166083276, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.44981204541914305}

RECENT RESULT
hypothesis: Processing frames 4–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 23 to 22 recurrent steps and total inference MACs by approximately 4.3%.
change: Remove frame 3 from the verified schedule while preserving the dense command-bearing region and both informative late samples.
mechanism: Iterative earliest-prefix frame ablation
evidence_used: Removing frame 2 from the verified 24-step schedule improved accuracy from 85.28% to 85.64%, while removing late frame 29 failed at 84.29%; this supports testing the next earliest frame without sacrificing late temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1273274637120059906, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 522, "p95_recurrent_steps": 22, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 243345960, "recurrent_steps": 17930, "total_inference_macs": 244102280, "training_seconds": 27.34699924988672, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.46668014058306173}

RECENT RESULT
hypothesis: Reducing the GRU width from 58 to 57 while preserving the verified 23-frame schedule will retain at least 85% validation accuracy and reduce total inference MACs from 255,163,460 to approximately 247,558,695.
change: Shrink the GRU hidden state by one unit and adjust the temporal summary and classifier dimensions accordingly.
mechanism: One-unit recurrent width contraction
evidence_used: The current 23-step, width-58 model achieved 85.64%, while reducing temporal coverage to 22 steps failed at 84.66%; preserving all successful frame selections and testing the smallest recurrent-width reduction is the best-supported orthogonal cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1291303823773123109, "maximum_recurrent_steps": 23, "mean_recurrent_steps": 23.0, "median_recurrent_steps": 23, "optimizer_steps": 522, "p95_recurrent_steps": 23, "parameters": 14469, "peak_hidden_elements": 58880, "recurrent_macs": 246815415, "recurrent_steps": 18745, "total_inference_macs": 247558695, "training_seconds": 45.6729015421588, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.45602725912456865}

RECENT RESULT
hypothesis: Processing frames 3, 5–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 23 to 22 recurrent steps and total inference MACs by approximately 4.3%.
change: Remove frame 4 while preserving frame 3 as the initial recurrent anchor, the remaining dense command-bearing prefix, and both informative late samples.
mechanism: Anchored-prefix interior-frame ablation
evidence_used: Removing frame 2 improved accuracy to 85.64%, whereas subsequently removing frame 3 fell to 84.66%; retaining frame 3 and ablating frame 4 directly tests whether the failure was specific to losing the earliest temporal anchor rather than requiring every early frame.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1273274637120059906, "maximum_recurrent_steps": 22, "mean_recurrent_steps": 22.0, "median_recurrent_steps": 22, "optimizer_steps": 522, "p95_recurrent_steps": 22, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 243345960, "recurrent_steps": 17930, "total_inference_macs": 244102280, "training_seconds": 47.512227582978085, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.4581193608009011}

RECENT RESULT
hypothesis: Processing frames 3, 6–23, 27, and 29 will retain at least 85% validation accuracy while reducing execution from 22 to 21 recurrent steps and total inference MACs by approximately 4.5%.
change: Remove frame 5 while retaining frame 3 as the initial recurrent anchor, the remaining dense command-bearing region, and both informative late samples.
mechanism: Anchored-prefix interior-frame ablation
evidence_used: Removing frame 3 failed at 84.66%, but retaining frame 3 and removing frame 4 achieved 85.03%; this motivates testing whether the next early interior frame is also redundant without sacrificing the demonstrated anchor or late temporal coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215577839078349111, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233041100, "training_seconds": 39.8916966249235, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.4791149864898869}

RECENT RESULT
hypothesis: Processing frames 3, 5, 7–23, 27, and 29 will achieve at least 85% validation accuracy with 21 recurrent steps because it preserves frame 5 and avoids the two-frame early gap created by the failed frame-5 ablation.
change: Remove frame 6 from the verified 22-step schedule, retaining frames 3 and 5 as early recurrent anchors plus the dense command-bearing region and both late samples.
mechanism: Preserve early anchors while ablating an adjacent redundant frame
evidence_used: The 22-step schedule using frames 3, 5–23, 27, and 29 achieved 85.03%, while removing frame 5 fell to 84.05%; testing frame 6 distinguishes whether that failure reflects frame 5’s specific importance or a general 21-step capacity limit.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1215577839078349111, "maximum_recurrent_steps": 21, "mean_recurrent_steps": 21.0, "median_recurrent_steps": 21, "optimizer_steps": 522, "p95_recurrent_steps": 21, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 232284780, "recurrent_steps": 17115, "total_inference_macs": 233041100, "training_seconds": 38.09167179092765, "validation_accuracy": 0.849079754601227, "validation_cross_entropy": 0.46601140308965205}



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
