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
verified_results: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1388668233203481496, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 522, "p95_recurrent_steps": 24, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 265468320, "recurrent_steps": 19560, "total_inference_macs": 266224640, "training_seconds": 40.900113959098235, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.44674931625647046}
prior_hypothesis: Processing frames 2–23, 27, and 29 will achieve at least 85% validation accuracy with 24 recurrent steps by removing frame 25, which is temporally closest to the preserved dense prefix.

## Recent verification evidence

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 29 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.3% versus the verified 30-frame model.
change: Omit the first three input frames while preserving the contiguous final 29-frame sequence and all training settings.
mechanism: Incremental causal prefix trimming
evidence_used: The final-30-frame model achieved 85.52% accuracy after the 31-frame model achieved 85.77%; removing one additional earliest frame is the smallest structural cost reduction and directly tests the remaining temporal boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1677152223412035471, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 522, "p95_recurrent_steps": 29, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 320774220, "recurrent_steps": 23635, "total_inference_macs": 321530540, "training_seconds": 68.43077654205263, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.447176777658287}

RECENT RESULT
hypothesis: The 58-unit GRU processing the final 28 frames will retain at least 85% validation accuracy while reducing total inference MACs and recurrent steps by approximately 3.4% versus the verified 29-frame model.
change: Omit the first four input frames while preserving the contiguous final 28-frame sequence and all other model and training settings.
mechanism: Incremental causal prefix trimming
evidence_used: The final-29-frame model achieved 85.28% accuracy after successive one-frame prefix trims through 31 and 30 frames remained viable; removing one additional earliest frame is the smallest direct test of the remaining temporal boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1619455425370324676, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 522, "p95_recurrent_steps": 28, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 309713040, "recurrent_steps": 22820, "total_inference_macs": 310469360, "training_seconds": 74.87525541614741, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4737021978647431}

RECENT RESULT
hypothesis: A 59-unit GRU processing the final 28 frames will recover validation accuracy to at least 85% while reducing total inference MACs by approximately 0.5% and recurrent steps from 29 to 28 versus the verified 58-unit, 29-frame model.
change: Increase recurrent and summary width from 58 to 59 units, resize the classifier, and omit the first four frames.
mechanism: Near-isocompute width–time reallocation
evidence_used: The 58-unit, 28-frame model narrowly missed the target at 84.66%, while 29 frames achieved 85.28%; adding one hidden unit is the smallest capacity increase and still leaves the 28-step recurrent computation below the current model’s MAC count.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1668445839435566769, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 522, "p95_recurrent_steps": 28, "parameters": 15329, "peak_hidden_elements": 60928, "recurrent_macs": 319092060, "recurrent_steps": 22820, "total_inference_macs": 319861420, "training_seconds": 59.07418387499638, "validation_accuracy": 0.8306748466257668, "validation_cross_entropy": 0.4990342403482074}

RECENT RESULT
hypothesis: Processing frames 2–29 will achieve at least 85% validation accuracy while matching the failed final-28-frame model’s 310,469,360 MACs and 28 recurrent steps.
change: Replace four-frame prefix trimming with two-frame trimming at each boundary, preserving 28 contiguous central frames.
mechanism: Symmetric boundary trimming
evidence_used: Frames 2–31 achieved 85.52% accuracy, whereas removing two additional leading frames fell narrowly to 84.66%; removing the likely redundant final two frames instead directly tests whether retaining earlier command onset information recovers accuracy at identical cost.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1619455425370324676, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 522, "p95_recurrent_steps": 28, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 309713040, "recurrent_steps": 22820, "total_inference_macs": 310469360, "training_seconds": 65.47450574999675, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4348838946570648}

RECENT RESULT
hypothesis: Processing frames 2–28 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.6% versus the verified 28-frame model.
change: Preserve the informative starting frame and omit one additional trailing frame, producing a contiguous 27-frame schedule.
mechanism: Incremental trailing-boundary trimming
evidence_used: Frames 2–29 achieved 86.50% accuracy at 28 steps, outperforming the 29-step final-frame schedule; this suggests earlier command-onset information is more valuable than the recording tail and motivates the smallest further trailing trim.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1561758627328613881, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 522, "p95_recurrent_steps": 27, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 298651860, "recurrent_steps": 22005, "total_inference_macs": 299408180, "training_seconds": 55.067767082946375, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.44136404961896086}

RECENT RESULT
hypothesis: Processing frames 2–27 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.7% versus the verified 27-step model.
change: Omit one additional trailing frame, producing a contiguous 26-frame schedule while preserving the model and training procedure.
mechanism: Incremental trailing-boundary trimming
evidence_used: Frames 2–28 achieved 86.26% accuracy after frames 2–29 achieved 86.50%, indicating that incremental removal from the less-informative recording tail has preserved a meaningful margin above the target.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1504061829286903086, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 522, "p95_recurrent_steps": 26, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 287590680, "recurrent_steps": 21190, "total_inference_macs": 288347000, "training_seconds": 60.41266495781019, "validation_accuracy": 0.8515337423312883, "validation_cross_entropy": 0.4519053921377732}

RECENT RESULT
hypothesis: Processing frames 2–26 will retain at least 85% validation accuracy while reducing recurrent steps and total inference MACs by approximately 3.8% versus the verified 26-step model.
change: Omit one additional trailing frame, producing a contiguous 25-frame schedule while preserving model capacity and training procedure.
mechanism: Incremental trailing-boundary trimming
evidence_used: Frames 2–27 achieved 85.15% accuracy after successive trailing trims at 28 and 27 endpoints also met the target; this is the smallest structural reduction that directly tests the remaining temporal boundary, though the narrow margin makes it a high-information threshold test.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 96, "examples_processed": 50000, "inference_cost": 1446365031245192291, "maximum_recurrent_steps": 25, "mean_recurrent_steps": 25.0, "median_recurrent_steps": 25, "optimizer_steps": 522, "p95_recurrent_steps": 25, "parameters": 14896, "peak_hidden_elements": 59904, "recurrent_macs": 276529500, "recurrent_steps": 20375, "total_inference_macs": 277285820, "training_seconds": 54.23268970893696, "validation_accuracy": 0.8331288343558282, "validation_cross_entropy": 0.46658252177794285}

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
