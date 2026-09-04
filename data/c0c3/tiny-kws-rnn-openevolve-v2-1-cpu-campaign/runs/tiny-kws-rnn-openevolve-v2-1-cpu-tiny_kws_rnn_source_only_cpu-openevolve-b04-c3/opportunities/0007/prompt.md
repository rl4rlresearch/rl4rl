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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 7735622093287437072, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1482178560, "recurrent_steps": 26080, "total_inference_macs": 1483013120, "training_seconds": 45.78908033296466, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.4032038870033311}
prior_hypothesis: starting design

REFERENCE DESIGN 1
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 107.587929709116, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4402542488706624}
prior_hypothesis: Using 28 uniformly spaced frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 12.5% versus the qualified 32-frame design.

REFERENCE DESIGN 2
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6670977582648793233, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1278079740, "recurrent_steps": 22820, "total_inference_macs": 1278907780, "training_seconds": 87.6996964160353, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4510241268602617}
prior_hypothesis: A 127-unit GRU using the qualified 28-frame schedule will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.45% versus the qualified 128-unit design.

## Recent verification evidence

RECENT RESULT
hypothesis: Processing 24 uniformly spaced frames instead of all 32 will preserve validation accuracy at or above 85% while reducing recurrent MACs and executed recurrent steps by 25%.
change: Replace the full-frame schedule with 24 unique, uniformly distributed causal indices spanning the complete recording.
mechanism: Uniform causal frame thinning
evidence_used: The 128-unit GRU reaches 87.24% accuracy using all 32 frames, providing a 2.24-point margin above the target; temporal redundancy is therefore the lowest-risk first target for structural cost reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5802804867952463512, "maximum_recurrent_steps": 24, "mean_recurrent_steps": 24.0, "median_recurrent_steps": 24, "optimizer_steps": 397, "p95_recurrent_steps": 24, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1111633920, "recurrent_steps": 19560, "total_inference_macs": 1112468480, "training_seconds": 95.58400704199448, "validation_accuracy": 0.8319018404907975, "validation_cross_entropy": 0.4934807151373179}

RECENT RESULT
hypothesis: Using 28 uniformly spaced frames will recover validation accuracy to at least 85% while reducing recurrent MACs and steps by 12.5% versus the qualified 32-frame design.
change: Replace the full 32-frame schedule with 28 unique, uniformly distributed indices spanning the complete recording.
mechanism: Moderate uniform causal frame thinning
evidence_used: The 32-frame model achieved 87.24% accuracy, while aggressive thinning to 24 frames achieved 83.19%; testing the midpoint of 28 frames is the most informative next cost-accuracy tradeoff.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 107.587929709116, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4402542488706624}

RECENT RESULT
hypothesis: A 124-unit GRU processing 28 uniformly spaced frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 5.7% relative to the qualified 128-unit, 28-frame design.
change: Apply the qualified 28-frame schedule and reduce the GRU state, temporal summary, and classifier width from 128 to 124.
mechanism: Mild recurrent-width reduction on the qualified 28-frame schedule
evidence_used: The 128-unit model achieved 85.52% accuracy at 28 frames, while 24 frames fell to 83.19%; preserving the qualified temporal coverage while making a modest 3.1% width reduction targets lower MACs without crossing the observed frame-resolution boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6380555062058626812, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 55352, "peak_hidden_elements": 127488, "recurrent_macs": 1222421760, "recurrent_steps": 22820, "total_inference_macs": 1223230240, "training_seconds": 47.113891292130575, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.46318858328041124}

RECENT RESULT
hypothesis: A 126-unit GRU using 28 uniformly spaced frames will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 2.9% versus the qualified 128-unit, 28-frame design.
change: Apply the qualified 28-frame schedule and reduce the GRU state, temporal summary, and classifier width from 128 to 126.
mechanism: Midpoint recurrent-width reduction on the qualified 28-frame schedule
evidence_used: At 28 frames, 128 units achieved 85.52% accuracy while 124 units achieved 84.66%; testing the midpoint width of 126 is the most informative next search within the observed qualification boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6573455880231520300, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 57000, "peak_hidden_elements": 129536, "recurrent_macs": 1259390160, "recurrent_steps": 22820, "total_inference_macs": 1260211680, "training_seconds": 93.50150845898315, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.44977021012569496}

RECENT RESULT
hypothesis: A 128-unit GRU using the qualified 28-frame schedule can retain at least 85% accuracy while a 0.99-confidence exit during the final four steps reduces exact validation MACs below the fixed 28-step design.
change: Restore the qualified 28-frame uniform schedule and allow highly confident examples to stop during its final four recurrent steps.
mechanism: Late confidence-gated early exit
evidence_used: The fixed 28-step, 128-unit design achieved 85.52% accuracy, while reducing recurrent width failed; confidence-gated execution targets dynamic step savings without removing recurrent capacity or temporal coverage from uncertain examples.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 127-unit GRU using the qualified 28-frame schedule will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.45% versus the qualified 128-unit design.
change: Use 28 uniformly spaced causal frames and reduce the GRU state, temporal summary, and classifier width from 128 to 127.
mechanism: One-unit recurrent-width boundary search
evidence_used: At 28 frames, 128 units achieved 85.52% accuracy while 126 units achieved 84.66%; testing the intervening 127-unit width directly resolves the narrowest observed qualification boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6670977582648793233, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 57833, "peak_hidden_elements": 130560, "recurrent_macs": 1278079740, "recurrent_steps": 22820, "total_inference_macs": 1278907780, "training_seconds": 87.6996964160353, "validation_accuracy": 0.852760736196319, "validation_cross_entropy": 0.4510241268602617}



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
