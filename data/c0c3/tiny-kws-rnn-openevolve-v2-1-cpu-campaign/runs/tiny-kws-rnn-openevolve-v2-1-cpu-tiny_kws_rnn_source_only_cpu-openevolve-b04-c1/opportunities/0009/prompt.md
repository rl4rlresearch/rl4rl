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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 89.91233604191802, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4015452285485765}
prior_hypothesis: A 96-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 7% versus the verified 100-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: Sampling 16 evenly spaced frames across the full recording will retain at least 85% validation accuracy while halving recurrent steps and recurrent MACs from the verified 32-step baseline.
change: Change the frame schedule from all 32 frames to 16 uniformly distributed frames, including the final frame.
mechanism: Uniform causal frame decimation
evidence_used: The current 128-unit GRU reaches 87.24% accuracy but executes 32 steps for every example, with recurrent computation accounting for 1,482,178,560 of 1,483,013,120 total MACs; temporal decimation directly targets nearly all inference cost while preserving full-recording coverage.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3869987642617489952, "maximum_recurrent_steps": 16, "mean_recurrent_steps": 16.0, "median_recurrent_steps": 16, "optimizer_steps": 397, "p95_recurrent_steps": 16, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 741089280, "recurrent_steps": 13040, "total_inference_macs": 741923840, "training_seconds": 67.05880595883355, "validation_accuracy": 0.8171779141104294, "validation_cross_entropy": 0.5212684678153757}

RECENT RESULT
hypothesis: Sampling 26 uniformly distributed frames across the full recording will preserve at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by 18.75% versus the verified 32-step baseline.
change: Replace the full 32-frame schedule with 26 evenly spaced, unique causal frame indices that include both endpoints.
mechanism: Moderate uniform causal frame decimation
evidence_used: The 32-step model achieved 87.24% accuracy, while aggressive reduction to 16 steps achieved only 81.72%; 26 steps is a conservative interpolation that retains substantially more temporal detail while still targeting the recurrent computation responsible for virtually all inference MACs.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6286009174286206902, "maximum_recurrent_steps": 26, "mean_recurrent_steps": 26.0, "median_recurrent_steps": 26, "optimizer_steps": 397, "p95_recurrent_steps": 26, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1204270080, "recurrent_steps": 21190, "total_inference_macs": 1205104640, "training_seconds": 103.90775570902042, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4891924150150978}

RECENT RESULT
hypothesis: Using 28 uniformly spaced frames will exceed 85% validation accuracy while reducing recurrent steps and MACs by 12.5% versus the verified 32-step baseline.
change: Select 28 unique, evenly distributed causal frames spanning the complete recording.
mechanism: Conservative uniform causal frame decimation
evidence_used: The 26-step model narrowly missed the target at 84.66%, while the 32-step baseline reached 87.24%; 28 steps conservatively explores the boundary while retaining a meaningful compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6769213480619950292, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1296906240, "recurrent_steps": 22820, "total_inference_macs": 1297740800, "training_seconds": 109.49587816605344, "validation_accuracy": 0.8552147239263803, "validation_cross_entropy": 0.4402542488706624}

RECENT RESULT
hypothesis: Sampling 27 uniformly distributed frames will retain at least 85% validation accuracy while reducing recurrent steps and recurrent MACs by approximately 3.6% versus the verified 28-step model.
change: Reduce the evenly spaced full-recording frame schedule from 28 steps to 27 steps.
mechanism: Boundary-search uniform causal frame decimation
evidence_used: The 26-step model narrowly missed the target at 84.66%, while 28 steps achieved 85.52%; testing 27 steps directly resolves the remaining compute–accuracy boundary.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6527611327453078597, "maximum_recurrent_steps": 27, "mean_recurrent_steps": 27.0, "median_recurrent_steps": 27, "optimizer_steps": 397, "p95_recurrent_steps": 27, "parameters": 58672, "peak_hidden_elements": 131584, "recurrent_macs": 1250588160, "recurrent_steps": 22005, "total_inference_macs": 1251422720, "training_seconds": 114.3457491248846, "validation_accuracy": 0.8404907975460123, "validation_cross_entropy": 0.46580634555933664}

RECENT RESULT
hypothesis: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.8% versus the verified 128-unit, 28-step model.
change: Reduce the GRU and classifier width from 128 to 112 units, resize recurrent state tensors accordingly, and restore the complete 32-frame causal schedule.
mechanism: Width-for-temporal-resolution tradeoff
evidence_used: The 128-unit model achieved 87.24% with 32 frames, while reducing temporal coverage to 27 frames fell to 84.05%; restoring all frames provides accuracy headroom while structural width reduction lowers dense MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 6037333084775166448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 45968, "peak_hidden_elements": 115200, "recurrent_macs": 1156700160, "recurrent_steps": 26080, "total_inference_macs": 1157430400, "training_seconds": 153.1809397500474, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.4392715641325968}

RECENT RESULT
hypothesis: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.7% versus the verified 112-unit model.
change: Reduce the GRU hidden state and classifier width from 112 to 104 units while preserving the full 32-frame schedule and training procedure.
mechanism: Recurrent width reduction with full temporal resolution
evidence_used: The 112-unit, 32-step model achieved 86.26% accuracy with 1,157,430,400 total MACs, leaving 1.26 percentage points of accuracy headroom; prior frame-decimation results show temporal resolution is valuable, so reducing width while retaining every frame is the most grounded next compute reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 5266546035573746592, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 40192, "peak_hidden_elements": 107008, "recurrent_macs": 1008983040, "recurrent_steps": 26080, "total_inference_macs": 1009661120, "training_seconds": 83.05264195799828, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.44775579721649733}

RECENT RESULT
hypothesis: A 100-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the verified 104-unit model.
change: Reduce recurrent width from 104 to 100, preserve all 32 frames, and classify from both the sequence-average output and final hidden output.
mechanism: Width reduction with dual-timescale recurrent readout
evidence_used: The 104-unit full-resolution GRU achieved 85.03% accuracy; retaining every frame while enriching its temporal readout may compensate for a modest width reduction, which structurally lowers recurrent MACs.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4904142805945688328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 38248, "peak_hidden_elements": 102912, "recurrent_macs": 938880000, "recurrent_steps": 26080, "total_inference_macs": 940184000, "training_seconds": 110.3439824578818, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.40625231222141006}

RECENT RESULT
hypothesis: A 96-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing dense inference MACs by approximately 7% versus the verified 100-unit model.
change: Reduce the GRU hidden width from 100 to 96 and resize the classifier and recurrent state tensors accordingly, preserving all 32 frames and the successful dual-timescale readout.
mechanism: Recurrent width reduction with dual-timescale readout
evidence_used: The 100-unit dual-readout model achieved 85.64% accuracy with 940,184,000 MACs, outperforming the wider 104-unit model’s 85.03%; this suggests the enriched readout provides enough accuracy headroom to test another modest structural width reduction.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4551262183702750928, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 35568, "peak_hidden_elements": 98816, "recurrent_macs": 871280640, "recurrent_steps": 26080, "total_inference_macs": 872532480, "training_seconds": 89.91233604191802, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.4015452285485765}



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
