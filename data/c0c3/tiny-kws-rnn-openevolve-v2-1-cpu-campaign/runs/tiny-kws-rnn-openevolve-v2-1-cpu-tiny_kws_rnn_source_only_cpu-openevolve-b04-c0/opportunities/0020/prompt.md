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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2915448281499296753, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23673, "peak_hidden_elements": 115712, "recurrent_macs": 557460000, "recurrent_steps": 26080, "total_inference_macs": 558927000, "training_seconds": 85.2473029170651, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4085738012395753}
prior_hypothesis: A 75-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.3% versus the verified 76-unit model.

## Recent verification evidence

RECENT RESULT
hypothesis: A 92-unit GRU processing all 32 frames with the dual mean/final-state readout will retain at least 85% accuracy while reducing exact MACs versus the verified 96-unit, 31-step model.
change: Reduce recurrent width from 96 to 92 and restore frame 0, trading one additional step for a structurally cheaper recurrent matrix at every step.
mechanism: Width-for-temporal-coverage rebalancing
evidence_used: The 104-unit dual-readout model reached 87.73%, showing the readout benefits from near-full temporal coverage; the 96-unit model still met 85.03%, while prior single-frame pruning caused a measurable accuracy loss. Since 92 units over 32 steps require fewer recurrent MACs than 96 units over 31 steps, restoring the omitted frame may offset the smaller state without sacrificing the primary cost objective.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 4211441137302266104, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 32984, "peak_hidden_elements": 94720, "recurrent_macs": 806184960, "recurrent_steps": 26080, "total_inference_macs": 807384640, "training_seconds": 90.52424379205331, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.3870472633034174}

RECENT RESULT
hypothesis: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing exact dense inference MACs relative to the verified 92-unit model.
change: Reduce the recurrent hidden state and temporal summary from 92 to 88 units, and reduce the dual-readout classifier input from 184 to 176 features.
mechanism: Conservative dual-readout width reduction
evidence_used: The 92-unit, 32-step dual-readout model achieved 87.12% accuracy, 2.12 points above the requirement, while already outperforming the 96-unit, 31-step model in both accuracy and MACs; this supports testing a modest structural width reduction without discarding temporal information.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3884679666744233856, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 30496, "peak_hidden_elements": 90624, "recurrent_macs": 743592960, "recurrent_steps": 26080, "total_inference_macs": 744740480, "training_seconds": 95.61552495881915, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.42309365301775786}

RECENT RESULT
hypothesis: An 86-unit GRU processing all 32 frames with the dual mean/final-state readout will retain at least 85% validation accuracy while reducing recurrent MACs by about 4.1% versus the verified 88-unit model.
change: Reduce the GRU hidden state and temporal summary from 88 to 86 units, and reduce the classifier input from 176 to 172 features.
mechanism: Fine-grained recurrent-width reduction
evidence_used: The 88-unit full-sequence model achieved 85.77% accuracy, leaving a 0.77-point margin; a two-unit reduction is the smallest practical structural step and preserves the temporal coverage whose reduction previously hurt accuracy.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3726196272406137448, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29288, "peak_hidden_elements": 88576, "recurrent_macs": 713235840, "recurrent_steps": 26080, "total_inference_macs": 714357280, "training_seconds": 144.33794829179533, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4322061175949003}

RECENT RESULT
hypothesis: An 86-unit GRU augmented with running temporal maxima will recover the small accuracy deficit and reach at least 85%, while retaining lower exact dense inference MACs than the verified 88-unit model.
change: Reduce the GRU to 86 units and classify from the temporal mean, final state, and running maximum over all 32 frames.
mechanism: Low-cost temporal max pooling
evidence_used: The 86-unit dual-readout model missed the threshold by only 0.34 points at 84.66%, while richer temporal readout previously raised the 104-unit model from 84.66% to 87.73%; max pooling adds complementary temporal evidence with only a small classifier-cost increase.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3729121073245854056, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 29976, "peak_hidden_elements": 132608, "recurrent_macs": 713235840, "recurrent_steps": 26080, "total_inference_macs": 714918000, "training_seconds": 197.66547083295882, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3962126562200441}

RECENT RESULT
hypothesis: An 84-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact recurrent MACs by about 3.8% versus the verified 86-unit model.
change: Reduce the GRU and all recurrent summary widths from 86 to 84 units, and reduce the classifier input from 258 to 252 features.
mechanism: Temporal-max-assisted recurrent width reduction
evidence_used: Adding temporal maximum pooling raised the 86-unit model from 84.66% to 86.87%, creating a 1.87-point margin; this supports a conservative two-unit structural reduction while preserving all 32 frames and the successful readout.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3573834554244191336, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 28776, "peak_hidden_elements": 129536, "recurrent_macs": 683504640, "recurrent_steps": 26080, "total_inference_macs": 685147680, "training_seconds": 121.60275137517601, "validation_accuracy": 0.8711656441717791, "validation_cross_entropy": 0.3870809168903374}

RECENT RESULT
hypothesis: An 82-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs relative to the verified 84-unit model.
change: Reduce the GRU and recurrent summaries from 84 to 82 units and the classifier input from 252 to 246 features, preserving all 32 frames and the proven training procedure.
mechanism: Temporal-max-assisted recurrent width reduction
evidence_used: The 84-unit triple-readout model achieved 87.12% accuracy, 2.12 points above the requirement, after the same readout enabled both 86- and 84-unit models to qualify; this supports another conservative two-unit structural reduction.
result: training did not finish within the verification time limit
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3421812929203141760, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 27600, "peak_hidden_elements": 126464, "recurrent_macs": 654399360, "recurrent_steps": 26080, "total_inference_macs": 656003280, "training_seconds": 206.8398099169135, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.40601107357469807}

RECENT RESULT
hypothesis: An 80-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by about 8.4% versus the verified 84-unit model.
change: Reduce the GRU and recurrent summaries from 84 to 80 units and the classifier input from 252 to 240 features, preserving all 32 causal frames and the established training procedure.
mechanism: Hardware-friendly temporal-max-assisted width reduction
evidence_used: The 82-unit model reached 86.75% accuracy despite exceeding the training-time limit, while the verified 84-unit model reached 87.12%; this indicates the triple readout retains sufficient accuracy at smaller widths and motivates testing the aligned 80-unit width.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3273056198122705328, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 26448, "peak_hidden_elements": 123392, "recurrent_macs": 625920000, "recurrent_steps": 26080, "total_inference_macs": 627484800, "training_seconds": 109.0368723331485, "validation_accuracy": 0.8564417177914111, "validation_cross_entropy": 0.4118331160282065}

RECENT RESULT
hypothesis: A 79-unit GRU with the proven mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by about 2.2% versus the verified 80-unit model.
change: Reduce the GRU and recurrent summary widths from 80 to 79 units and the classifier input from 240 to 237 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 80-unit model achieved 85.64% accuracy, while the 82-unit model achieved 86.75%; a one-unit reduction is the smallest structural probe of the accuracy-cost boundary after larger two-unit reductions showed measurable accuracy sensitivity.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3199902167817717041, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25881, "peak_hidden_elements": 121856, "recurrent_macs": 611915040, "recurrent_steps": 26080, "total_inference_macs": 613460280, "training_seconds": 102.87217433284968, "validation_accuracy": 0.8723926380368098, "validation_cross_entropy": 0.3965815725502061}

RECENT RESULT
hypothesis: A 78-unit GRU with the proven mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by roughly 2.3% versus the verified 79-unit model.
change: Reduce the GRU and recurrent summaries from 79 to 78 units and the classifier input from 237 to 234 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 79-unit model achieved 87.24% accuracy, 2.24 points above the requirement, and outperformed the qualifying 80-unit model; this supports another minimal one-unit structural reduction at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3127564361002882040, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 25320, "peak_hidden_elements": 120320, "recurrent_macs": 598066560, "recurrent_steps": 26080, "total_inference_macs": 599592240, "training_seconds": 85.23865499999374, "validation_accuracy": 0.8613496932515338, "validation_cross_entropy": 0.4022246752779908}

RECENT RESULT
hypothesis: A 77-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.3% versus the verified 78-unit model.
change: Reduce the GRU and recurrent summary widths from 78 to 77 units and the classifier input from 234 to 231 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 78-unit model achieved 86.13% accuracy, 1.13 points above the requirement, after both 79- and 78-unit one-unit reductions qualified; this motivates one more minimal structural probe of the accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3056042777678200325, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24765, "peak_hidden_elements": 118784, "recurrent_macs": 584374560, "recurrent_steps": 26080, "total_inference_macs": 585880680, "training_seconds": 83.19529145793058, "validation_accuracy": 0.8687116564417178, "validation_cross_entropy": 0.3903258902894939}

RECENT RESULT
hypothesis: A 76-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.3% versus the verified 77-unit model.
change: Reduce the GRU and recurrent summaries from 77 to 76 units and the classifier input from 231 to 228 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 77-unit model achieved 86.87% accuracy, 1.87 points above the requirement, after the 78- and 79-unit models also qualified; this supports another minimal structural reduction at the current accuracy-cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2985337417843671896, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 24216, "peak_hidden_elements": 117248, "recurrent_macs": 570839040, "recurrent_steps": 26080, "total_inference_macs": 572325600, "training_seconds": 82.21133245783858, "validation_accuracy": 0.8736196319018404, "validation_cross_entropy": 0.37948165612718077}

RECENT RESULT
hypothesis: A 75-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.3% versus the verified 76-unit model.
change: Reduce the GRU and recurrent summary widths from 76 to 75 units and the classifier input from 228 to 225 features, preserving all 32 frames and the established training procedure.
mechanism: One-unit recurrent-width frontier search
evidence_used: The 76-unit model achieved 87.36% validation accuracy, 2.36 points above the requirement, after the 77-, 78-, and 79-unit models also qualified; this supports another minimal structural reduction at the current cost frontier.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2915448281499296753, "maximum_recurrent_steps": 32, "mean_recurrent_steps": 32.0, "median_recurrent_steps": 32, "optimizer_steps": 397, "p95_recurrent_steps": 32, "parameters": 23673, "peak_hidden_elements": 115712, "recurrent_macs": 557460000, "recurrent_steps": 26080, "total_inference_macs": 558927000, "training_seconds": 85.2473029170651, "validation_accuracy": 0.8650306748466258, "validation_cross_entropy": 0.4085738012395753}



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
