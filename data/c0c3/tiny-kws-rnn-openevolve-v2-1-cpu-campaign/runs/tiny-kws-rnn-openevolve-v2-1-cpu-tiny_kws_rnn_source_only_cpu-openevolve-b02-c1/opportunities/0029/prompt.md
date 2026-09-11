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
verified_results: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 54.18845779192634, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45900973454574867}
prior_hypothesis: An orthonormal seven-coordinate parameterization of the eight centered logits will retain at least 85% accuracy while matching the prior seven-logit MAC and parameter reductions.

## Recent verification evidence

RECENT RESULT
hypothesis: The verified 60+59 GRU will retain at least 85% validation accuracy when processing frames 3–31, while reducing recurrent execution from 30 to 29 steps and lowering total inference MACs below 694,740,230.
change: Skip the first three input frames instead of the first two, preserving the model architecture and training procedure.
mechanism: Third boundary-frame trimming
evidence_used: Processing frames 2–31 achieved 85.64% accuracy at 694,740,230 MACs; the preceding one-frame trim also remained feasible, making one additional boundary trim the most direct test of the minimum viable temporal budget.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3503209966281375417, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 30097, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671608085, "training_seconds": 93.30611424986273, "validation_accuracy": 0.8503067484662576, "validation_cross_entropy": 0.45579669226898006}

RECENT RESULT
hypothesis: Representing eight-class softmax probabilities with seven learned logits and one fixed-zero reference logit will retain at least 85% validation accuracy while reducing classifier MACs by 119 per example and learned parameters by 120.
change: Change the classifier from eight outputs to seven and append a fixed-zero eighth logit, preserving the full softmax function class and the verified 29-step recurrent architecture.
mechanism: Reference-class logit gauge fixing
evidence_used: The 60+59 GRU over frames 3–31 met the target at 85.03% accuracy; softmax is invariant to subtracting one class logit from every logit, so fixing that reference logit to zero removes redundant learned computation without reducing representational capacity.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 69.34918704093434, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.481317831402176}

RECENT RESULT
hypothesis: Two independent 59-unit GRUs over frames 3–31 will retain at least 85% validation accuracy while reducing total inference MACs from 671,608,085 to approximately 661,745,770.
change: Reduce the 60-unit GRU to 59 units, producing two balanced 59-unit blocks and resizing the recurrent state, temporal summary, and classifier input to 118 features.
mechanism: Balanced two-block recurrent width reduction
evidence_used: Reducing the successful 60+60 model to 60+59 improved observed accuracy from 85.28% to 86.50% while lowering cost; removing one unit from the larger remaining block is the smallest recurrent-capacity reduction and is slightly cheaper than an unequal 60+58 split of the same aggregate width.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3451766630581662271, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29666, "peak_hidden_elements": 121344, "recurrent_macs": 660976410, "recurrent_steps": 23635, "total_inference_macs": 661745770, "training_seconds": 77.23555104108527, "validation_accuracy": 0.8429447852760736, "validation_cross_entropy": 0.47705680051464244}

RECENT RESULT
hypothesis: Two independent 100-unit tanh RNNs over the verified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs below 570 million.
change: Replace the gated 60+59 recurrence with wider 100+100 ungated recurrent blocks, resizing the state, temporal summary, and classifier while preserving causal mean pooling and frames 3–31.
mechanism: Wider ungated block-diagonal tanh recurrence
evidence_used: Parallel recurrent blocks already achieved 86.50% accuracy before trimming; replacing each GRU’s three gated matrices with one RNN transition permits 200 aggregate hidden features at 24,000 recurrent MACs per step versus 28,383 currently. This challenges the load-bearing assumption that learned gates are necessary for these short normalized sequences.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2965612016587133683, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 26048, "peak_hidden_elements": 205312, "recurrent_macs": 567240000, "recurrent_steps": 23635, "total_inference_macs": 568544000, "training_seconds": 57.03650337504223, "validation_accuracy": 0.7177914110429447, "validation_cross_entropy": 0.8712244513576016}

RECENT RESULT
hypothesis: Balanced 61+60 GRUs processing frames 4–31 will retain at least 85% validation accuracy while reducing exact total inference MACs from 671,608,085 to approximately 667,657,780.
change: Remove one additional leading frame, widen the recurrent blocks from 60+59 to 61+60 units, and resize the state, summary, and classifier to 121 features.
mechanism: Reallocate one recurrent step’s MAC budget into balanced hidden width
evidence_used: The 60+59 model remained feasible through three consecutive frame trims, reaching 85.03% at 29 steps, while 59+59 failed; this suggests recurrent width is at its capacity boundary. At 28 steps, 61+60 is the widest near-balanced two-block configuration that still costs less than the current design.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482604574237516365, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30965, "peak_hidden_elements": 124416, "recurrent_macs": 666868860, "recurrent_steps": 22820, "total_inference_macs": 667657780, "training_seconds": 81.21272766706534, "validation_accuracy": 0.845398773006135, "validation_cross_entropy": 0.47602081298828125}

RECENT RESULT
hypothesis: Compressing each normalized 20-band frame to 16 learned features while preserving the verified 60+59 hidden capacity and 29-step schedule will retain at least 85% validation accuracy and reduce total inference MACs below 646 million.
change: Add a shared bias-free 20-to-16 linear projection and change both GRUs to consume its output, leaving recurrent width, pooling, classifier, training, and frame schedule unchanged.
mechanism: Learned low-rank mel input projection
evidence_used: The 60+59 model over frames 3–31 achieved 85.03%, while reducing hidden width to 59+59 failed at 84.29%; this motivates reducing input-side matrix rank without removing recurrent state capacity or additional temporal evidence.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3366611564273013929, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 28989, "peak_hidden_elements": 122368, "recurrent_macs": 644644625, "recurrent_steps": 23635, "total_inference_macs": 645420505, "training_seconds": 78.76660037506372, "validation_accuracy": 0.841717791411043, "validation_cross_entropy": 0.4927171110375527}

RECENT RESULT
hypothesis: Merging the two highest adjacent normalized mel bands into one feature will retain at least 85% accuracy while reducing total inference MACs from 671,608,085 to approximately 663,170,390.
change: Preserve the verified 60+59 hidden state and 29-frame schedule, but reduce each GRU’s input width from 20 to 19 by averaging the final two normalized mel bands.
mechanism: Adjacent high-mel band coalescing
evidence_used: The 20-to-16 projection reached 84.17%, while reducing recurrent width to 59+59 reached 84.29%; this suggests hidden capacity is more load-bearing and motivates a conservative one-feature input reduction that preserves information from every band.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3459197665273266165, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29740, "peak_hidden_elements": 122368, "recurrent_macs": 662394510, "recurrent_steps": 23635, "total_inference_macs": 663170390, "training_seconds": 49.59928729198873, "validation_accuracy": 0.8355828220858895, "validation_cross_entropy": 0.4746463892650019}

RECENT RESULT
hypothesis: An orthonormal seven-coordinate parameterization of the eight centered logits will retain at least 85% accuracy while matching the prior seven-logit MAC and parameter reductions.
change: Replace the eight-output classifier with seven learned contrast coordinates and expand them through a fixed orthonormal basis into eight zero-sum logits.
mechanism: Orthonormal softmax gauge fixing
evidence_used: The fixed-zero seven-logit model reached 84.42% despite preserving softmax expressivity, suggesting its asymmetric parameterization hurt optimization; balanced orthonormal contrasts preserve the same full function class with better-conditioned, nonconstant logits for every class.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3502704077764040712, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 670832205, "recurrent_steps": 23635, "total_inference_macs": 671511100, "training_seconds": 54.18845779192634, "validation_accuracy": 0.8539877300613496, "validation_cross_entropy": 0.45900973454574867}

RECENT RESULT
hypothesis: A 61+60 GRU processing frames 4–31 with the verified orthonormal seven-coordinate classifier will recover the 0.46-point accuracy shortfall of the prior 28-step model, reaching at least 85% while reducing total inference MACs below the current 671,511,100.
change: Skip one additional leading frame, widen the parallel GRUs from 60+59 to 61+60 units, resize the recurrent state and summary to 121 features, and retain the seven-coordinate orthonormal classifier.
mechanism: Step-to-width reallocation with orthonormal logit contrasts
evidence_used: The prior 61+60, 28-step model reached 84.54%, narrowly missing the threshold, while the orthonormal seven-coordinate classifier subsequently produced 85.40% on the verified 29-step architecture and reduced classifier MACs; this directly tests whether its balanced parameterization can close the small 28-step accuracy gap.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3482090183392159228, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 30843, "peak_hidden_elements": 124416, "recurrent_macs": 666868860, "recurrent_steps": 22820, "total_inference_macs": 667559165, "training_seconds": 54.79871074995026, "validation_accuracy": 0.8343558282208589, "validation_cross_entropy": 0.4892535437835506}

RECENT RESULT
hypothesis: A 60+58 GRU with the verified orthonormal seven-coordinate classifier will retain at least 85% validation accuracy while reducing total inference MACs from 671,511,100 to approximately 661,791,410.
change: Remove one hidden unit from the smaller GRU branch and resize its state, temporal summary, and classifier input from 119 to 118 features.
mechanism: Preserve-dominant-branch asymmetric width reduction
evidence_used: The 60+59 orthonormal model achieved 85.40% accuracy, whereas reducing the larger branch to form a symmetric 59+59 model failed at 84.29%; preserving the 60-unit branch tests whether the successful asymmetry is load-bearing while making the smallest remaining recurrent reduction.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3452004695766290198, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 29553, "peak_hidden_elements": 121344, "recurrent_macs": 661118220, "recurrent_steps": 23635, "total_inference_macs": 661791410, "training_seconds": 72.39296129206195, "validation_accuracy": 0.8441717791411043, "validation_cross_entropy": 0.46837381000167755}

RECENT RESULT
hypothesis: Processing frames 3–30 will retain at least 85% validation accuracy while reducing recurrent execution from 29 to 28 steps and total inference MACs from 671,511,100 to approximately 648,378,955.
change: Preserve the verified 60+59 GRUs and orthonormal seven-coordinate classifier, but omit the final input frame.
mechanism: Opposite-boundary frame trimming
evidence_used: The current frames 3–31 design achieved 85.40% accuracy. A 28-step design that instead removed frame 3 failed, indicating that testing the opposite boundary is the cleanest way to distinguish temporal-position importance from a fundamental 29-step requirement.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 3382043289552224552, "maximum_recurrent_steps": 28, "mean_recurrent_steps": 28.0, "median_recurrent_steps": 28, "optimizer_steps": 397, "p95_recurrent_steps": 28, "parameters": 29977, "peak_hidden_elements": 122368, "recurrent_macs": 647700060, "recurrent_steps": 22820, "total_inference_macs": 648378955, "training_seconds": 77.13366170902736, "validation_accuracy": 0.8466257668711656, "validation_cross_entropy": 0.4665845344403039}

RECENT RESULT
hypothesis: Replacing each three-gate GRU with a width-matched coupled forget cell will retain at least 85% validation accuracy while reducing recurrent MACs by exactly one third, from 670,832,205 to 447,221,470, with the same 29 recurrent steps.
change: Preserve the verified 60+59 hidden widths, causal mean summary, frame schedule, and orthonormal classifier, but replace each GRU with a two-output-per-hidden-unit `nn.Linear` recurrent cell.
mechanism: Coupled single-gate recurrent cells
evidence_used: The 100+100 ungated tanh RNN fell to 71.78% while the 60+59 GRU reached 85.40%, indicating gating is load-bearing. A coupled forget cell retains learned elementwise gating and recurrent dependence while removing one third of the GRU’s dense gate computation.
result: did not meet the accuracy requirement
reported_values: {"batch_size": 128, "examples_processed": 50000, "inference_cost": 2336316459170982440, "maximum_recurrent_steps": 29, "mean_recurrent_steps": 29.0, "median_recurrent_steps": 29, "optimizer_steps": 397, "p95_recurrent_steps": 29, "parameters": 20040, "peak_hidden_elements": 122368, "recurrent_macs": 447221470, "recurrent_steps": 23635, "total_inference_macs": 447900365, "training_seconds": 105.49562495900318, "validation_accuracy": 0.8478527607361963, "validation_cross_entropy": 0.46414298777199964}



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
