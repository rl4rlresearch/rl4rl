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
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2578450007024290268, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 62448, "peak_hidden_elements": 200704, "recurrent_macs": 492794640, "recurrent_steps": 16300, "total_inference_macs": 494320320, "training_seconds": 112.68029429204762, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.3949216503307132}
prior_hypothesis: A 78-unit fast GRU over all 20 scheduled frames plus a 78-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 506,066,100 to approximately 494,320,320.

REFERENCE DESIGN 1
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2701699754037438588, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 65488, "peak_hidden_elements": 205824, "recurrent_macs": 516384000, "recurrent_steps": 16300, "total_inference_macs": 517948800, "training_seconds": 137.62998966616578, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.3985315240965299}
prior_hypothesis: Updating the qualified 80-unit slow GRU from four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 549,244,800 to approximately 517,948,800.

REFERENCE DESIGN 2
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2639717782753922359, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 63959, "peak_hidden_elements": 203264, "recurrent_macs": 504520860, "recurrent_steps": 16300, "total_inference_macs": 506066100, "training_seconds": 140.39403608301654, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.39557285425853145}
prior_hypothesis: A 79-unit fast GRU updated for all 20 frames plus a 79-unit slow GRU updated from four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 517,948,800 to approximately 506,066,100.

REFERENCE DESIGN 3
verified_results: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2458057042226678500, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 59480, "peak_hidden_elements": 195584, "recurrent_macs": 469752960, "recurrent_steps": 16300, "total_inference_macs": 471239520, "training_seconds": 113.94506791699678, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.3694320304262126}
prior_hypothesis: A 76-unit fast GRU over all 20 scheduled frames plus a 76-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 482,711,460 to approximately 471,239,520.

## Recent verification evidence

RECENT RESULT
hypothesis: A 110-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 712,502,340 to approximately 700,704,400.
change: Reduce the recurrent state, temporal summary, and both readout widths from 111 to 110 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 111-unit design achieved 86.75% accuracy at 712,502,340 MACs, leaving a 1.75-point margin after every tested 111–114-unit batch-64 GRU qualified; 110 units is therefore the most informative remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3654980771912270076, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 45376, "peak_hidden_elements": 113152, "recurrent_macs": 699270000, "recurrent_steps": 16300, "total_inference_macs": 700704400, "training_seconds": 77.22509833402, "validation_accuracy": 0.8576687116564418, "validation_cross_entropy": 0.4034261036504266}

RECENT RESULT
hypothesis: A 109-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 724,398,080 to approximately 689,004,260.
change: Reduce the recurrent state, temporal summary, and both readout widths from 112 to 109 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch recurrent-width boundary probe
evidence_used: The 110-unit batch-64 design qualified at 85.77% and 700,704,400 MACs; testing the adjacent 109-unit width is the most informative remaining cost-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3593951061367266797, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 44637, "peak_hidden_elements": 112128, "recurrent_macs": 687582900, "recurrent_steps": 16300, "total_inference_macs": 689004260, "training_seconds": 78.46343158301897, "validation_accuracy": 0.8699386503067484, "validation_cross_entropy": 0.3816709647149396}

RECENT RESULT
hypothesis: A 108-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 689,004,260 to approximately 677,401,920.
change: Reduce the recurrent state, temporal summary, and both readout widths from 113 to 108 units while preserving the qualified 20-frame schedule and batch-64 training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 109-unit design achieved 86.99% validation accuracy at 689,004,260 MACs, leaving a 1.99-point margin; testing 108 units is the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3533431490503609324, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 43904, "peak_hidden_elements": 111104, "recurrent_macs": 675993600, "recurrent_steps": 16300, "total_inference_macs": 677401920, "training_seconds": 68.05786091601476, "validation_accuracy": 0.8748466257668711, "validation_cross_entropy": 0.38222030920485045}

RECENT RESULT
hypothesis: A 107-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 677,401,920 to approximately 665,897,380.
change: Reduce the recurrent state, temporal summary, and both readout widths from 108 to 107 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 108-unit design achieved 87.48% validation accuracy at 677,401,920 MACs, leaving a 2.48-point margin after every tested 108–114-unit batch-64 GRU qualified; 107 units is the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3473422059321297657, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 43177, "peak_hidden_elements": 110080, "recurrent_macs": 664502100, "recurrent_steps": 16300, "total_inference_macs": 665897380, "training_seconds": 76.27699274988845, "validation_accuracy": 0.8638036809815951, "validation_cross_entropy": 0.3855161233913679}

RECENT RESULT
hypothesis: A 106-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 665,897,380 to approximately 654,490,640.
change: Reduce the recurrent state, temporal summary, and both readout widths from 109 to 106 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 107-unit design achieved 86.38% validation accuracy at 665,897,380 MACs, and every tested 107–114-unit batch-64 GRU qualified, making 106 units the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3413922767820331796, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 42456, "peak_hidden_elements": 109056, "recurrent_macs": 653108400, "recurrent_steps": 16300, "total_inference_macs": 654490640, "training_seconds": 67.40046083414927, "validation_accuracy": 0.8760736196319019, "validation_cross_entropy": 0.39349540289194307}

RECENT RESULT
hypothesis: A 105-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 654,490,640 to approximately 643,181,700.
change: Reduce the recurrent state, temporal summary, and both readout widths from 110 to 105 units while preserving the qualified 20-frame schedule and training procedure.
mechanism: Smaller-batch adjacent recurrent-width probe
evidence_used: The adjacent 106-unit design achieved 87.61% validation accuracy at 654,490,640 MACs, leaving a 2.61-point margin after every tested 106–114-unit batch-64 GRU qualified; 105 units is therefore the most informative remaining width-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 3354933616000711741, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 41741, "peak_hidden_elements": 108032, "recurrent_macs": 641812500, "recurrent_steps": 16300, "total_inference_macs": 643181700, "training_seconds": 64.90611437498592, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.389607791081528}

RECENT RESULT
hypothesis: An 80-unit fast GRU updated for all 20 scheduled frames plus an 80-unit slow GRU updated from four-frame summaries will retain at least 85% validation accuracy while reducing expected total inference MACs from 643,181,700 to approximately 549,244,800.
change: Replace the monolithic recurrent state with fast and slow GRUCell states, execute the slow transition every fourth frame, classify from fast-scale and slow-scale summaries, and use the proven batch-64 training regime.
mechanism: Clocked hierarchical dual-timescale GRU
evidence_used: The 105-unit monolithic GRU achieved 86.75% at 643,181,700 MACs, while the custom bottleneck-gated recurrence narrowly missed at 84.91%. This challenges the shared assumption that one full-width transition must process every frame while retaining standard, well-qualified GRU gating at both timescales.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2864944452068094588, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 65488, "peak_hidden_elements": 205824, "recurrent_macs": 547680000, "recurrent_steps": 16300, "total_inference_macs": 549244800, "training_seconds": 139.1944524589926, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.38399118060714627}

RECENT RESULT
hypothesis: Updating the qualified 80-unit slow GRU from four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 549,244,800 to approximately 517,948,800.
change: Change the slow recurrent clock from every fourth fast step to every fifth fast step and average each corresponding five-frame block.
mechanism: Five-frame clocked slow recurrence
evidence_used: The current dual-timescale model achieved 86.75% accuracy with five slow transitions, leaving a 1.75-point margin; reducing to four slow transitions directly tests how much slow temporal resolution is necessary while preserving both proven 80-unit GRUs and the 20-frame acoustic path.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2701699754037438588, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 65488, "peak_hidden_elements": 205824, "recurrent_macs": 516384000, "recurrent_steps": 16300, "total_inference_macs": 517948800, "training_seconds": 137.62998966616578, "validation_accuracy": 0.8588957055214724, "validation_cross_entropy": 0.3985315240965299}

RECENT RESULT
hypothesis: A 79-unit fast GRU updated for all 20 frames plus a 79-unit slow GRU updated from four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 517,948,800 to approximately 506,066,100.
change: Replace the 105-unit monolithic GRU with 79-unit fast and slow GRUCells, update the slow state every fifth step from the mean fast-state block, and classify from fast-mean, slow-mean, and slow-endpoint summaries.
mechanism: Adjacent-width clocked dual-timescale GRU
evidence_used: The qualified 80-unit dual-timescale design achieved 85.89% accuracy at 517,948,800 MACs; an adjacent one-unit reduction in both recurrent widths is the most direct probe of its remaining capacity boundary.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2639717782753922359, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 63959, "peak_hidden_elements": 203264, "recurrent_macs": 504520860, "recurrent_steps": 16300, "total_inference_macs": 506066100, "training_seconds": 140.39403608301654, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.39557285425853145}

RECENT RESULT
hypothesis: A 78-unit fast GRU over all 20 scheduled frames plus a 78-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 506,066,100 to approximately 494,320,320.
change: Replace the 106-unit monolithic GRU with 78-unit fast and slow GRUCells, update the slow state every fifth step from the mean fast-state block, and classify from fast-mean, slow-mean, and slow-endpoint summaries.
mechanism: Adjacent-width clocked dual-timescale GRU
evidence_used: The adjacent 79-unit dual-timescale design achieved 86.26% accuracy at 506,066,100 MACs, leaving a 1.26-point margin; reducing both recurrent widths by one unit is the most direct remaining capacity-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2578450007024290268, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 62448, "peak_hidden_elements": 200704, "recurrent_macs": 492794640, "recurrent_steps": 16300, "total_inference_macs": 494320320, "training_seconds": 112.68029429204762, "validation_accuracy": 0.8625766871165644, "validation_cross_entropy": 0.3949216503307132}

RECENT RESULT
hypothesis: A 77-unit fast GRU over all 20 scheduled frames plus a 77-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 494,320,320 to approximately 482,711,460.
change: Replace the 107-unit monolithic GRU with 77-unit fast and slow GRUCells, update the slow state every fifth step from the mean fast-state block, and classify from fast-mean, slow-mean, and slow-endpoint summaries.
mechanism: Adjacent-width clocked dual-timescale GRU
evidence_used: The adjacent 78-unit dual-timescale design achieved 86.26% validation accuracy at 494,320,320 MACs, leaving a 1.26-point margin; reducing both recurrent widths by one unit is the most direct remaining capacity-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2517896426848542315, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 60955, "peak_hidden_elements": 198144, "recurrent_macs": 481205340, "recurrent_steps": 16300, "total_inference_macs": 482711460, "training_seconds": 123.12135720788501, "validation_accuracy": 0.8674846625766871, "validation_cross_entropy": 0.38127958145609664}

RECENT RESULT
hypothesis: A 76-unit fast GRU over all 20 scheduled frames plus a 76-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 482,711,460 to approximately 471,239,520.
change: Reduce both recurrent widths from 77 to 76 units and resize the three-summary classifier input from 231 to 228 features, preserving the qualified schedule and training procedure.
mechanism: Adjacent-width clocked dual-timescale GRU
evidence_used: The adjacent 77-unit design achieved 86.75% validation accuracy at 482,711,460 MACs, leaving a 1.75-point margin after every tested 77–80-unit dual-timescale design qualified; 76 units is the most informative remaining capacity-boundary probe.
result: met the accuracy requirement and became an available design
reported_values: {"batch_size": 64, "examples_processed": 50000, "inference_cost": 2458057042226678500, "maximum_recurrent_steps": 20, "mean_recurrent_steps": 20.0, "median_recurrent_steps": 20, "optimizer_steps": 786, "p95_recurrent_steps": 20, "parameters": 59480, "peak_hidden_elements": 195584, "recurrent_macs": 469752960, "recurrent_steps": 16300, "total_inference_macs": 471239520, "training_seconds": 113.94506791699678, "validation_accuracy": 0.8773006134969326, "validation_cross_entropy": 0.3694320304262126}



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
