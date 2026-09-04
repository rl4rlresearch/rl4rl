# Optimize a transformer for 10-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two 10-digit numbers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% accuracy under the fixed verification process. A
smaller implementation is useful only when it meets that accuracy requirement.
Every submitted implementation is trained from a fresh initialization.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must:

- have nonzero trainable parameters;
- contain and use at least one learned causal self-attention module;
- map token inputs to token logits through the learned model;
- train from a fresh initialization during verification;
- write both `checkpoints/best.pt` and a positive-step `checkpoints/last.pt`;
- keep source code unchanged while training; and
- use the protected generic decoding interface exactly as supplied.

Do not implement or embed decimal arithmetic, carry propagation, place-value
rules, digit lookup tables, finite-state addition transitions, fixed answer
rules, or input-dependent Python logic that directly computes the sum. Do not
hide such a solver in model generation, token processing, training, or saved
weights. Do not add dummy or zero-length parameters to disguise a fixed
algorithm as a learned model.

Do not modify protected files. Do not perform post-training state-dictionary
surgery, substitute a different saved model, truncate weights after training,
or report a parameter count that differs from the submitted model.

## Work boundaries

Minimize parameters. Required result: accuracy >= 0.99.
Editable source files: src/model.py, src/train.py.
Results reported after each verification: accuracy, parameters, training_steps.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or verification yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9989, "parameters": 1637, "training_steps": 4999}
prior_hypothesis: Omitting seven redundant key-bias coordinates in a 4:3 distribution will reduce the model to 1,637 learned parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9942, "parameters": 1639, "training_steps": 4999}
prior_hypothesis: Omitting five redundant key-bias coordinates distributed 3:2 across the two attention heads will produce a 1,639-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9992, "parameters": 1636, "training_steps": 4999}
prior_hypothesis: Retaining the qualified 4:3 key-bias layout while omitting one redundant value-bias coordinate will produce a 1,636-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Omitting three key-bias coordinates after preserving the baseline initialization sequence will reduce the model from 1,644 to 1,641 parameters while retaining at least 99% accuracy.
change: Add a compact QKV projection that reconstructs three omitted key-bias coordinates as zeros, installed only after full baseline initialization.
mechanism: Incremental post-initialization key-bias gauge fixing
evidence_used: Post-initialization omission of two key-bias coordinates achieved 99.98% accuracy at 1,642 parameters, while omission of all eight failed; removing one additional coordinate is the smallest informative next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7112999999999999, "parameters": 1641, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting three redundant key-bias coordinates distributed 2:1 across the two attention heads will produce a 1,641-parameter model with at least 99% accuracy.
change: Preserve the baseline initialization sequence, then replace QKV with a compact projection omitting the first two key-bias coordinates of head 0 and the first key-bias coordinate of head 1.
mechanism: Head-balanced post-initialization key-bias gauge fixing
evidence_used: Omitting two key-bias coordinates achieved 99.98% at 1,642 parameters, while omitting three consecutive coordinates failed at 71.13%; distributing the third omission across heads tests whether the failure arose from concentrating all three omissions in one head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1641, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting four redundant key-bias coordinates evenly distributed 2:2 across the two attention heads will produce a 1,640-parameter model with at least 99% accuracy.
change: Extend the successful head-balanced compact QKV projection by omitting a second key-bias coordinate from head 1 while preserving the complete baseline initialization sequence.
mechanism: Symmetric per-head key-bias gauge fixing
evidence_used: The 2:1 head-balanced omission achieved 99.93% at 1,641 parameters, whereas three consecutive omissions achieved only 71.13%; this motivates testing the smallest further reduction while maintaining equal omissions across heads.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9971, "parameters": 1640, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting five redundant key-bias coordinates distributed 3:2 across the two attention heads will produce a 1,639-parameter model with at least 99% accuracy.
change: Extend post-initialization QKV compaction to omit three key-bias coordinates from head 0 and two from head 1 while preserving all baseline initialization draws.
mechanism: Near-balanced per-head key-bias gauge fixing
evidence_used: The balanced 2:2 omission achieved 99.71% at 1,640 parameters, while 3:0 failed and 2:1 succeeded; this indicates per-head balance matters and motivates the smallest further reduction, 3:2.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9942, "parameters": 1639, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting six redundant key-bias coordinates evenly distributed 3:3 across the two heads will reduce the model to 1,638 learned parameters while retaining at least 99% accuracy.
change: Generalize post-initialization QKV compaction to reconstruct three zero key-bias coordinates in each head while preserving the baseline initialization sequence and every QKV weight.
mechanism: Symmetric three-coordinate key-bias gauge fixing per attention head
evidence_used: The near-balanced 3:2 omission achieved 99.42% at 1,639 parameters, while balanced omissions consistently outperformed concentrated ones; extending it to a symmetric 3:3 layout is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1638, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting seven redundant key-bias coordinates in a 4:3 distribution will reduce the model to 1,637 learned parameters while retaining at least 99% accuracy.
change: Extend post-initialization QKV compaction to omit all four key-bias coordinates from head 0 and three from head 1, preserving baseline initialization draws and every QKV weight.
mechanism: Near-balanced seven-coordinate key-bias gauge fixing
evidence_used: Symmetric 3:3 omission achieved 99.98% at 1,638 parameters, while complete 4:4 omission failed; 4:3 is the smallest informative step between those results.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9989, "parameters": 1637, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the qualified 4:3 key-bias layout while fixing one redundant positional-embedding coordinate will produce a 1,636-parameter model with at least 99% accuracy.
change: Upgrade QKV compaction from 3:2 to the qualified 4:3 layout, then preserve the initialized positional embedding exactly while making its first coordinate non-trainable; per-position uniform hidden-state shifts are removed by every pre-LayerNorm path and the final LayerNorm.
mechanism: Post-initialization positional row-shift gauge fixing
evidence_used: The 4:3 QKV design achieved 99.89% at 1,637 parameters, while 4:4 failed at 30.16%; retaining the remaining key-bias coordinate and removing an independent one-dimensional positional gauge is the smallest informative reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1399, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the qualified 4:3 key-bias layout while removing one scalar-shift degree of freedom from the MLP output bias will produce a 1,636-parameter model with at least 99% accuracy.
change: Upgrade QKV compaction from 2:2 to the qualified 4:3 layout, then replace the initialized MLP output projection with an equivalent seven-parameter bias whose omitted coordinate represents the uniform shift eliminated by the final LayerNorm.
mechanism: Final residual-bias shift gauge fixing
evidence_used: The 4:3 QKV design achieved 99.89% at 1,637 parameters. The positional-gauge attempt at 1,636 failed, motivating a different one-parameter reduction that acts directly on the final residual stream immediately before LayerNorm.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3924, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the final redundant key-bias coordinate to an existing learned query-bias scalar will reduce the model to 1,636 parameters while retaining at least 99% accuracy.
change: Remove the remaining independent key-bias parameter from the qualified 4:3 design and reconstruct it from the first learned query-bias coordinate.
mechanism: Parameter-free adaptive key-bias gauge coupling
evidence_used: The independent 4:3 key-bias layout achieved 99.89% at 1,637 parameters, whereas fixing all eight key-bias coordinates to zero collapsed to 30.16%; coupling the last coordinate preserves an adaptive numerical offset without adding a parameter.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5595, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the qualified 4:3 key-bias layout with one fixed `ln2` beta coordinate will produce a 1,636-parameter model with at least 99% accuracy because the unrestricted MLP input bias can absorb that coordinate’s effect.
change: Upgrade QKV compaction from 3:2 to 4:3 and, after baseline initialization, replace `ln2` with an equivalent LayerNorm retaining seven learned bias coordinates and reconstructing the omitted coordinate as zero.
mechanism: Downstream-absorbable LayerNorm beta gauge fixing
evidence_used: The 4:3 QKV design achieved 99.89% at 1,637 parameters. Since positional and final residual-bias gauges failed at 1,636, this tests a distinct affine redundancy directly absorbable by the fully learned `fc1` bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7001999999999999, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining the qualified 4:3 key-bias layout while omitting one redundant value-bias coordinate will produce a 1,636-parameter model with at least 99% accuracy.
change: Upgrade QKV compaction from 2:2 to 4:3 key-bias omission and reconstruct the first value-bias coordinate as zero; its position-independent effect remains representable by the learned attention output-projection bias.
mechanism: Value-to-output bias reparameterization
evidence_used: The 4:3 key-bias design achieved 99.89% at 1,637 parameters. Prior 1,636 attempts modified other gauges unsuccessfully, motivating the smallest reduction along the distinct exact redundancy between value bias and output-projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting a second redundant value-bias coordinate while retaining the qualified 4:3 key-bias layout will reduce the model to 1,635 parameters and maintain at least 99% accuracy.
change: Extend CompactQKV to reconstruct the first two value-bias coordinates as zeros, preserving all QKV weights and the baseline initialization sequence.
mechanism: Incremental value-to-output bias reparameterization
evidence_used: Omitting one value-bias coordinate achieved 99.92% accuracy at 1,636 parameters; the same position-independent value offset remains representable by the learned attention output-projection bias, motivating the smallest further reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5034000000000001, "parameters": 1635, "training_steps": 4999}



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
blocks have been applied. All blocks must apply. They may edit either or both
editable files, but together they must describe one implementation ready for
verification. The mechanism name is descriptive, not chosen from a fixed list.
Do not paste whole files, lengthy logs, or routine progress reports outside the
patch.
