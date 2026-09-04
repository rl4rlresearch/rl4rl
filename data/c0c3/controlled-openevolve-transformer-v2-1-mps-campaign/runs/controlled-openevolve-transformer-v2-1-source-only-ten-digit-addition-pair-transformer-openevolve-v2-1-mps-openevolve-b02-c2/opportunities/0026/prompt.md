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
verified_results: {"accuracy": 0.996, "parameters": 1631, "training_steps": 4999}
prior_hypothesis: Omitting three value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,631-parameter model with at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1632, "training_steps": 4999}
prior_hypothesis: Omitting two value-bias coordinates from head 0 and three from head 1 alongside the qualified 4:3 key-bias layout will produce a 1,632-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1633, "training_steps": 4999}
prior_hypothesis: Omitting two value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,633-parameter model with at least 99% accuracy.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Omitting one value-bias coordinate from each attention head while retaining the qualified 4:3 key-bias layout will produce a 1,635-parameter model with at least 99% accuracy.
change: Extend `CompactQKV` to reconstruct the first value-bias coordinate of each head as zero while preserving all initialized QKV weights and the baseline initialization sequence.
mechanism: Head-balanced value-to-output bias reparameterization
evidence_used: One value-bias omission achieved 99.92% at 1,636 parameters, while two consecutive omissions in the first head fell to 50.34%; key-bias experiments showed that distributing omissions across heads avoided similar optimization collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9967, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting one value-bias coordinate from head 0 and two from head 1 alongside the qualified 4:3 key-bias layout will produce a 1,634-parameter model with at least 99% accuracy.
change: Reconstruct seven key-bias coordinates and three value-bias coordinates as zeros, distributing total key-plus-value omissions evenly at five per attention head while preserving baseline initialization draws and all QKV weights.
mechanism: Cross-projection per-head bias-capacity balancing
evidence_used: The head-balanced value omission design achieved 99.67% at 1,635 parameters, whereas two value omissions concentrated in head 0 achieved only 50.34%; assigning the additional omission to head 1 balances total omissions across the asymmetric 4:3 key layout.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting two value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,633-parameter model with at least 99% accuracy.
change: Remove one additional value-bias coordinate from head 0, yielding a symmetric 2:2 value-bias omission while preserving every QKV weight and the baseline initialization sequence.
mechanism: Head-balanced value-to-output bias reparameterization
evidence_used: The current 1:2 value-bias layout achieved 99.95% at 1,634 parameters, and the 1:1 head-balanced layout achieved 99.67%, whereas concentrating two omissions in head 0 collapsed to 50.34%; extending the successful distributed pattern to 2:2 is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1633, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting two value-bias coordinates from head 0 and three from head 1 alongside the qualified 4:3 key-bias layout will produce a 1,632-parameter model with at least 99% accuracy.
change: Extend `CompactQKV` to reconstruct two head-0 and three head-1 value-bias coordinates as zeros, balancing total key-plus-value omissions at six per head while preserving all QKV weights and baseline initialization draws.
mechanism: Cross-projection per-head bias-capacity balancing
evidence_used: The balanced 2:2 value-bias design achieved 99.96% at 1,633 parameters, and the 1:2 layout achieved 99.95%; assigning the next omission to head 1 balances the asymmetric 4:3 key omissions and is the smallest informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1632, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting three value-bias coordinates from each attention head alongside the qualified 4:3 key-bias layout will produce a 1,631-parameter model with at least 99% accuracy.
change: Extend `CompactQKV` to reconstruct three value-bias coordinates per head as zeros while preserving every QKV weight and the baseline initialization sequence.
mechanism: Alternating per-head value-bias compaction
evidence_used: The 2:2 value layout achieved 99.96% at 1,633 parameters, and extending the less-compacted head to 2:3 improved accuracy to 99.99% at 1,632; the next alternating reduction is 3:3 and retains one learned value-bias coordinate in each head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.996, "parameters": 1631, "training_steps": 4999}



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
