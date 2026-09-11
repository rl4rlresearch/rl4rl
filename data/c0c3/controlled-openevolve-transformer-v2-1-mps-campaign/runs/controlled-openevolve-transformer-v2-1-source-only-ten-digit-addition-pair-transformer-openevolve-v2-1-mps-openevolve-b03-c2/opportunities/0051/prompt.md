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
verified_results: {"accuracy": 0.9997, "parameters": 1505, "training_steps": 4999}
prior_hypothesis: Fixing one query-bias coordinate to zero will reduce the current model from 1506 to 1505 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1502, "training_steps": 4999}
prior_hypothesis: Fixing a second relative attention-output coordinate in the first head will reduce the verified 1503-parameter design to 1502 parameters while maintaining at least 99% accuracy, because two conditioned pivot directions can select this representative through an invertible value/output basis change without changing the initialized function.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1503, "training_steps": 4999}
prior_hypothesis: Fixing the same relative attention-output coordinate independently in the first head will reduce the verified 1504-parameter model to 1503 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1507, "training_steps": 4999}
prior_hypothesis: Quotienting the seven remaining feature coordinates shared between every token row and every positional row will reduce the verified 1514-parameter model to 1507 parameters while retaining at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Applying the qualified residual-output quotients and removing the feature-uniform input coordinate from every QKV and `fc1` row will produce a 1514-parameter model with at least 99% accuracy, because non-affine LayerNorm outputs have zero feature mean and the optimizer reconstructs the omitted coordinates for folded-factor AdamW and clipping.
change: Store attention and MLP residual-output weights with seven relative output coordinates, and store all 24 QKV plus 12 `fc1` rows with seven relative input coordinates while maintaining their full-coordinate optimizer dynamics.
mechanism: LayerNorm-nullspace input-weight quotient
evidence_used: Reference Design 3 achieved 100% accuracy at 1550 parameters after quotienting all attention and MLP residual-output columns. Its QKV and `fc1` projections both consume non-affine LayerNorm outputs, making one feature-uniform input direction per row functionally null and motivating the additional 36-parameter reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1514, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the qualified input- and output-weight quotients to the current folded-LayerNorm model will reduce it from 1571 to 1514 learned parameters while retaining at least 99% accuracy.
change: Store QKV and `fc1` rows with seven relative input coordinates, store attention and MLP residual-output projections with seven relative output coordinates, and preserve their full-coordinate AdamW and clipping dynamics.
mechanism: LayerNorm-nullspace input and residual-output weight quotients
evidence_used: Reference Design 2 verified the same combined quotient implementation at 1514 parameters and 99.97% accuracy; Reference Design 3 independently verified the residual-output portion at 1550 parameters and 100% accuracy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.733, "parameters": 1514, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the seven remaining feature coordinates shared between every token row and every positional row will reduce the verified 1514-parameter model to 1507 parameters while retaining at least 99% accuracy.
change: Fix the final token-embedding row to zero, transfer its feature offset into every positional row, and preserve full-coordinate embedding AdamW and clipping dynamics with a coupled quotient update.
mechanism: Joint token-position common-offset quotient
evidence_used: The current 1514-parameter implementation achieved 99.97% after trajectory-preserving input/output quotients. A common token-row shift can be canceled by the opposite positional shift while changing output logits only uniformly across the vocabulary, so this removes seven parameters without restricting the learned function class.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1507, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified 1507-parameter input/output and token-position quotients, then fixing one final-LayerNorm scale coordinate to select a positive global-logit-scale representative, will yield 1506 parameters and at least 99% accuracy.
change: Reproduce Reference Design 3’s trajectory-aware quotients and replace the 16-parameter final LayerNorm affine transform with 15 learned coordinates whose last scale is fixed to one.
mechanism: Joint quotient compression with final-LayerNorm affine-scale gauge fixing
evidence_used: Reference Design 3 achieved 99.91% accuracy at 1507 parameters. Multiplying every final-LayerNorm scale and bias coordinate by the same positive scalar changes only the global logit scale and therefore not protected argmax decoding, motivating removal of one additional affine-scale degree.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1506, "training_steps": 4999}

RECENT RESULT
hypothesis: Reproducing the verified 1506-parameter design and fixing the last final-LayerNorm bias coordinate to zero will yield 1505 parameters while maintaining at least 99% accuracy.
change: Apply the qualified weight, residual-output, and token-position quotients with their trajectory-aware optimizer; retain the final-LayerNorm scale gauge; and store seven rather than eight final-LayerNorm bias coordinates.
mechanism: Final-LayerNorm bias-coordinate ablation atop the qualified joint quotients
evidence_used: Reference Design 3 achieved 99.96% accuracy at 1506 parameters. Removing one final-affine bias coordinate is the smallest remaining capacity restriction and directly tests whether that margin survives a one-parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7043, "parameters": 1505, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate to zero will reduce the verified 1506-parameter design to 1505 parameters while retaining at least 99% accuracy, because an invertible within-head query/key basis rotation can select that representative without changing attention dot products.
change: Reproduce the qualified weight, residual-output, token-position, value-bias, LayerNorm-folding, and final-scale quotients, then store seven query-bias coordinates and reconstruct the eighth as zero.
mechanism: Within-head query-key basis gauge fixing atop trajectory-preserving quotients
evidence_used: The 1506-parameter joint-quotient design achieved 99.96%, while directly ablating a final-LayerNorm bias coordinate collapsed to 70.43%; this motivates testing an attention basis gauge that preserves the generic learned function class instead of another final-affine capacity restriction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1505, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate in each attention head will reduce the verified 1505-parameter model to 1504 parameters while retaining at least 99% accuracy, because each head has an independent invertible query/key basis symmetry.
change: Store three query-bias coordinates per four-dimensional head and reconstruct the fourth coordinate of each head as zero.
mechanism: Per-head query-key basis gauge fixing
evidence_used: Fixing one query-bias coordinate already achieved 100% accuracy at 1505 parameters; applying the same gauge independently to the other head is the smallest symmetry-grounded extension.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9861, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one query-bias coordinate to zero will reduce the current model from 1506 to 1505 parameters while maintaining at least 99% accuracy.
change: Store seven learned query-bias coordinates and reconstruct the eighth as zero, leaving the protected decoding interface and training procedure unchanged.
mechanism: Single-coordinate query-key basis gauge fixing
evidence_used: Reference Design 1 verified this exact 1505-parameter gauge at 100% accuracy; the 1504-parameter per-head extension reached only 98.61%, identifying the single-coordinate version as the qualified boundary.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1505, "training_steps": 4999}

RECENT RESULT
hypothesis: Concentrating both fixed query-bias coordinates in the already-gauged second head will produce a 1504-parameter model with at least 99% accuracy, because it preserves a fully unconstrained first head while using the second head’s within-head query/key basis symmetry.
change: Apply the verified final-LayerNorm scale gauge and store six query-bias coordinates, reconstructing the final two coordinates of the second head as zero.
mechanism: Single-head concentrated query-basis gauge fixing
evidence_used: The single-coordinate query gauge reached 99.97% at 1505 parameters, while distributing two fixed coordinates across both heads reached 98.61% at 1504; concentrating both constraints in one head tests whether leaving one head fully unconstrained restores the missing margin.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6047, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying the verified 1505-parameter quotient design and fixing one relative attention-output coordinate through the exact within-head value/output basis symmetry will yield 1504 parameters with at least 99% accuracy.
change: Reproduce the qualified weight, embedding, residual-output, LayerNorm-scale, and single query-bias quotients, then store one attention-projection row with six coordinates and reconstruct its seventh coordinate as zero after a function-preserving initialization basis change.
mechanism: Independent attention value/output basis gauge fixing
evidence_used: The single query-bias gauge achieved 99.97–100% accuracy at 1505 parameters, while both 1504 query-only extensions failed; an independent value/output basis symmetry removes one parameter without imposing another constraint on attention logits.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1504, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the same relative attention-output coordinate independently in the first head will reduce the verified 1504-parameter model to 1503 parameters while maintaining at least 99% accuracy.
change: Store six relative coordinates for one projection row in each attention head, reconstruct both fixed coordinates as zero, and apply function-preserving within-head value/output basis changes at initialization.
mechanism: Per-head attention value/output basis gauge fixing
evidence_used: The current 1504-parameter design achieved 99.79% after one value/output gauge; using the independent symmetry of the other head extends that successful mechanism without adding another query-bias restriction, whose tested 1504-parameter variants failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1503, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second relative attention-output coordinate in the first head will reduce the verified 1503-parameter design to 1502 parameters while maintaining at least 99% accuracy, because two conditioned pivot directions can select this representative through an invertible value/output basis change without changing the initialized function.
change: Reproduce the verified per-head projection gauges, then store five coordinates for the first head’s gauged projection row and six for the second head’s row, reconstructing the omitted coordinates as zero and applying matching value-weight basis changes at initialization.
mechanism: Additional within-head value/output basis gauge fixing
evidence_used: Reference Design 1 achieved 99.90% accuracy at 1503 parameters after independently fixing one attention-output coordinate in each head; extending that successful value/output symmetry by one coordinate is the smallest symmetry-grounded reduction, while prior additional query-bias restrictions failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1502, "training_steps": 4999}



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
