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
verified_results: {"accuracy": 0.9991, "parameters": 1586, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,587-parameter design by quotienting positional row 2 will produce a 1,586-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first LayerNorm.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9936, "parameters": 1588, "training_steps": 4999}
prior_hypothesis: Quotienting attention projection columns 0, 1, 2, 4, 5, and 6 while leaving optimization-sensitive column 3 unrestricted will produce a 1,588-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9986, "parameters": 1587, "training_steps": 4999}
prior_hypothesis: Quotienting attention projection columns 0, 1, 2, 4, 5, 6, and 7 while leaving optimization-sensitive column 3 unrestricted will produce a 1,587-parameter model with at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9991, "parameters": 1589, "training_steps": 4999}
prior_hypothesis: Quotienting attention projection columns 0, 1, 2, 4, and 5 while leaving optimization-sensitive column 3 unrestricted will produce a 1,589-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified 1,597-parameter design by quotienting `fc2` column 7 will produce a 1,596-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.
change: Compact `fc2` columns 0, 1, 2, 4, 5, 6, and 7, preserve their original ordering during reconstruction, and retain full-coordinate AdamW optimization through the existing quotient machinery.
mechanism: Seven-column MLP residual-output common-mode quotient
evidence_used: Quotienting columns 0, 1, 2, 4, 5, and 6 achieved 99.93% accuracy at 1,597 parameters; extending the successful run after column 4 to adjacent column 7 is the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1596, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,596-parameter design by quotienting `fc2` column 8 will produce a 1,595-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.
change: Parameterize `fc2` columns 0, 1, 2, and 4 through 8 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave column 3 and columns 9 through 11 unrestricted.
mechanism: Eighth-column MLP residual-output common-mode quotient
evidence_used: Reference Design 3 achieved 99.81% accuracy at 1,596 parameters after successfully extending the compacted run through column 7; quotienting adjacent column 8 is the closest incremental reduction using the same exact LayerNorm-null direction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,595-parameter design by quotienting `fc2` column 9 will produce a 1,594-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.
change: Parameterize `fc2` columns 0, 1, 2, and 4 through 9 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave column 3 and columns 10–11 unrestricted.
mechanism: Ninth-column MLP residual-output common-mode quotient
evidence_used: Quotienting through column 8 achieved 99.97% accuracy at 1,595 parameters; extending the same exact LayerNorm-null quotient to adjacent column 9 is the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified 1,594-parameter design by quotienting `fc2` column 10 will produce a 1,593-parameter model with at least 99% accuracy, because its uniform output component is erased by the final LayerNorm while column 3 remains unrestricted.
change: Parameterize `fc2` columns 0, 1, 2, and 4 through 10 in the seven-dimensional zero-sum basis, reconstruct their original ordering, and leave columns 3 and 11 unrestricted.
mechanism: Tenth-column MLP residual-output common-mode quotient
evidence_used: Quotienting columns 0, 1, 2, and 4 through 9 achieved 99.81% accuracy at 1,594 parameters; extending the successful adjacent run to column 10 is the closest incremental reduction using the same exact LayerNorm-null direction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9969, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting `fc2` column 11 will reduce the model from 1,593 to 1,592 learned parameters while retaining at least 99% accuracy, because its uniform residual-channel component is erased by the final LayerNorm and optimization-sensitive column 3 remains unrestricted.
change: Parameterize every `fc2` output column except column 3 in the seven-dimensional zero-sum basis, reconstruct the original ordering in the forward pass, and continue using full-coordinate AdamW moments.
mechanism: Final-column MLP residual-output common-mode quotient
evidence_used: The current design achieved 99.69% accuracy after quotienting columns 0, 1, 2, and 4 through 10; extending the same exact function-null quotient to the sole remaining adjacent column 11 is the closest incremental reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,592-parameter design by quotienting the third attention output-projection column will produce a 1,591-parameter model with at least 99% accuracy, because its uniform residual-channel component is erased by subsequent LayerNorms.
change: Reproduce the qualified all-but-column-3 `fc2` compaction, then parameterize the first three attention projection columns in the seven-dimensional zero-sum basis; the existing matrix-aware optimizer continues to maintain full-coordinate AdamW moments.
mechanism: Third-column attention residual-output common-mode quotient
evidence_used: Reference Design 2 achieved 99.97% accuracy at 1,592 parameters with every `fc2` column except column 3 compacted, while the qualified architecture already compacts the first two attention projection columns through the same exact LayerNorm-null direction; extending that adjacent attention quotient is the closest untested reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,591-parameter design by quotienting the fourth attention output-projection column will produce a 1,590-parameter model with at least 99% accuracy, because its uniform residual-channel component is erased by subsequent LayerNorms.
change: Reproduce the qualified all-but-column-3 `fc2` compaction, then parameterize the first four attention output-projection columns in the seven-dimensional zero-sum basis; the existing matrix-aware optimizer supports both expanded compact matrices.
mechanism: Fourth-column attention residual-output common-mode quotient
evidence_used: Reference Design 1 achieved 99.81% accuracy at 1,591 parameters after quotienting the first three attention projection columns, so extending the same exact LayerNorm-null quotient to adjacent column 3 is the closest incremental reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6817, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting attention projection columns 0, 1, 2, and 4 while leaving optimization-sensitive column 3 unrestricted will produce a 1,590-parameter model with at least 99% accuracy.
change: Reproduce the qualified all-but-column-3 MLP quotient, then compact attention projection columns 0, 1, 2, and 4 with full-coordinate AdamW moments and reconstruct their original ordering during the forward pass.
mechanism: Alternate-column attention residual-output quotient
evidence_used: The 1,591-parameter design with attention columns 0–2 compacted achieved 99.81%, whereas adding column 3 collapsed to 68.17%; the analogous MLP collapse was avoided by leaving column 3 unrestricted and quotienting column 4 instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting attention projection columns 0, 1, 2, 4, and 5 while leaving optimization-sensitive column 3 unrestricted will produce a 1,589-parameter model with at least 99% accuracy.
change: Extend the current alternate-column attention quotient to column 5, preserve original column ordering during reconstruction, and retain full-coordinate AdamW optimization.
mechanism: Fifth-column attention residual-output common-mode quotient
evidence_used: The current design achieved 99.90% at 1,590 parameters after avoiding the column-3 collapse by quotienting column 4 instead; the analogous MLP sequence successfully extended from alternate column 4 to adjacent column 5, reaching 99.93%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting attention projection columns 0, 1, 2, 4, 5, and 6 while leaving optimization-sensitive column 3 unrestricted will produce a 1,588-parameter model with at least 99% accuracy.
change: Extend the qualified alternate-column attention quotient through column 6, preserve original column ordering during reconstruction, and retain full-coordinate AdamW optimization.
mechanism: Sixth-column attention residual-output common-mode quotient
evidence_used: Reference Design 3 achieved 99.91% accuracy at 1,589 parameters with attention columns 0, 1, 2, 4, and 5 compacted; adding adjacent column 6 is the closest incremental reduction that avoids the documented column-3 collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9936, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting attention projection columns 0, 1, 2, 4, 5, 6, and 7 while leaving optimization-sensitive column 3 unrestricted will produce a 1,587-parameter model with at least 99% accuracy.
change: Extend the qualified alternate-column attention quotient to column 7, reconstruct the original column ordering, and retain full-coordinate AdamW optimization.
mechanism: Seventh-column attention residual-output common-mode quotient
evidence_used: The 1,588-parameter reference achieved 99.36% accuracy with attention columns 0, 1, 2, 4, 5, and 6 compacted; quotienting adjacent column 7 is the closest incremental reduction that avoids the documented column-3 collapse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9986, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,587-parameter design by quotienting positional row 2 will produce a 1,586-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first LayerNorm.
change: Reproduce the qualified all-but-column-3 MLP and attention projection quotients, then compact positional rows 0–2 and the final nine rows while retaining full-coordinate AdamW moments.
mechanism: Twelfth-row positional common-mode quotient
evidence_used: Reference Design 2 achieved 99.86% accuracy at 1,587 parameters with all attention projection columns except column 3 compacted; its existing eleven positional-row quotients also succeeded, motivating the closest incremental extension to adjacent row 2.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1586, "training_steps": 4999}



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
