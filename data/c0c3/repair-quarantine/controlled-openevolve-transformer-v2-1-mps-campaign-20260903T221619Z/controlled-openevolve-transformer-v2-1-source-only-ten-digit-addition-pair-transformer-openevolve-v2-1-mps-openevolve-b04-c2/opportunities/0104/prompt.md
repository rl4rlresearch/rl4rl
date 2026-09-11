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
verified_results: {"accuracy": 0.9953, "parameters": 1582, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,583-parameter design by quotienting positional row 6 will produce a 1,582-parameter model with at least 99% accuracy, because that row’s uniform channel component is erased by the first and final LayerNorms.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1576, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,577-parameter row-12 design with one analogous second-column value-basis rotation in the second attention head will produce a 1,576-parameter model with at least 99% accuracy.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9952, "parameters": 1578, "training_steps": 4999}
prior_hypothesis: Compacting positional row 11 while retaining optimization-sensitive row 7 unrestricted will reduce the qualified 1,579-parameter design to 1,578 parameters while maintaining at least 99% accuracy.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9978, "parameters": 1577, "training_steps": 4999}
prior_hypothesis: Extending the verified 1,578-parameter design by quotienting positional row 12 while retaining optimization-sensitive row 7 unrestricted will produce a 1,577-parameter model with at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Compacting positional row 14 while leaving optimization-sensitive rows 7 and 13 unrestricted will produce a 1,576-parameter model with at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows; reconstruct skipped rows exactly; and retain full-coordinate AdamW moments.
mechanism: Alternate-row positional common-mode quotient beyond sensitive row 13
evidence_used: The 1,577-parameter design compacting through row 12 achieved 99.78%, while compacting row 13 fell to 97.72%; previously, bypassing sensitive row 7 and compacting row 8 restored accuracy to 99.80%, motivating the analogous bypass to row 14.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting positional row 14 while leaving sensitive rows 7 and 13 unrestricted will reduce the model to 1,576 parameters while retaining at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows; reconstruct both skipped rows exactly; and preserve full-coordinate AdamW moments.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design achieved 99.78% accuracy through row 12, while row 13 compaction failed at 97.72%; earlier, bypassing sensitive row 7 and compacting row 8 restored accuracy to 99.80%, motivating the analogous correctly reconstructed bypass to row 14.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compacting positional rows 12 and 14 while leaving optimization-sensitive rows 7 and 13 unrestricted will reduce the model from 1,578 to 1,576 parameters while retaining at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows; reconstruct both skipped rows exactly in inference and full-coordinate AdamW optimization.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design compacting through row 12 achieved 99.78%; row 13 compaction fell to 97.72%, while the earlier analogous bypass of sensitive row 7 and compaction of row 8 restored accuracy to 99.80%.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compacting positional row 14 while retaining sensitive rows 7 and 13 unrestricted will produce a 1,576-parameter model with at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows, with explicit reconstruction of both skipped rows during inference and full-coordinate AdamW optimization.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design compacting through row 12 achieved 99.78%; row 13 compaction failed at 97.72%, while the analogous bypass of sensitive row 7 and compaction of row 8 achieved 99.80%. Prior row-14 submissions were unverifiable and provide no contrary accuracy result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting positional row 14 while leaving sensitive rows 7 and 13 unrestricted will produce a 1,576-parameter model with at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows; reconstruct both skipped rows exactly during inference and full-coordinate AdamW optimization.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design compacting through row 12 achieved 99.78%; row 13 compaction failed, while the analogous bypass of sensitive row 7 and compaction of row 8 achieved 99.80%. Prior row-14 submissions were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting positional row 14 while leaving sensitive rows 7 and 13 unrestricted will reduce the model from 1,577 to 1,576 parameters while retaining at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows, with exact inference reconstruction and full-coordinate AdamW updates.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design achieved 99.78% through row 12; row 13 compaction fell to 97.72%, while the analogous bypass of sensitive row 7 and compaction of row 8 achieved 99.80%. The row-14 attempts were unverifiable, so they provide no contrary accuracy result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting positional rows 12 and 14 while retaining sensitive rows 7 and 13 will produce a 1,576-parameter model with at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows, with exact reconstruction during inference and full-coordinate AdamW updates.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design compacting through row 12 achieved 99.78%; row 13 compaction failed at 97.72%, while the analogous bypass of sensitive row 7 and compaction of row 8 achieved 99.80%. Prior row-14 attempts were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compacting positional row 14 while retaining sensitive rows 7 and 13 unrestricted will reduce the verified 1,577-parameter design to 1,576 parameters while maintaining at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows, with exact reconstruction in inference and full-coordinate AdamW updates.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: Compacting through row 12 achieved 99.78% at 1,577 parameters; row 13 compaction fell to 97.72%, while the analogous bypass of sensitive row 7 and compaction of row 8 achieved 99.80%. Prior row-14 submissions were unverifiable and provide no contrary accuracy result.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Compacting positional row 14 while retaining rows 7 and 13 unrestricted will reduce the verified 1,577-parameter design to 1,576 parameters and maintain at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows, reconstruct their original order exactly, and preserve full-coordinate AdamW moments.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design compacting through row 12 achieved 99.78%; row 13 compaction failed at 97.72%, while bypassing the earlier sensitive row 7 and compacting row 8 achieved 99.80%. Prior row-14 attempts were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Quotienting positional row 14 while retaining sensitive rows 7 and 13 unrestricted will reduce the verified model to 1,576 parameters and maintain at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, 14, and the final nine rows, with matching reconstruction and full-coordinate AdamW updates.
mechanism: Skip-sensitive-row positional common-mode quotient
evidence_used: The 1,577-parameter design achieved 99.78% through row 12; row 13 compaction fell to 97.72%, while bypassing sensitive row 7 and compacting row 8 achieved 99.80%. Earlier row-14 attempts were unverifiable and provide no accuracy evidence against this corrected reconstruction.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Adding one joint query–key rotation to the verified 1,577-parameter row-12 design will produce a 1,576-parameter model with at least 99% accuracy.
change: Compact positional row 12 as previously verified, then jointly rotate one query/key coordinate pair and omit the query coefficient made exactly zero.
mechanism: Joint query–key orthogonal gauge fixation
evidence_used: The row-12 positional design achieved 99.78% at 1,577 parameters; the qualified models also successfully omit eight coefficients through analogous value-coordinate rotations, while further positional compaction at row 13 failed and row-14 attempts were unverifiable.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8275, "parameters": 1576, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1,577-parameter row-12 design with one analogous second-column value-basis rotation in the second attention head will produce a 1,576-parameter model with at least 99% accuracy.
change: Compact positional rows 0–6, 8–12, and the final nine rows, then omit one additional value-weight coefficient through a head-local rotation with a matching output-projection counter-rotation.
mechanism: Additional value-coordinate gauge fixation on the verified row-12 positional quotient
evidence_used: The row-12 positional design achieved 99.78% accuracy at 1,577 parameters, and its eight existing value-coordinate rotations remained qualified; this tests the same established symmetry while avoiding row 13, which fell to 97.72%, and the query–key rotation, which fell to 82.75%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1576, "training_steps": 4999}



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
