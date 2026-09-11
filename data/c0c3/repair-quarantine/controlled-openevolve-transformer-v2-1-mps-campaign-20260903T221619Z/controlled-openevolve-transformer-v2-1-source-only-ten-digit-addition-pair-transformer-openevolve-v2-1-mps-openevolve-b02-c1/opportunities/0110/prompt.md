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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"accuracy": 0.9998999999999999, "parameters": 982, "training_steps": 4999}
prior_hypothesis: Tying head 0’s load-bearing fourth-farthest relative bias to its nearest learned neighbor will reduce the model from 983 to 982 parameters while retaining at least 99% accuracy, because it preserves a learned nonzero boundary contrast rather than collapsing that distance to the zero reference.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing the final five nonreference LayerNorm bias coordinates will reduce the model from 987 to 986 parameters while retaining at least 99% accuracy, because the immediately preceding four-way sharing constraint achieved 99.98%.
change: Store three rather than four final-LayerNorm bias contrasts and reuse the third across the final five nonreference residual channels.
mechanism: Extended five-way terminal final-LayerNorm bias sharing
evidence_used: Extending final-LayerNorm sharing from three coordinates at 988 parameters to four coordinates at 987 parameters maintained 99.98% accuracy, making one further contiguous extension the smallest evidence-backed compression step.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 986, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final six nonreference LayerNorm bias coordinates will reduce the model from 986 to 985 parameters while retaining at least 99% accuracy, because each successive extension from three-way through five-way sharing maintained at least 99.97% accuracy.
change: Store two rather than three final-LayerNorm bias contrasts and reuse the second across the final six nonreference residual channels.
mechanism: Extended six-way terminal final-LayerNorm bias sharing
evidence_used: Five-way terminal sharing achieved 99.97% at 986 parameters, following four-way sharing at 99.98% and three-way sharing at 99.97%; extending the same isolated constraint by one adjacent coordinate is the smallest evidence-backed compression step.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 985, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the learned canonical value map across both attention heads will reduce the model from 985 to 973 parameters while retaining at least 99% accuracy, because addition needs distinct attention routes but can encode the digits retrieved by those routes in one common learned feature space.
change: Store one learned 4-by-3 value tail, use it for both heads, initialize it by projecting the original head-specific maps onto their shared mean, and project the virtual optimizer state back onto that shared value subspace after every update.
mechanism: Head-shared canonical value subspace
evidence_used: Fixing five coefficients of one key channel preserved 99.94% accuracy while retaining both score dimensions, showing that head-local content projections contain substantial redundancy; unlike the failed rank-one change, this patch preserves both rank-two routing mechanisms, their relative biases, and their independent output readouts.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.15710000000000002, "parameters": 973, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing all seven nonreference final-LayerNorm bias coordinates will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because every incremental extension from three-way through six-way terminal sharing maintained at least 99.97% accuracy.
change: Store one final-LayerNorm bias contrast instead of two and reuse it across all seven nonreference residual coordinates.
mechanism: Complete nonreference final-LayerNorm bias sharing
evidence_used: Six-way terminal bias sharing achieved 99.98% at 985 parameters, following successful five-way, four-way, and three-way sharing; completing the same isolated constraint is the smallest evidence-backed next compression.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1427, "parameters": 984, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final three nonreference LayerNorm scale coordinates will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because the verified 985-parameter design already shares the final two scales and this leaves the load-bearing distinct bias coordinate unchanged.
change: Store five rather than six final-LayerNorm scale parameters and reuse the fifth across the final three nonreference residual coordinates.
mechanism: Extended terminal final-LayerNorm scale sharing
evidence_used: The current design achieves 99.98% while tying the final two nonreference scales; extending that isolated constraint by one coordinate is the smallest orthogonal test after complete bias sharing failed, indicating that the first bias contrast should remain independent.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8195, "parameters": 984, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing final-LayerNorm scales for residual coordinates two and three will reduce the model from 985 to 984 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient and tolerate zero companion-key contributions, while the sensitive later coordinates remain independent.
change: Store five final-LayerNorm scales instead of six and reuse the third scale for coordinates two and three, preserving the existing scale sharing for coordinates five and six.
mechanism: Mid-pair final-LayerNorm scale sharing
evidence_used: The 989-parameter design achieved 99.98% with the first two companion-key tail coefficients—corresponding to residual coordinates two and three—fixed at zero, and disjoint MLP input pairing achieved 99.93%; unlike the failed terminal scale extension, this change leaves coordinates four through six unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9975, "parameters": 984, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing final-LayerNorm scales for residual coordinates zero and one will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because those coordinates already share an MLP input coefficient, while all scale and bias distinctions implicated by prior failures remain unchanged.
change: Store four final-LayerNorm scales instead of five and reuse the first scale for coordinates zero and one, preserving the existing sharing for coordinates two/three and five/six.
mechanism: Early-pair final-LayerNorm scale sharing
evidence_used: Mid-pair final-LayerNorm scale sharing achieved 99.75% at 984 parameters when aligned with an existing MLP input pair; applying the same isolated constraint to the other early MLP-shared pair is the smallest informative continuation, unlike the failed extension of the sensitive terminal scale group.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3494, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the shared bias of residual coordinates one through six to the reference value will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because it preserves the distinct first-coordinate bias destroyed by the failed complete-sharing change.
change: Store only the first final-LayerNorm bias contrast and reconstruct all remaining coordinates with zero bias.
mechanism: Reference-fixed shared terminal final-LayerNorm bias
evidence_used: Six-way terminal bias sharing achieved 99.98% while complete nonreference sharing collapsed to 14.27%; this isolates the first-coordinate contrast as load-bearing and makes removal of the remaining shared terminal contrast the most informative complementary test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8633, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 1’s third and fourth companion-key tail coefficients will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because both coefficients require learned nonzero contributions and correspond to residual coordinates already successfully paired by the MLP.
change: Replace the two independently stored companion-key coefficients with one shared coefficient, initialize it from their mean, split its gradient across both virtual coordinates, and reproject the virtual key factor after every optimizer update.
mechanism: Shared load-bearing companion-key pair
evidence_used: Fixing the third coefficient caused complete failure and fixing the fourth reached only 92.14%, showing both are load-bearing; meanwhile, disjoint MLP input pairing achieved 99.93%, specifically supporting shared treatment of their corresponding residual coordinates without deleting their learned contribution.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5524, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s third-farthest relative-attention distance to its existing zero reference will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because its two farthest distances are already tied successfully and all sensitive LayerNorm, MLP, and key-projection distinctions remain unchanged.
change: Store one fewer head-0 relative-bias parameter and reconstruct its three farthest distances as the shared zero reference.
mechanism: Extended head-0 far-distance bias sharing
evidence_used: The verified 984-parameter design achieves 99.75% while already tying head 0’s two farthest distances; extending that same isolated positional-bias constraint by one adjacent distance is the smallest untested compression after additional LayerNorm and companion-key sharing failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 983, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s fourth-farthest relative-attention distance to its existing zero reference will reduce the model from 983 to 982 parameters while retaining at least 99% accuracy.
change: Store one fewer head-0 relative-bias parameter and reconstruct its four farthest distances as the shared zero reference.
mechanism: Four-way head-0 far-distance bias sharing
evidence_used: Extending head 0’s zero-reference sharing from two to three farthest distances achieved 99.94% accuracy at 983 parameters, making one further adjacent extension the smallest evidence-backed compression step.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1419, "parameters": 982, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying head 0’s load-bearing fourth-farthest relative bias to its nearest learned neighbor will reduce the model from 983 to 982 parameters while retaining at least 99% accuracy, because it preserves a learned nonzero boundary contrast rather than collapsing that distance to the zero reference.
change: Store one fewer head-0 relative-bias parameter and reuse its final learned bias for both the fourth- and fifth-farthest distances.
mechanism: Adjacent head-0 boundary-bias sharing
evidence_used: Three zero-referenced farthest distances achieved 99.94% at 983 parameters, whereas extending the zero-reference group to the fourth-farthest distance collapsed accuracy to 14.19%; adjacent sharing directly tests whether that distance needs a nonzero learned contrast rather than an independent scalar.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 982, "training_steps": 4999}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the transformer represents or computes the task. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
