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
verified_results: {"accuracy": 0.9998, "parameters": 985, "training_steps": 4999}
prior_hypothesis: Sharing the final six nonreference LayerNorm bias coordinates will reduce the model from 986 to 985 parameters while retaining at least 99% accuracy, because each successive extension from three-way through five-way sharing maintained at least 99.97% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing the first and second effective MLP input coefficients alongside the successful third/fourth and fifth/sixth pairs will reduce the model from 997 to 996 parameters while retaining at least 99% accuracy, because all three constraints remain disjoint and preserve the reference-sensitive seventh coefficient independently.
change: Store four `fc1` coefficients and reconstruct effective pairs at channels one/two, three/four, and five/six, while preserving the seventh coefficient and the original constructor and initialization RNG streams.
mechanism: Complete disjoint nonterminal scalar-MLP input pairing
evidence_used: Disjoint third/fourth and fifth/sixth sharing achieved 99.96% at 997 parameters, while constraints involving the seventh coefficient missed or collapsed; this motivates extending the demonstrated nonterminal pairing pattern without altering that sensitive terminal coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 996, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing head 1’s second key projection to a residual coordinate will reduce the model from 996 to 991 parameters while retaining at least 99% accuracy, because it preserves two-dimensional content scoring and lets the learned embedding align the required source feature with that coordinate.
change: Remove the five learned tail coefficients from head 1’s second key channel, reconstruct that channel with an identity prefix and zero tail, and project the virtual optimizer state back onto this constraint after every update.
mechanism: Coordinate-aligned second key channel
evidence_used: Rank-two query/key routing achieved 99.93%, whereas rank one failed; unlike rank reduction, this preserves both score channels while challenging the assumption that every channel requires an independently learned full key mixture. Head 1’s strongly structured relative-bias pathway further makes a coordinate-aligned content channel plausible.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 991, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the leading tail coefficient of head 1’s first key channel will reduce the model from 991 to 990 parameters while retaining at least 99% accuracy, because it incrementally extends the successful coordinate-aligned key constraint without reducing the two-dimensional score rank.
change: Omit one coefficient from head 1’s first key-tail row, reconstruct it as zero, and project the virtual optimizer state onto both key constraints after every update.
mechanism: Partial companion-key coordinate alignment
evidence_used: Fixing all five tail coefficients of head 1’s second key channel achieved 99.94% at 991 parameters; removing only one neighboring coefficient is the smallest direct test of whether that demonstrated coordinate alignment extends to the companion channel.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 990, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the second tail coefficient of head 1’s first key channel will reduce the model from 990 to 989 parameters while retaining at least 99% accuracy, because fixing its adjacent leading coefficient achieved 99.97% without reducing the proven two-dimensional attention score rank.
change: Omit the first two tail coefficients from head 1’s first key channel, reconstruct them as zeros, and project the virtual optimizer state onto this extended constraint after every update.
mechanism: Incremental companion-key coordinate alignment
evidence_used: The immediately preceding result fixed one companion-channel tail coefficient and achieved 99.97% at 990 parameters, while fixing all five coefficients of head 1’s second key channel achieved 99.94%; extending the successful constraint by one adjacent coefficient is the smallest informative compression step.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 989, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the third tail coefficient of head 1’s first key channel will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because fixing its first two adjacent coefficients achieved 99.98% without reducing the proven two-dimensional attention score rank.
change: Omit the first three tail coefficients from head 1’s first key channel, reconstruct them as zeros, and project the virtual optimizer state onto the extended constraint after every update.
mechanism: Extended companion-key coordinate alignment
evidence_used: Fixing the first companion-channel coefficient achieved 99.97% at 990 parameters, and extending the same contiguous constraint to two coefficients improved to 99.98% at 989 parameters; removing the adjacent third coefficient is the smallest informative continuation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0, "parameters": 988, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the fourth tail coefficient of head 1’s first key channel will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because the first two coefficients tolerate zero constraints, the third is specifically load-bearing, and this leaves the terminal coefficient aligned with the reference-sensitive seventh MLP input coordinate learned independently.
change: Omit the fourth tail coefficient from head 1’s first key channel, reconstruct it as zero alongside the existing constraints, and project the virtual optimizer state onto the same constraint after every update.
mechanism: Selective companion-key coordinate alignment
evidence_used: Fixing the first two companion-channel tail coefficients achieved 99.98% at 989 parameters, while additionally fixing the third collapsed to 0%; this indicates coefficient-specific sensitivity and motivates testing the next nonterminal coefficient without altering the critical third or terminal fifth coefficient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9214, "parameters": 988, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the first two stored MLP input coefficients will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because their respective effective channel pairs already tolerate internal sharing, while the key-projection evidence shows features two and three tolerate zero contribution and the sensitive later coordinates remain independent.
change: Store three `fc1` coefficients instead of four, reuse the first across effective input channels one through four, preserve the fifth/sixth pair and seventh coefficient, and consume the removed constructor and initialization draws.
mechanism: Four-channel scalar-MLP input sharing
evidence_used: Complete disjoint nonterminal MLP pairing achieved 99.93%, and zeroing the first two companion-key tail coefficients achieved 99.98%; merging the corresponding early MLP groups is the smallest test that preserves the independently parameterized coordinates implicated by the failed third- and fourth-key-tail removals.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0668, "parameters": 988, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final three nonreference LayerNorm bias coordinates will reduce the model from 989 to 988 parameters while retaining at least 99% accuracy, because terminal bias-pair sharing previously achieved 99.98% and this leaves the sensitive attention and MLP coordinates unchanged.
change: Store five rather than six final-LayerNorm bias contrasts and reuse the fifth across the final three nonreference residual channels.
mechanism: Extended terminal final-LayerNorm bias sharing
evidence_used: Terminal final-LayerNorm bias sharing achieved 99.98%, whereas recent failures involved additional key-tail removal or broader MLP sharing; extending the successful isolated bias constraint is the smallest orthogonal compression step.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 988, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final four nonreference LayerNorm bias coordinates will reduce the model from 988 to 987 parameters while retaining at least 99% accuracy, because sharing the final three achieved 99.97% and this extends only the same successful isolated constraint.
change: Store four rather than five final-LayerNorm bias contrasts and reuse the fourth across the final four nonreference residual channels.
mechanism: Extended four-way terminal final-LayerNorm bias sharing
evidence_used: The current 988-parameter design shares the final three nonreference LayerNorm biases and achieved 99.97%; extending that contiguous terminal sharing by one coordinate is the smallest informative compression step and leaves attention and MLP parameters unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 987, "training_steps": 4999}

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
