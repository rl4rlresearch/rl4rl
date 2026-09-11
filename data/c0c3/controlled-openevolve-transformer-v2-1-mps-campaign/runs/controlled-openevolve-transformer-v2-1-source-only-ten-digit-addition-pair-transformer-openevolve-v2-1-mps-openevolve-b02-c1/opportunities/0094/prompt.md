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
verified_results: {"accuracy": 0.9998, "parameters": 989, "training_steps": 4999}
prior_hypothesis: Fixing the second tail coefficient of head 1’s first key channel will reduce the model from 990 to 989 parameters while retaining at least 99% accuracy, because fixing its adjacent leading coefficient achieved 99.97% without reducing the proven two-dimensional attention score rank.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing the final two learned LayerNorm bias coordinates will reduce the model from 1,001 to 1,000 parameters while retaining at least 99% accuracy, because the corresponding terminal scale pair already tolerates sharing.
change: Store six final LayerNorm bias values and reuse the last value for the seventh coordinate, while retaining the fixed eighth-coordinate reference.
mechanism: Matched terminal LayerNorm bias sharing
evidence_used: Sharing the terminal LayerNorm scale pair achieved 99.95%, whereas extending that constraint to a scale triplet reached 96.84%; this motivates the analogous isolated pair constraint in the previously untested bias parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1000, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the width-one MLP’s hidden bias at zero will reduce the model from 1,000 to 999 parameters while retaining at least 99% accuracy, because it preserves the learned nonlinear MLP weights and all demonstrated load-bearing rank-two and relative-position attention pathways.
change: Remove the single trainable `fc1` bias while consuming its constructor-equivalent random draw to preserve initialization of subsequent parameters.
mechanism: Zero-offset scalar MLP
evidence_used: The current model achieves 99.98% at 1,000 parameters; prior failures identify attention rank and relative-position structure as sensitive, so the previously untested scalar MLP offset is a more informative compression target while those pathways remain unchanged.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 999, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final two learned MLP output contrasts will reduce the model from 999 to 998 parameters while retaining at least 99% accuracy, because terminal residual-channel sharing has succeeded in both final LayerNorm scale and bias parameters, and removing the scalar MLP offset retained 99.98% accuracy.
change: Store six instead of seven `fc2` output contrasts, reconstruct the seventh by reusing the sixth, and preserve the existing constructor and initialization RNG streams.
mechanism: Terminal-paired scalar MLP output
evidence_used: Sharing the terminal final-LayerNorm bias pair achieved 99.98%, and fixing the width-one MLP bias at zero preserved 99.98%; together these results motivate testing one isolated terminal constraint on the remaining scalar MLP output vector without modifying sensitive attention pathways.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2576, "parameters": 998, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final two effective MLP input coefficients will reduce the model from 999 to 998 parameters while retaining at least 99% accuracy, because removing the scalar MLP offset retained 99.98% and this preserves the independently learned output routing that failed under terminal `fc2` sharing.
change: Replace `fc1` with a six-parameter input projection that reuses its final coefficient for the seventh input channel, while preserving the original constructor and initialization RNG streams.
mechanism: Terminal-paired scalar MLP input
evidence_used: Fixing the width-one MLP bias at zero retained 99.98%, whereas sharing the final two MLP output contrasts collapsed to 25.76%; this motivates compressing the scalar neuron’s input direction while leaving its sensitive output vector unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9895, "parameters": 998, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing the seventh effective MLP input coefficient three-quarters of the way from the sixth coefficient toward the fixed zero reference will reduce the model from 999 to 998 parameters while exceeding 99% accuracy.
change: Store six `fc1` input coefficients, derive the seventh as `0.75` times the sixth, and preserve the original constructor and initialization RNG streams.
mechanism: Reference-skewed terminal MLP input interpolation
evidence_used: Directly sharing the terminal `fc1` coefficients narrowly missed at 98.95%, while introducing a distinct quarter-step transition instead of equality previously raised head-1 relative-bias compression from failure to 99.97%.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.26890000000000003, "parameters": 998, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the fifth and sixth effective MLP input coefficients while keeping the seventh coefficient independent will reduce the model from 999 to 998 parameters and retain at least 99% accuracy.
change: Store six `fc1` coefficients, reuse the fifth for the sixth effective input channel, preserve the terminal coefficient, and consume constructor and initialization draws for the removed parameter.
mechanism: Nonterminal-paired scalar MLP input
evidence_used: Terminal `fc1` sharing narrowly missed at 98.95%, while changing the terminal coefficient toward its fixed reference collapsed to 26.89%; preserving that reference-sensitive terminal coefficient and moving the sharing constraint to the neighboring nonterminal pair isolates whether the failure was specific to the terminal geometry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 998, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the third and fourth effective MLP input coefficients in addition to the successful fifth/sixth pair will reduce the model from 998 to 997 parameters while retaining at least 99% accuracy, because it preserves the first two score-basis coefficients and the reference-sensitive seventh coefficient independently.
change: Store five `fc1` coefficients and reconstruct effective pairs at channels three/four and five/six, while preserving constructor and initialization RNG streams.
mechanism: Disjoint nonterminal scalar-MLP input pairing
evidence_used: Fifth/sixth `fc1` sharing achieved 99.83%, whereas terminal sharing narrowly missed at 98.95% and interpolating the terminal coefficient collapsed to 26.89%; this motivates another disjoint nonterminal pair while leaving the terminal coefficient independent.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 997, "training_steps": 4999}

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
