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
verified_results: {"accuracy": 0.9983, "parameters": 998, "training_steps": 4999}
prior_hypothesis: Sharing the fifth and sixth effective MLP input coefficients while keeping the seventh coefficient independent will reduce the model from 999 to 998 parameters and retain at least 99% accuracy.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing head 1’s remaining pre-boundary bias with two evenly spaced interpolants will reduce the model from 1,002 to 1,001 parameters while maintaining at least 99% accuracy, because it preserves distinct transition values instead of extending the failed equality constraint.
change: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct the two preceding plateau-transition biases at one-third and two-thirds between neighboring learned endpoints.
mechanism: Two-step learned head-1 boundary interpolation
evidence_used: Head 1’s single learned boundary interpolation retained 99.67%, while extending the plateau by direct equality collapsed to 47.11%; this motivates lengthening the successful smooth transition without compressing the sensitive head-0 pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9831, "parameters": 1001, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the first two final LayerNorm scales while preserving the successful shared terminal pair will reduce the model from 1,002 to 1,001 parameters and retain at least 99% accuracy, because it avoids extending the terminal pair into the failed flat triplet.
change: Store five final LayerNorm scale parameters, reconstruct the first two coordinates from one learned scale, preserve independent middle scales and the existing shared sixth/seventh scale pair, and retain the fixed eighth-coordinate reference.
mechanism: Disjoint adjacent final-scale sharing
evidence_used: Sharing the final two learned scales achieved 99.95%, while extending that same group to three scales fell to 96.84%; a separate pair tests additional scale compression without altering the demonstrated terminal-pair structure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3901, "parameters": 1001, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing head 1’s two transition biases at one-quarter and five-eighths between learned endpoints will reduce the model from 1,002 to 1,001 parameters while achieving at least 99% accuracy, because it preserves the successful midpoint hierarchy while separating the transition more strongly from the plateau than the 98.31%-accurate equal-thirds reconstruction.
change: Remove one head-1 relative-bias parameter and derive the two preceding plateau-transition biases using quarter-step interpolation followed by a midpoint to the plateau.
mechanism: Endpoint-skewed hierarchical boundary interpolation
evidence_used: Equal-thirds two-step interpolation reached 98.31%, the closest failed 1,001-parameter design, while direct plateau extension reached only 47.11%; shifting the reconstructed boundary away from the plateau tests whether insufficient transition separation caused the narrow miss.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1001, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing one additional head-1 transition bias with the successful quarter-step hierarchy will reduce the model from 1,001 to 1,000 parameters while retaining at least 99% accuracy.
change: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct three transition biases at one-quarter, seven-sixteenths, and twenty-three-thirty-seconds between learned endpoints.
mechanism: Recursively skewed head-1 boundary interpolation
evidence_used: Endpoint-skewed two-bias interpolation achieved 99.97% at 1,001 parameters, whereas equal-thirds interpolation reached 98.31%; recursively extending the successful skew preserves its transition geometry while testing one further parameter reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9732, "parameters": 1000, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining each tied token embedding to six learned contrast coordinates will remove one parameter per vocabulary token while retaining at least 99% accuracy, because rank-two content routing already achieved 99.93% and the independently learned relative-position pathways—not maximal content width—were the demonstrated load-bearing mechanism.
change: Replace the seven-degree token embedding with a six-degree learned representation whose final two nonreference channels share one learned coordinate; preserve the eight-dimensional residual stream, tied learned classifier, causal attention, and quotient optimization.
mechanism: Six-dimensional tied lexical bottleneck
evidence_used: Reducing query/key routing from rank three to rank two retained 99.93%, while rank one failed and relative-bias replacement failed completely. This patch therefore preserves the proven rank-two attention and positional tables while challenging the separate assumption that every vocabulary token requires all seven gauge-fixed lexical degrees.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5196000000000001, "parameters": 887, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing the removed head-1 boundary bias at one-eighth between learned endpoints will reduce the model from 1,001 to 1,000 parameters while retaining at least 99% accuracy, because the two downstream transition fractions from the 99.97%-accurate design remain unchanged.
change: Shorten head 1’s relative-bias parameter by one coordinate, insert a conservative one-eighth interpolant for the removed boundary, and preserve the successful quarter and five-eighths transition biases.
mechanism: Successor-preserving head-1 transition interpolation
evidence_used: The current quarter/five-eighths reconstruction achieved 99.97%, whereas the 1,000-parameter recursive reconstruction achieved 97.32% after changing all downstream transition fractions; preserving those successful fractions isolates compression of only the additional boundary.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5112, "parameters": 1000, "training_steps": 4999}

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
