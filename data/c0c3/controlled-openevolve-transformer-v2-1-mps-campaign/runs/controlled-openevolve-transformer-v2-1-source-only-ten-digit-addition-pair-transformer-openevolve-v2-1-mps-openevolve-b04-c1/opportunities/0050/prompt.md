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
verified_results: {"accuracy": 0.9995, "parameters": 1596, "training_steps": 4999}
prior_hypothesis: Initialization-preserving quotienting of the seventh `fc2` input column will reduce the model from 1597 to 1596 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining one attention-output projection column, while retaining the two successful `fc2` quotients, will achieve at least 99% accuracy with 1602 parameters because its omitted all-ones component is erased by subsequent LayerNorms without further perturbing MLP-channel optimization.
change: Generalize `OneColumnGaugeLinear` to accept a configurable number of quotient columns and apply a single-column quotient to the attention output projection.
mechanism: Distributed residual-stream shift quotient
evidence_used: Two quotiented `fc2` columns achieved 99.98% accuracy at 1603 parameters, while adding a third collapsed accuracy to 28.45%; distributing the next exact residual-shift quotient to the attention branch tests whether the failure was specific to accumulating constraints in the MLP.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9115000000000001, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Reparameterizing a third `fc2` column while exactly preserving the successful 1603-parameter model’s initialization stream and initial function modulo final-LayerNorm shifts will achieve at least 99% accuracy with 1602 parameters.
change: Split the first dense remainder column into a third Householder zero-mean coordinate row, initialize it from the same conceptual dense weight, and preserve all random draws used by the successful two-column design.
mechanism: Initialization-preserving third MLP column quotient
evidence_used: Two quotiented `fc2` columns achieved 99.98% accuracy, whereas the naive third quotient collapsed to 28.45%; because this is an exact final-residual symmetry, controlling the otherwise changed seeded initialization isolates the added quotient’s optimization effect.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9612, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the learned causal transformer block for a gated second refinement step will maintain at least 99% accuracy with 1587 parameters, because recurrent depth can recover the nonlinear and cross-token capacity lost by narrowing the MLP from 12 to 11 channels.
change: Narrow the MLP by one channel and add a single learned, zero-initialized gate that lets the same attention-and-MLP block iteratively refine its first-pass state without duplicating weights.
mechanism: Gated recurrent transformer refinement
evidence_used: The seven-channel one-pass SwiGLU fell to 53.57%, indicating that simply compressing the token-local mixer removes load-bearing capacity; meanwhile repeated exact gauge reductions became optimization-fragile at 1602 parameters. This challenges the shared single-pass-width assumption by trading one MLP channel for parameter-shared computational depth while preserving the successful 1603-parameter model at initialization up to the modest width change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.24969999999999998, "parameters": 1587, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the last `fc2` input column instead of the optimization-sensitive third column will reduce the model from 1603 to 1602 parameters while retaining at least 99% accuracy, because the removed all-ones output component is erased exactly by the final LayerNorm and initialization is preserved modulo that symmetry.
change: Keep the two successful leading `fc2` quotient columns, parameterize the final column in the same zero-mean basis, and reproduce the successful 1603-parameter design’s conceptual dense initialization and random-number stream.
mechanism: Initialization-preserving last-channel MLP quotient
evidence_used: Two leading quotients achieved 99.98% accuracy, while quotienting the adjacent third column reached only 96.12% even with initialization preservation; testing a different hidden channel isolates whether the remaining failure is channel-specific optimization sensitivity rather than loss of functional capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1602, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the penultimate `fc2` input column alongside the two leading and final columns will retain at least 99% accuracy while reducing the model from 1602 to 1601 parameters, because this column shares the successful trailing-edge initialization context while its removed all-ones output component is erased exactly by the final LayerNorm.
change: Replace the dense penultimate `fc2` column with a fourth Householder zero-mean column, preserving the conceptual dense initialization and random-number stream of the verified 1602-parameter design.
mechanism: Initialization-preserving penultimate-channel MLP quotient
evidence_used: The initialization-preserving final-column quotient achieved 99.99% accuracy at 1602 parameters, whereas quotienting the adjacent third column reached only 96.12%; extending inward from the successful trailing edge is the most direct test of the observed channel-specific optimization sensitivity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.354, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the attention value bias from three learned coordinates to two will lower the model from 1602 to 1601 parameters while retaining at least 99% accuracy, because attention rows sum to one and the omitted value-bias effect can be absorbed by the seven-coordinate output-projection bias modulo residual-stream shifts.
change: Remove the third learned value-bias coordinate and increase its fixed-zero padding accordingly, leaving initialization, architecture, and training unchanged.
mechanism: Redundant value-bias elimination
evidence_used: The verified 1602-parameter design reached 99.99%, while further `fc2` quotienting was optimization-sensitive; this instead removes one parameter from a different branch where the projection bias already spans every residual-relevant constant output direction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7121999999999999, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Quotienting the penultimate `fc2` column while updating it through eight-dimensional AdamW moments will retain at least 99% accuracy with 1601 parameters, because it preserves the successful 1602-parameter model’s dense-column optimization trajectory modulo the exact final-LayerNorm shift symmetry.
change: Represent the penultimate `fc2` column with seven Householder coordinates and optimize those coordinates using projected updates from ambient eight-dimensional AdamW state.
mechanism: Ambient-state AdamW for a quotiented MLP channel
evidence_used: The same penultimate-column quotient fell to 35.4% under coordinate-wise AdamW, while the dense penultimate column in the 1602-parameter design reached 99.99%; this directly tests whether Adam’s lack of rotation invariance, rather than functional capacity, caused that failure.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1601, "training_steps": 4999}

RECENT RESULT
hypothesis: Applying ambient eight-dimensional AdamW to an initialization-preserving quotient of the third `fc2` input column will achieve at least 99% accuracy with 1600 parameters.
change: Quotient the third `fc2` column while preserving conceptual dense initialization, then optimize both it and the penultimate quotiented column with ambient-state AdamW.
mechanism: Ambient-state AdamW for a fifth quotiented MLP channel
evidence_used: Ambient-state AdamW raised the penultimate-column quotient from 35.4% to 99.97%; the initialization-preserving third-column quotient already reached 96.12% with coordinate-wise AdamW, making optimizer geometry the most directly supported next variable.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1600, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the fourth `fc2` input column will reduce the model from 1600 to 1599 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the first remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and add it to the ambient-state optimizer.
mechanism: Ambient-state AdamW for a sixth quotiented MLP channel
evidence_used: Ambient AdamW rescued the penultimate-column quotient from 35.4% to 99.97% and the third-column quotient from 96.12% to 99.93%; this directly supports applying the same optimizer geometry to the adjacent fourth column.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1599, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the fifth `fc2` input column will reduce the model from 1599 to 1598 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the next remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and add its coordinates to the ambient-state optimizer.
mechanism: Ambient-state AdamW for a seventh quotiented MLP channel
evidence_used: Ambient AdamW enabled the third, fourth, and penultimate column quotients to reach at least 99.93% accuracy, and the adjacent fourth-column extension achieved 100% accuracy at 1599 parameters; quotienting the fifth column is the closest evidence-backed continuation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1598, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the sixth `fc2` input column will reduce the model from 1598 to 1597 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the first remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and include its coordinates in the ambient-state optimizer.
mechanism: Ambient-state AdamW for an eighth quotiented MLP channel
evidence_used: Ambient AdamW enabled the third, fourth, fifth, and penultimate column quotients to meet the accuracy threshold, and the adjacent fifth-column extension achieved 100% accuracy at 1598 parameters; quotienting the sixth column is the closest evidence-backed continuation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1597, "training_steps": 4999}

RECENT RESULT
hypothesis: Initialization-preserving quotienting of the seventh `fc2` input column will reduce the model from 1597 to 1596 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.
change: Convert the first remaining dense `fc2` column into a Householder zero-mean column, preserve the conceptual dense initialization and random stream, and include its coordinates in the ambient-state optimizer.
mechanism: Ambient-state AdamW for a ninth quotiented MLP channel
evidence_used: The adjacent sixth-column extension achieved 100% accuracy at 1597 parameters, while ambient AdamW has enabled every tested initialization-preserving quotient from the third through sixth columns to meet the threshold.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1596, "training_steps": 4999}

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
