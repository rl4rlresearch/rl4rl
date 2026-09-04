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
verified_results: {"accuracy": 1.0, "parameters": 1552, "training_steps": 4999}
prior_hypothesis: Removing a fourth `ln1` bias coordinate will reduce the model from 1,553 to 1,552 parameters while retaining at least 99% accuracy, because the verified three-coordinate pruning shows that downstream query and value biases can absorb these constant affine effects while the key-side effect is softmax-invariant.

## Recent verification evidence

RECENT RESULT
hypothesis: Constraining only the sixth positional coordinate to be mean-free across positions will reduce the model from 1,555 to 1,554 parameters while retaining at least 99% accuracy, because it preserves all position-dependent variation in the coordinate whose wholesale removal caused the 52.62% collapse.
change: Reparameterize the sixth positional coordinate with `INPUT_LEN - 1` orthogonal coordinates, removing only its position-independent component while preserving the original full-width initialization draw and all other behavior.
mechanism: Position-axis common-mode quotient
evidence_used: Reducing every positional vector from six to five coordinates failed at 52.62%, whereas six coordinates achieved 100%; retaining the sixth coordinate’s complete relative-position variation isolates whether only its non-positional common mode is dispensable.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1554, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fifth positional coordinate to be mean-free across positions will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because all six position-dependent coordinates remain available and only another position-independent offset is removed.
change: Reparameterize the final two positional coordinates using `INPUT_LEN - 1` orthogonal position-axis coordinates each, preserving the full-width initialization draw and all other model and training behavior.
mechanism: Second position-axis common-mode quotient
evidence_used: Removing one positional common mode achieved 99.92% at 1,554 parameters, whereas deleting an entire positional coordinate collapsed accuracy to 52.62%; this motivates removing another common mode without sacrificing any relative-position variation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3822, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Explicitly isolating the token-common content vector before removing the fifth positional common mode will produce a 1,553-parameter model with at least 99% accuracy, because this single Adam coordinate can absorb the removed position-independent offset without coordinated updates across every token row.
change: Reparameterize token content into mean-free token variation plus an explicit decay-free common vector, then constrain both final positional coordinates to be mean-free across positions.
mechanism: Optimizer-aligned token-common positional gauge
evidence_used: Removing the second positional common mode directly fell to 38.22%, while independently parameterizing the analogous token-row-mean absorption channel previously turned a failing final-bias quotient into a 100%-accurate model; this motivates changing the optimizer coordinates rather than restoring the redundant positional scalar.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7491, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` input-weight scalar will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because `ln2`’s normalized activations lie in a zero-sum hyperplane and the first neuron’s retained bias absorbs the affine offset.
change: Reparameterize the first `fc1` weight row with its final coordinate fixed at zero, while preserving initialization outputs, RNG consumption, all six learned `ln2` bias coordinates, and the successful one-pruned `fc1` output bias.
mechanism: LayerNorm zero-sum input-weight quotient
evidence_used: The 1,554-parameter design achieved 99.92%, while removing a second `fc1` bias fell to 84.48% and fixing an `ln1` scale fell to 86.89%; this motivates removing a distinct exact redundancy that retains both the downstream neuron bias and every LayerNorm scale and bias coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7334, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one attention output-projection bias coordinate will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because the full learned value bias can generate the omitted context-independent output offset through the learned projection.
change: Prune the final mean-free bias coordinate only from the attention output projection, while preserving its full weight matrix, all value-bias coordinates, initialization draws, and every other model and training setting.
mechanism: Reverse value-bias/output-bias affine quotient
evidence_used: Removing one value-projection bias coordinate fell to 79.24% at 1,577 parameters, indicating that the value-bias optimization pathway is important; this patch tests the reverse orientation of the same exact affine redundancy by retaining the full value bias and removing a downstream projection-bias coordinate instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7371, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Freezing the fifth positional coordinate’s common component at its fresh-initialization value will reduce the model from 1,554 to 1,553 learned parameters while retaining at least 99% accuracy, because it preserves the successful model’s initial positional tensor exactly while removing only that component’s subsequent optimization.
change: Split the fifth positional coordinate into a learned mean-free component and a persistent fixed common component captured from the original full-width initialization draw; retain the existing learned mean-free sixth coordinate and all other settings.
mechanism: Initialization-anchored positional common mode
evidence_used: Setting the fifth positional common mode to zero collapsed accuracy to 38.22%, whereas retaining it produced 99.92%; anchoring it to its original random initialization distinguishes a required initial offset from a required learned degree of freedom without perturbing initialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.185, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the fourth positional coordinate to be mean-free across positions will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because it preserves all six position-dependent coordinates and leaves the essential fifth-coordinate common component learned.
change: Reparameterize the fourth and sixth positional coordinates with `INPUT_LEN - 1` orthogonal position-axis coordinates, preserving the full-width initialization draw and all other model and training settings.
mechanism: Alternative-axis positional common-mode quotient
evidence_used: Removing the sixth coordinate’s common mode achieved 99.92%, while removing the fifth coordinate’s common mode fell to 38.22%; testing the fourth coordinate determines whether that failure is specific to the fifth coordinate rather than a general limit on removing a second positional common mode.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9262999999999999, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Factorizing the token-content table through a learned six-dimensional subspace will reduce the model from 1,554 to 1,482 parameters while retaining at least 99% accuracy, because the 100 operand-pair tokens need learned behavioral similarity rather than seven independent content coordinates, while token-specific output biases and the full eight-dimensional transformer state remain available.
change: Replace the independent seven-coordinate content vector for every vocabulary item with six learned token factors and a shared learned 6-by-7 mixing matrix, initialized by the best rank-six approximation of the original fresh random embedding; keep the independent mean-free token-row offsets, tied output projection, attention, positional representation, and training schedule unchanged.
mechanism: Learned low-rank vocabulary content subspace
evidence_used: The successful 1,554-parameter model still spends 798 parameters on an unconstrained per-token content table, while positional-coordinate removal failed despite preserving that table. This tests the different load-bearing assumption that every token requires seven unrelated content coordinates; learned factorization is plausible because operand-pair symbols can share latent behavior, and it preserves the row-offset channel required by the successful final-LayerNorm quotient.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Constraining the fifth token-content coordinate to be mean-free across vocabulary will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because its removed common vector is output-softmax-invariant and can be transferred exactly into the retained fifth positional common mode.
change: Reparameterize one token-content column with `VOCAB_SIZE - 1` orthogonal coordinates, transfer its initialization common mode into the corresponding positional coordinate, and optimize the new token parameter without weight decay.
mechanism: Tied-softmax token/position common-mode quotient
evidence_used: Removing the fifth positional common mode collapsed accuracy to 38.22%, indicating that its positional optimization pathway is load-bearing; this reverse quotient preserves that pathway while removing the equivalent vocabulary-common token direction instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1585, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining the third positional coordinate to be mean-free across positions will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because this coordinate lies entirely within one four-channel attention head, unlike the fourth coordinate whose cross-head quotient reached only 92.63%.
change: Reparameterize the third and sixth positional coordinates using `INPUT_LEN - 1` orthogonal position-axis coordinates, while preserving the original full-width initialization draw and all other model and training behavior.
mechanism: Head-local positional common-mode quotient
evidence_used: Removing the sixth positional common mode achieved 99.92%, proving a second position-axis parameterization is viable; removing the fourth common mode reached 92.63%, substantially outperforming fifth-coordinate removal at 38.22% and motivating a test of the adjacent, head-local third coordinate.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8059999999999999, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third `ln1` bias coordinate will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because downstream query and value biases absorb its constant affine effects while the corresponding key shift is softmax-invariant.
change: Reuse the unused LayerNorm variant as a three-coordinate-pruned LayerNorm and apply it only before attention, leaving `ln2` and all training settings unchanged.
mechanism: Third pre-attention LayerNorm bias quotient
evidence_used: The verified 1,554-parameter model already succeeds with two bias coordinates removed from `ln1`; unlike the failed `ln1` scale removal, another bias removal follows the same downstream-affine redundancy already tolerated by the successful design.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1553, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth `ln1` bias coordinate will reduce the model from 1,553 to 1,552 parameters while retaining at least 99% accuracy, because the verified three-coordinate pruning shows that downstream query and value biases can absorb these constant affine effects while the key-side effect is softmax-invariant.
change: Replace the three-coordinate-pruned LayerNorm used before attention with a four-coordinate-pruned variant, leaving `ln2`, initialization, and training unchanged.
mechanism: Fourth pre-attention LayerNorm bias quotient
evidence_used: The immediately preceding change removed a third `ln1` bias coordinate and achieved 99.91% accuracy at 1,553 parameters; extending the same successful quotient by one coordinate is the most direct incremental test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1552, "training_steps": 4999}



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
