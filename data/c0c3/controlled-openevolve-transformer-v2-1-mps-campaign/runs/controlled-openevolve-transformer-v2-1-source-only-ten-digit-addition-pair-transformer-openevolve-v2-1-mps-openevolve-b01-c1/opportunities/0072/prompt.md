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
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1547, "training_steps": 4999}
prior_hypothesis: Replacing the three remaining `ln2` bias parameters with the restored twelfth `fc1` bias will reduce the model from 1,549 to 1,547 parameters while retaining at least 99% accuracy, because `fc1` can absorb every constant contribution of the preceding LayerNorm bias.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing a fifth `ln1` bias coordinate will reduce the model from 1,552 to 1,551 parameters while retaining at least 99% accuracy, because four consecutive coordinates were pruned successfully and downstream query/value biases can absorb the omitted affine effect while its key-side shift is softmax-invariant.
change: Replace the four-coordinate-pruned LayerNorm before attention with a five-coordinate-pruned variant, leaving initialization, `ln2`, and training unchanged.
mechanism: Fifth pre-attention LayerNorm bias quotient
evidence_used: Extending `ln1` bias pruning from three to four coordinates improved the verified result from 99.91% at 1,553 parameters to 100% at 1,552 parameters, directly motivating one further incremental quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7304, "parameters": 1551, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the common mode of the four currently learned `ln1` bias coordinates will reduce the model to 1,551 parameters while retaining at least 99% accuracy, because it preserves three relative directions spanning all four successful coordinates while downstream query/value biases absorb the omitted constant affine effect.
change: Reparameterize the four learned `ln1` bias coordinates with a three-dimensional orthonormal mean-free basis instead of fixing an additional coordinate.
mechanism: Balanced pre-attention LayerNorm bias quotient
evidence_used: Four-coordinate pruning achieved 100% at 1,552 parameters, whereas directly fixing a fifth coordinate fell to 73.04%; preserving variation across all four previously learned coordinates tests whether that failure was caused by the coordinate-aligned parameterization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7384999999999999, "parameters": 1551, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a third `ln2` bias coordinate will reduce the model from 1,552 to 1,551 parameters while retaining at least 99% accuracy, because the current 100%-accurate model already removes two such coordinates and the following `fc1` bias can absorb the additional constant affine effect.
change: Replace the two-coordinate-pruned LayerNorm before the MLP with a three-coordinate-pruned variant, leaving attention, initialization, and training unchanged.
mechanism: Third pre-MLP LayerNorm bias quotient
evidence_used: The current 1,552-parameter design achieves 100% accuracy while pruning two `ln2` bias coordinates; unlike the failed fifth `ln1` pruning, this tests an unextended downstream-affine quotient with a retained learned `fc1` bias pathway.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1551, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fourth `ln2` bias coordinate will reduce the model from 1,551 to 1,550 parameters while retaining at least 99% accuracy, because the following learned `fc1` bias can absorb the omitted constant affine effect.
change: Extend pre-MLP LayerNorm bias pruning from three coordinates to four, leaving attention, initialization, and training unchanged.
mechanism: Fourth pre-MLP LayerNorm bias quotient
evidence_used: Removing the third `ln2` bias coordinate achieved 99.94% accuracy at 1,551 parameters, making one further incremental extension of this still-successful quotient the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1550, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing a fifth `ln2` bias coordinate will reduce the model from 1,550 to 1,549 parameters while retaining at least 99% accuracy, because the following learned `fc1` bias can absorb its constant affine contribution.
change: Extend pre-MLP LayerNorm bias pruning from four coordinates to five while leaving attention, initialization, and training unchanged.
mechanism: Fifth pre-MLP LayerNorm bias quotient
evidence_used: Removing the fourth `ln2` bias coordinate retained 99.98% accuracy at 1,550 parameters; unlike the failed fifth `ln1` pruning, this pathway is immediately followed by the learned `fc1` bias that motivated the successful `ln2` reductions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1549, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing head-specific key/value projections with one learned shared key/value projection will reduce the model from 1,549 to 1,481 parameters while retaining at least 99% accuracy, because independent learned queries still let the two heads retrieve different positions while the retrieved operand-pair representation can be shared.
change: Convert causal self-attention to multi-query attention: retain two independent query heads, share one learned four-dimensional key/value memory across them, and preserve the original initialization distribution and RNG sequence.
mechanism: Query-routed shared attention memory
evidence_used: The 1,549-parameter model achieves 99.98% accuracy, while failures from pruning positional or normalization coordinates show that representation channels are load-bearing. This instead challenges the untested assumption that both heads need separate key/value feature maps; distinct query projections and output slots remain intact.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0664, "parameters": 1481, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the three remaining `ln2` bias parameters with the restored twelfth `fc1` bias will reduce the model from 1,549 to 1,547 parameters while retaining at least 99% accuracy, because `fc1` can absorb every constant contribution of the preceding LayerNorm bias.
change: Make pre-MLP LayerNorm bias-free and restore a conventional fully biased `fc1`, eliminating two redundant learned coordinates without reducing the attainable MLP preactivation-bias space.
mechanism: Complete pre-MLP affine-bias quotient
evidence_used: Five consecutive `ln2` bias coordinates were removed while retaining 99.98% accuracy at 1,549 parameters; unlike the failed attention-side pruning, this LayerNorm is immediately followed by a learned affine layer whose bias can represent the omitted contribution directly.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1547, "training_steps": 4999}



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
