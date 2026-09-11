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
verified_results: {"accuracy": 0.9974, "parameters": 638, "training_steps": 4999}
prior_hypothesis: Constraining each token’s learned rank-four code to a trainable radius and phase will reduce the model from 850 to 638 parameters while retaining at least 99% accuracy, because it preserves four ambient lexical channels while replacing four independent coordinates per token with two learned semantic coordinates.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing the tied token representation and output projection from rank five to rank four will remove `VOCAB_SIZE` learned parameters while retaining at least 99% accuracy, because the rank-five model achieved 99.92% and this change leaves its proven attention-routing and MLP mechanisms intact.
change: Set `token_rank` to four and retain the existing adaptive pivot-gauge factorization for the resulting four-dimensional lexical space.
mechanism: Four-dimensional tied lexical bottleneck
evidence_used: The rank-five lexical bottleneck reached 99.92% with 995 parameters, whereas altering attention scoring collapsed accuracy to 30.33%; the next informative compression is therefore another lexical-rank reduction without perturbing attention.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 881, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing the tied token representation and logit projection from rank four to rank three will remove `VOCAB_SIZE + 2` learned parameters while retaining at least 99% accuracy, because the prior rank reduction improved accuracy to 99.98% without changing attention or MLP computation.
change: Set `token_rank` to three and retain the adaptive pivot-gauge factorization for the resulting three-dimensional lexical space.
mechanism: Three-dimensional tied lexical bottleneck
evidence_used: The rank-five model achieved 99.92% with 995 parameters, and the rank-four model achieved 99.98% with 881 parameters; this supports testing the next lexical-rank reduction while preserving the proven attention-routing structure.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7928000000000001, "parameters": 765, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the rank-four model from 881 to 880 learned parameters while retaining at least 99% accuracy, because the rank-four model achieved 99.98% and the adjacent twenty-second-distance tie previously achieved 99.91%.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the removed distance from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The four-dimensional lexical bottleneck reached 99.98% with 881 parameters, while nine consecutive adjacent cross-head bias ties through the twenty-second distance preserved the accuracy requirement; prior twenty-third-distance attempts were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the rank-four model from 881 to 880 learned parameters while retaining at least 99% accuracy, because the adjacent twenty-second-distance tie retained 99.91% accuracy and the current rank-four model has 99.98% accuracy.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the removed twenty-third-farthest distance from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: Nine consecutive cross-head distance ties through the twenty-second-farthest distance preserved the accuracy requirement; later twenty-third-distance submissions were unverifiable and therefore provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the rank-four model from 881 to 880 learned parameters while retaining at least 99% accuracy.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the removed distance from one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias sharing
evidence_used: The rank-four model achieved 99.98% accuracy, and nine consecutive adjacent cross-head bias ties through the twenty-second-farthest distance preserved at least 99%; the prior twenty-third-distance submissions were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Reducing `d_ff` from 11 to 10 will lower the model from 881 to 866 learned parameters while retaining at least 99% accuracy.
change: Remove one hidden unit from the transformer MLP without changing the proven rank-four lexical representation or attention routing.
mechanism: One-unit MLP width reduction
evidence_used: The rank-four model achieved 99.98% accuracy, while reducing lexical rank collapsed accuracy to 79.28%; testing a single-unit MLP reduction is therefore the smallest informative compression along a different capacity axis.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9922, "parameters": 866, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the model from 866 to 865 learned parameters while retaining at least 99% accuracy.
change: Shorten the per-head relative-bias core by one distance and reconstruct the removed distance with one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The current 866-parameter model achieved 99.22% accuracy, and nine consecutive adjacent cross-head distance ties through the twenty-second-farthest distance previously preserved at least 99% accuracy.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the model from 866 to 865 learned parameters while retaining at least 99% accuracy.
change: Shorten the independent per-head relative-bias core by one distance and reconstruct the removed distance with one shared learned scalar.
mechanism: Cross-head twenty-third-distance bias tying
evidence_used: The 866-parameter model achieved 99.22% accuracy, and nine consecutive adjacent cross-head distance ties through the twenty-second-farthest distance previously retained at least 99%; the later attempts were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Eliminating the 16-dimensional invertible basis redundancy between the shared value map and attention output projection will reduce the model from 866 to 850 learned parameters while retaining at least 99% accuracy.
change: Adaptively select four value-map pivot columns, fix them to an identity matrix, retain only the remaining coefficients, and compensate exactly in each head’s output-projection block.
mechanism: Shared-value/output-projection change-of-basis gauge fixing
evidence_used: The adaptive pivot-gauge lexical factorization retained 99.98% accuracy at 881 parameters, while the current width-10 model retained 99.22%; this applies the same verified redundancy-removal principle without changing attention scores, initialization function, or model capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 850, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining each token’s learned rank-four code to a trainable radius and phase will reduce the model from 850 to 638 parameters while retaining at least 99% accuracy, because it preserves four ambient lexical channels while replacing four independent coordinates per token with two learned semantic coordinates.
change: Generate tied input/output token codes by harmonically lifting learned per-token phases and radii into four dimensions, restore a full learned token-to-residual projection, and exempt periodic phase coordinates from weight decay.
mechanism: Learned scalar-phase harmonic token representation
evidence_used: Rank four achieved 99.98% while rank three collapsed to 79.28%, showing that four ambient lexical channels are load-bearing. The successful 850-parameter model does not establish that every token needs four independently learned coordinates, so a two-coordinate harmonic manifold cleanly tests that shared assumption without narrowing the proven four-dimensional interface.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9974, "parameters": 638, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one token’s phase as the global angular reference will reduce the model from 638 to 637 learned parameters while retaining at least 99% accuracy, because common phase shifts are exactly absorbable by rotations of the learned four-dimensional token projection.
change: Preserve the first token’s initialized phase as a fixed buffer and learn only the remaining relative token phases, without narrowing the rank-four lexical interface or changing the initialized function.
mechanism: Harmonic phase gauge fixing
evidence_used: The learned scalar-phase harmonic representation achieved 99.74% accuracy with 638 parameters, while rank three collapsed to 79.28%; this removes a redundant coordinate from the successful four-dimensional representation rather than reducing its load-bearing ambient rank.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.048, "parameters": 637, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the twenty-third-farthest head-specific relative biases with one shared learned scalar will reduce the verified 638-parameter model to 637 parameters while retaining at least 99% accuracy.
change: Shorten the per-head relative-bias core by one distance and explicitly reconstruct the removed distance from a shared trainable scalar.
mechanism: Cross-head twenty-third-distance bias sharing
evidence_used: The current harmonic-token model achieved 99.74% accuracy, and nine consecutive adjacent cross-head distance ties through the twenty-second-farthest distance retained at least 99%; the prior twenty-third-distance submissions were unverifiable and provide no contrary accuracy evidence.
result: the implementation could not be verified



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
