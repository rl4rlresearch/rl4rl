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
verified_results: {"accuracy": 0.9968, "parameters": 1040, "training_steps": 4999}
prior_hypothesis: Sharing head 1’s fourth- through seventh-farthest relative biases will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy, because the adjacent-boundary triplet achieved 99.81% and this removes only one neighboring positional degree of freedom.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the seven-parameter MLP residual bias will reduce the model from 1,053 to 1,046 parameters while retaining at least 99% accuracy, because the width-1 model achieved 100% and the failed bias-only limit indicates the nonlinear neuron—not the standalone bias—is the load-bearing component.
change: Delete the learned residual bias from the final one-neuron MLP while preserving its learned GELU input and output projections.
mechanism: Complementary nonlinear-branch isolation
evidence_used: The width-1 MLP reached 100% accuracy, whereas retaining only its residual bias collapsed accuracy to 23.29%; the complementary ablation directly tests whether that bias contributes anything when the necessary nonlinear pathway remains intact.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1046, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the final MLP neuron’s scalar input bias will reduce the model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because the nonlinear pathway is necessary but the latest successful result shows its seven-parameter residual bias is not.
change: Make the width-one MLP input projection bias-free while consuming the removed constructor draw so every remaining parameter keeps the current initialization stream.
mechanism: Bias-free nonlinear bottleneck
evidence_used: The width-one MLP achieved 100% accuracy, the bias-only replacement collapsed to 23.29%, and removing the nonlinear branch’s residual bias still achieved 99.95%; this isolates the only remaining scalar bias in the load-bearing nonlinear unit with the smallest possible ablation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.37810000000000005, "parameters": 1045, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the seven learned final LayerNorm bias coordinates will reduce the model from 1,046 to 1,039 parameters while retaining at least 99% accuracy, because the successful residual-bias ablation indicates constant feature offsets are dispensable, while the failed MLP input-bias ablation specifically preserves evidence that offsets before the nonlinear bottleneck are essential.
change: Make the final LayerNorm bias-free and remove its deleted bias from the quotient optimizer.
mechanism: Bias-free final normalization
evidence_used: Removing the seven-parameter MLP residual bias retained 99.95% accuracy, whereas removing the scalar MLP input bias collapsed accuracy to 37.81%; this motivates testing another post-computation feature offset without disturbing the load-bearing nonlinear bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1564, "parameters": 1039, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying one final LayerNorm bias coordinate to its common-mode reference will reduce the model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because it preserves six learned feature offsets rather than deleting the entire essential bias pathway.
change: Store six final LayerNorm bias coordinates and pad two reference-valued coordinates; the existing quotient optimizer automatically trains the resulting seven-coordinate virtual bias representation.
mechanism: Single-coordinate final-normalization bias tie
evidence_used: Removing all seven final LayerNorm bias coordinates collapsed accuracy to 15.64%, while removing the seven-parameter MLP residual bias retained 99.95%; this makes a one-coordinate final-bias ablation the most informative test of whether the essential offset pathway contains any residual redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1967, "parameters": 1045, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying each head’s two farthest relative-distance biases will reduce the model from 1,046 to 1,044 parameters while retaining at least 99% accuracy, because it preserves independent, otherwise unconstrained positional tables and alters only the two sparsest causal distances.
change: Store two fewer relative-bias coordinates and reconstruct both farthest distances from each head’s existing omitted reference; the quotient optimizer continues to train the resulting per-head virtual reference automatically.
mechanism: Per-head farthest-distance bias tying
evidence_used: Independent affine positional pointers failed at 0% accuracy, showing that aggressive table compression loses essential positional structure, while the current independent-table model achieved 99.95%; a single boundary tie per head is the smallest structured positional ablation that preserves head independence.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9979, "parameters": 1044, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying each head’s three farthest relative-distance biases will reduce the model from 1,044 to 1,042 parameters while retaining at least 99% accuracy, because tying the two farthest distances already achieved 99.79% and this changes only one additional sparse distance per head.
change: Store one fewer relative-bias coordinate per head and reconstruct the three farthest distances from each head’s existing common-mode reference.
mechanism: Per-head three-distance boundary tying
evidence_used: The current two-distance tie retained 99.79% accuracy at 1,044 parameters, while aggressive affine compression failed completely; the smallest further positional-table ablation is therefore one additional boundary tie per head.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.19010000000000002, "parameters": 1042, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the third-farthest relative-distance bias in only one attention head will reduce the model from 1,044 to 1,043 parameters while retaining at least 99% accuracy, because the other head retains the positional degree of freedom whose removal from both heads caused failure.
change: Give the two heads separate relative-bias parameters, retaining the current two-distance tie for head 0 while tying three farthest distances for head 1; update reconstruction and quotient optimization accordingly.
mechanism: Asymmetric third-distance boundary tie
evidence_used: Tying two farthest distances in both heads achieved 99.79%, whereas tying three in both collapsed to 19.01%; an asymmetric one-parameter ablation directly tests whether only one head needs the additional boundary coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1043, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the fourth-farthest relative-distance bias only in head 1 will reduce the model from 1,043 to 1,042 parameters while retaining at least 99% accuracy, because head 1 already tolerated the third-distance tie while head 0 retained the positional coordinate whose removal from both heads caused failure.
change: Extend head 1’s boundary tie from its three farthest relative distances to its four farthest distances, leaving head 0’s successful two-distance tie and all other model, optimization, checkpointing, and decoding behavior unchanged.
mechanism: Asymmetric fourth-distance boundary tie
evidence_used: Tying three farthest distances in both heads collapsed accuracy to 19.01%, but tying the third-farthest distance only in head 1 achieved 99.73% at 1,043 parameters; this isolates another one-parameter ablation in the head that has demonstrated greater boundary redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1436, "parameters": 1042, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing head 1’s fourth- and fifth-farthest learned relative biases will reduce the model from 1,043 to 1,042 parameters while retaining at least 99% accuracy, because it preserves an adaptive boundary value instead of forcing the load-bearing fourth-farthest bias to the softmax reference.
change: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct its fourth-farthest bias from the neighboring fifth-farthest learned value; leave head 0 and all other behavior unchanged.
mechanism: Learned adjacent-boundary bias sharing
evidence_used: Head 1 tolerated tying its three farthest biases at 99.73%, but tying the fourth directly to the reference collapsed accuracy to 14.36%; adjacent sharing tests whether that distance needs a learned value without requiring an independent parameter.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9937999999999999, "parameters": 1042, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing each head’s learned query/key score rank from four to three will lower the model from 1,042 to 1,026 parameters while retaining at least 99% accuracy, because the evidence identifies flexible independent relative-position tables—not full-rank content scores—as load-bearing.
change: Decouple attention-score width from value-head width, retaining four-dimensional learned values but using three-dimensional learned query/key factors and the corresponding scale.
mechanism: Decoupled three-dimensional content routing with four-dimensional value heads
evidence_used: Replacing the independent relative-bias tables with affine positional pointers failed at 0%, while the current independent-table model achieved 99.38%; preserving those tables and compressing only the orthogonal content-routing pathway cleanly tests the shared assumption that standard attention must couple score rank to value width.
result: the implementation could not be verified

RECENT RESULT
hypothesis: Sharing head 1’s fourth-, fifth-, and sixth-farthest learned relative biases will reduce the model from 1,042 to 1,041 parameters while retaining at least 99% accuracy, because sharing the fourth and fifth biases already achieved 99.38%, whereas forcing the fourth-farthest bias to the fixed reference failed.
change: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct its two following boundary biases from the final learned value; leave head 0 and all other behavior unchanged.
mechanism: Learned adjacent-boundary bias triplet sharing
evidence_used: The 1,042-parameter adjacent-sharing design achieved 99.38% accuracy after the direct fourth-distance reference tie collapsed to 14.36%, indicating that this boundary region needs an adaptive value but may not need independent neighboring coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1041, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing head 1’s fourth- through seventh-farthest relative biases will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy, because the adjacent-boundary triplet achieved 99.81% and this removes only one neighboring positional degree of freedom.
change: Shorten head 1’s relative-bias parameter by one coordinate and extend its final learned boundary value across three following distances, leaving head 0 and all other behavior unchanged.
mechanism: Learned adjacent-boundary bias quadruplet sharing
evidence_used: Sharing head 1’s fourth-, fifth-, and sixth-farthest biases achieved 99.81% at 1,041 parameters, substantially above the threshold; extending that successful learned sharing pattern by one adjacent distance is the smallest informative ablation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9968, "parameters": 1040, "training_steps": 4999}



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
