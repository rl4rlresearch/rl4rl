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
verified_results: {"accuracy": 0.9991, "parameters": 1637, "training_steps": 4999}
prior_hypothesis: Fixing six key-bias coordinates while distributing them evenly across the two attention heads will produce a 1,637-parameter model with at least 99% accuracy, because each key bias remains softmax-invariant and the balanced layout may avoid the optimization sensitivity observed with six consecutive anchors.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing only the 8-dimensional key bias will reduce parameters from 1,644 to 1,636 while retaining at least 99% accuracy, because adding the same learned key bias to every position shifts all unmasked attention logits by an equal amount and therefore cannot change the attention probabilities.
change: Replace the combined QKV bias with separate learned query and value biases, leaving the key projection unbiased while preserving `d_ff=12`.
mechanism: Eliminate softmax-invariant key bias
evidence_used: The 1,644-parameter `d_ff=12` model reached 99.96%, while reducing `d_ff` to 11 collapsed accuracy to 74.06%; this motivates preserving demonstrated feed-forward capacity and removing a mathematically redundant attention parameter instead.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8187000000000001, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the first LayerNorm bias at zero will reduce the model from 1,644 to 1,643 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offset.
change: Add a LayerNorm equivalent with seven learned bias coordinates and one fixed-zero coordinate, and use it only before self-attention.
mechanism: One-coordinate pre-attention bias anchoring
evidence_used: The 1,644-parameter model reached 99.96%, while larger structural reductions collapsed accuracy; this motivates the smallest possible reduction in a bias that is representationally redundant with the following affine projection.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1643, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second coordinate of the first LayerNorm bias at zero will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offsets.
change: Store six learned bias coordinates in the pre-attention LayerNorm and append two fixed-zero coordinates.
mechanism: Incremental pre-attention bias anchoring
evidence_used: Fixing one LayerNorm bias coordinate reduced the model to 1,643 parameters and achieved 99.95% accuracy, while larger structural reductions failed; this supports testing the smallest incremental extension of the successful redundancy removal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.47759999999999997, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing all eight key-bias parameters while preserving the successful model’s RNG consumption and shared QKV weight initialization will produce a 1,635-parameter model with at least 99% accuracy.
change: Retain the combined QKV projection, discard its registered bias after construction, and learn only query and value biases while inserting a fixed-zero key bias.
mechanism: Initialization-preserving elimination of softmax-invariant key bias
evidence_used: The earlier 1,636-parameter key-bias removal changed the projection structure and achieved only 81.87%, while the current anchored model achieved 99.95%; this patch tests the same exact attention redundancy without perturbing initialization of the functionally relevant weights.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2205, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Anchoring one bias coordinate in the pre-MLP LayerNorm will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because the following learned `fc1` bias can absorb the omitted constant offset without further constraining the already-sensitive pre-attention normalization.
change: Reuse `AnchoredLayerNorm` for `ln2`, preserving all attention and feed-forward widths.
mechanism: Distributed LayerNorm bias anchoring
evidence_used: One-coordinate pre-attention anchoring achieved 99.95% with 1,643 parameters, while anchoring a second coordinate in that same normalization collapsed accuracy; distributing the next one-coordinate reduction to the independently absorbable pre-MLP bias is the smallest informative alternative.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3961, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one key-bias coordinate at zero will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because a position-independent key offset only adds a common shift to each query’s attention logits.
change: Replace the combined QKV linear layer with an initialization-preserving subclass that learns 23 bias coordinates and inserts one fixed-zero coordinate in the key-bias segment.
mechanism: Single-coordinate softmax-invariant key-bias anchoring
evidence_used: The one-coordinate LayerNorm anchor achieved 99.95% at 1,643 parameters, while removing all key-bias coordinates collapsed accuracy; an incremental one-coordinate key anchor tests the exact attention redundancy with the smallest possible optimization perturbation.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1642, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a second key-bias coordinate at zero will reduce the model from 1,642 to 1,641 parameters while retaining at least 99% accuracy, because every position-independent key-bias coordinate contributes only a common shift to each query’s attention logits.
change: Store 22 learned QKV bias coordinates and insert two fixed-zero coordinates at the start of the key-bias segment.
mechanism: Incremental softmax-invariant key-bias anchoring
evidence_used: One key-bias anchor achieved 99.98% accuracy with 1,642 parameters, while removing all key biases at once failed; this motivates the smallest incremental extension of the successful exact redundancy removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9962000000000001, "parameters": 1641, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a third key-bias coordinate at zero will reduce the model from 1,641 to 1,640 parameters while retaining at least 99% accuracy, because each position-independent key-bias coordinate adds only a common shift to a query’s attention logits.
change: Store 21 learned QKV bias coordinates and insert three fixed-zero coordinates at the start of the key-bias segment.
mechanism: Incremental softmax-invariant key-bias anchoring
evidence_used: Fixing two key-bias coordinates achieved 99.62% accuracy with 1,641 parameters, after one coordinate achieved 99.98%; since removing all key biases at once failed, the smallest incremental extension is the most informative next test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1640, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fourth key-bias coordinate at zero will reduce the model from 1,640 to 1,639 parameters while retaining at least 99% accuracy, because every position-independent key-bias coordinate contributes only a common shift to each query’s attention logits.
change: Store 20 learned QKV bias coordinates and insert four fixed-zero coordinates at the start of the key-bias segment.
mechanism: Incremental softmax-invariant key-bias anchoring
evidence_used: Fixing three key-bias coordinates achieved 99.92% accuracy with 1,640 parameters, following successful one- and two-coordinate anchors; this motivates the smallest incremental extension of the demonstrated redundancy removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1639, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a fifth key-bias coordinate at zero will reduce the model from 1,639 to 1,638 parameters while retaining at least 99% accuracy, because every position-independent key-bias coordinate contributes only a common shift to each query’s attention logits.
change: Store 19 learned QKV bias coordinates and insert five fixed-zero coordinates at the start of the key-bias segment.
mechanism: Incremental softmax-invariant key-bias anchoring
evidence_used: Fixing four key-bias coordinates achieved 99.93% accuracy with 1,639 parameters, following successful one-, two-, and three-coordinate anchors; this motivates the smallest incremental extension of the demonstrated redundancy removal.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1638, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing a sixth key-bias coordinate at zero will reduce the model from 1,638 to 1,637 parameters while retaining at least 99% accuracy, because every position-independent key-bias coordinate contributes only a common shift to each query’s attention logits.
change: Store 18 learned QKV bias coordinates and insert six fixed-zero coordinates at the start of the key-bias segment.
mechanism: Incremental softmax-invariant key-bias anchoring
evidence_used: Fixing five key-bias coordinates achieved 99.90% accuracy with 1,638 parameters, extending successful one-through-four-coordinate anchors; the smallest incremental extension is therefore the most informative next change.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9884000000000001, "parameters": 1637, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing six key-bias coordinates while distributing them evenly across the two attention heads will produce a 1,637-parameter model with at least 99% accuracy, because each key bias remains softmax-invariant and the balanced layout may avoid the optimization sensitivity observed with six consecutive anchors.
change: Store 18 QKV bias parameters and reconstruct each four-dimensional key bias with three fixed-zero coordinates and one learned coordinate.
mechanism: Head-balanced key-bias anchoring
evidence_used: Five consecutive key anchors achieved 99.90% accuracy at 1,638 parameters, while six consecutive anchors narrowly missed at 98.84%; balancing the same six exact-redundancy removals across heads tests whether that miss arose from numerical or optimization sensitivity rather than lost capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1637, "training_steps": 4999}



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
