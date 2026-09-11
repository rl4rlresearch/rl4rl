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
verified_results: {"accuracy": 0.9995, "parameters": 1635, "training_steps": 4999}
prior_hypothesis: Fixing one coordinate of the final MLP output bias at zero will reduce the model from 1,636 to 1,635 parameters while retaining at least 99% accuracy, because the omitted common-mode bias degree of freedom is exactly removed by the following final LayerNorm.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Sharing the single learned key-bias coordinate across both attention heads will reduce the model from 1,637 to 1,636 parameters while retaining at least 99% accuracy, because the duplicated coordinate remains position-independent and softmax-invariant while preserving the successful head-balanced bias layout.
change: Store one shared key-bias parameter instead of two and insert that same learned scalar as the fourth key-bias coordinate in each head.
mechanism: Head-shared key-bias anchoring
evidence_used: Six head-balanced key-bias anchors achieved 99.91% accuracy with 1,637 parameters, whereas six consecutive anchors narrowly failed at 98.84%; sharing the two remaining balanced coordinates is the smallest reduction that preserves the successful per-head geometry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1636, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the shared key-bias scalar with eight fixed zeros will reduce the model from 1,636 to 1,635 parameters while retaining at least 99% accuracy, because the scalar is initialized to zero and any position-independent key bias cancels under attention softmax.
change: Keep the combined QKV layer and its current initialization path, but store only the eight query and eight value bias coordinates and reconstruct the entire key-bias segment as fixed zeros.
mechanism: Initialization-preserving elimination of the final invariant key-bias scalar
evidence_used: The head-shared key-bias model achieved 99.92% with 1,636 parameters; unlike the failed all-at-once removal, this edit starts from the successful staged layout and preserves its registered Linear bias initialization while removing only the final functionally invariant scalar.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.45189999999999997, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the final MLP output bias at zero will reduce the model from 1,636 to 1,635 parameters while retaining at least 99% accuracy, because the omitted common-mode bias degree of freedom is exactly removed by the following final LayerNorm.
change: Replace `fc2` with an initialization-preserving linear layer that learns seven bias coordinates and appends one fixed-zero coordinate.
mechanism: Final-normalization common-mode bias anchoring
evidence_used: The current head-shared key-bias design achieved 99.92% at 1,636 parameters, while removing its last key-bias scalar collapsed accuracy; this motivates preserving attention unchanged and instead fixing an exact downstream bias gauge immediately before final normalization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1635, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the attention output-projection bias at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the omitted common-mode offset is removed by both the downstream pre-MLP LayerNorm and final LayerNorm.
change: Reuse the initialization-preserving `OutputAnchoredLinear` for the attention output projection, learning seven bias coordinates and appending one fixed-zero coordinate.
mechanism: Residual-stream common-mode bias anchoring
evidence_used: Anchoring one coordinate of the final MLP output bias achieved 99.95% accuracy at 1,635 parameters, demonstrating that removing a common-mode residual-stream bias degree of freedom before final normalization preserves performance; the attention projection has the same gauge because its common offset survives only in the residual stream while normalized downstream computations remain unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2597, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one coordinate of the final MLP output weight at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because its contribution differs from an unrestricted weight only by a token-dependent common-mode offset removed by the final LayerNorm.
change: Store 95 learned `fc2` weight coordinates, append one fixed-zero coordinate during the forward pass, and preserve the existing seven-coordinate anchored bias.
mechanism: Final-MLP weight common-mode gauge anchoring
evidence_used: Anchoring the final MLP output bias achieved 99.95% accuracy at 1,635 parameters, while applying a similar anchor inside the attention projection failed; this motivates extending the successful final-MLP gauge by the smallest possible one-coordinate reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.49829999999999997, "parameters": 1634, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` bias coordinate at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because the omitted offset can be absorbed exactly into the following `fc1` bias.
change: Replace the standard pre-MLP LayerNorm with the existing initialization-preserving `AnchoredLayerNorm`, which learns seven bias coordinates and appends one fixed zero.
mechanism: Pre-MLP LayerNorm-to-linear bias gauge anchoring
evidence_used: A one-coordinate LayerNorm bias anchor previously achieved 99.95% accuracy, and the current design also achieves 99.95%; applying the same successful reduction before `fc1` preserves initialization and removes an exact LayerNorm-linear parameter redundancy.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1353, "parameters": 1634, "training_steps": 4999}



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
