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

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"accuracy": 0.9998, "parameters": 1593, "training_steps": 4999}
prior_hypothesis: Extending the qualified value-bias quotient from one to two omitted coordinates will produce a 1593-parameter model with at least 99% accuracy while preserving both omitted coordinates’ clipping and AdamW dynamics.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9995999999999999, "parameters": 1644, "training_steps": 4999}
prior_hypothesis: starting design

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9994, "parameters": 1587, "training_steps": 4999}
prior_hypothesis: Omitting the final retained value-bias degree will produce a 1587-parameter model with at least 99% accuracy because all eight value-bias coordinates have the same projection-bias-absorbable role and the existing optimizer preserves their full-coordinate updates.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9981, "parameters": 1588, "training_steps": 4999}
prior_hypothesis: Retaining value-bias coordinate 1 while absorbing coordinates 0 and 2–7 will produce a 1588-parameter model with at least 99% accuracy, showing whether the failed prior reduction was caused by discarding coordinate 1 rather than by the seven-coordinate quotient itself.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the qualified positional-plus-MLP quotient design with removal of the tied token embedding’s single globally uniform degree will produce a 1611-parameter model with at least 99% accuracy.
change: Compress each positional row and the tied token embedding into relative coordinates, reconstruct them for embedding and logit computation, and preserve their full-coordinate clipping and AdamW difference dynamics alongside the MLP bias quotient.
mechanism: Global tied-embedding quotient under pre-LayerNorm and logit-shift invariance
evidence_used: The 1612-parameter positional-plus-MLP quotient design achieved 99.82%. The failed 1611 design additionally quotienting the attention-output bias indicates that bias interaction should remain untouched; this patch instead removes a distinct exact gauge while retaining the full attention projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9951000000000001, "parameters": 1611, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing both block LayerNorm bias vectors will produce a 1595-parameter model with at least 99% accuracy, because their affine offsets can be represented by the existing QKV and `fc1` biases without reducing the learned model’s function class.
change: Disable the eight-parameter bias in `ln1` and `ln2` while retaining their learned scale parameters and all downstream projection biases.
mechanism: Downstream projection-bias absorption of pre-LayerNorm offsets
evidence_used: The current 1611-parameter quotient design achieved 99.51% accuracy, and prior successful reductions show that exact redundant degrees can be removed while retaining accuracy. LayerNorm biases initialize to zero, while `qkv` and `fc1` already provide the downstream offsets needed to absorb them.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9942, "parameters": 1595, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified 1595-parameter design by fixing one `ln2` scale coordinate at unity will produce a 1594-parameter model with at least 99% accuracy, because the omitted common scale can be absorbed by `fc1` while preserving the initialized forward function.
change: Adopt the qualified tied-token quotient and biasless block LayerNorms, then store seven learned `ln2` scale coordinates, reconstruct the eighth as one, and retain quotient-aware optimization for token, positional, and MLP-output parameters.
mechanism: One-coordinate pre-LayerNorm scale gauge
evidence_used: The 1595-parameter design achieved 99.42% accuracy after token quotienting and removal of both block LayerNorm biases; fixing one `ln2` scale coordinate is the smallest next reduction and exploits its exact multiplicative redundancy with the following learned `fc1` matrix.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2896, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1595-parameter design, removing one attention value-bias coordinate and projecting its full-coordinate AdamW update into `attn.proj.bias` will produce a 1594-parameter model with at least 99% accuracy.
change: Adopt the qualified token, positional, MLP-output, key-bias, and biasless-block-LayerNorm reductions; then store seven value-bias coordinates and preserve the omitted eighth coordinate’s gradient clipping and AdamW effect through the downstream attention projection bias.
mechanism: Quotient-aware value-bias absorption into the attention output bias
evidence_used: The 1595-parameter design achieved 99.42% after downstream biases successfully absorbed both block LayerNorm offsets, while trajectory-preserving quotient optimization qualified for positional, token, and MLP-output gauges; an attention value bias is likewise passed unchanged through normalized attention and can be absorbed by the existing projection bias.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1594, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified value-bias quotient from one to two omitted coordinates will produce a 1593-parameter model with at least 99% accuracy while preserving both omitted coordinates’ clipping and AdamW dynamics.
change: Store six value-bias coordinates, reconstruct two zero-gauge coordinates, and generalize the optimizer and gradient clipping to track and absorb both omitted updates.
mechanism: Multi-coordinate value-bias absorption into the attention projection bias
evidence_used: The current 1594-parameter design achieved 99.73% after one value-bias coordinate was absorbed into `attn.proj.bias`; the second coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1593, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified two-coordinate value-bias quotient to three omitted coordinates will produce a 1592-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the newly omitted coordinate.
change: Store five learned value-bias coordinates, reconstruct three zero-gauge coordinates, and absorb their independently tracked AdamW updates into `attn.proj.bias`.
mechanism: Three-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting one value-bias coordinate achieved 99.73% at 1594 parameters, and omitting two achieved 99.98% at 1593; the third coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1592, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified three-coordinate value-bias quotient to four omitted coordinates will produce a 1591-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Adopt the qualified token, positional, MLP-output, biasless-LayerNorm, and key-bias reductions; store four fewer value-bias coordinates and absorb their tracked updates into `attn.proj.bias`.
mechanism: Four-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting one, two, and three value-bias coordinates achieved 99.73%, 99.98%, and 100% accuracy respectively; the fourth coordinate has the same attention-invariant role as the three verified omissions.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9992, "parameters": 1591, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified four-coordinate value-bias quotient to five omitted coordinates will produce a 1590-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Store three learned value-bias coordinates, reconstruct five zero-gauge coordinates, and absorb all five independently tracked AdamW updates into `attn.proj.bias`.
mechanism: Five-coordinate value-bias absorption into the attention projection bias
evidence_used: Successive omission of one, two, three, and four value-bias coordinates achieved 99.73%, 99.98%, 100%, and 99.92% accuracy respectively; the fifth coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1590, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified five-coordinate value-bias quotient to six omitted coordinates will produce a 1589-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Store two learned value-bias coordinates, reconstruct six zero-gauge coordinates, and absorb all six independently tracked updates into `attn.proj.bias`.
mechanism: Six-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting five value-bias coordinates achieved 99.95% accuracy at 1590 parameters, after one through four omissions also exceeded 99%; the sixth coordinate has the same attention-invariant computational role.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9973000000000001, "parameters": 1589, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified six-coordinate value-bias quotient to seven omitted coordinates will produce a 1588-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.
change: Retain one learned value-bias coordinate, reconstruct seven zero-gauge coordinates, and absorb all seven independently tracked updates into `attn.proj.bias`.
mechanism: Seven-coordinate value-bias absorption into the attention projection bias
evidence_used: Omitting six value-bias coordinates achieved 99.73% accuracy at 1589 parameters, after omissions of one through five also exceeded 99%; the seventh coordinate has the same attention-invariant computational role.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7326, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Retaining value-bias coordinate 1 while absorbing coordinates 0 and 2–7 will produce a 1588-parameter model with at least 99% accuracy, showing whether the failed prior reduction was caused by discarding coordinate 1 rather than by the seven-coordinate quotient itself.
change: Store only the second value-bias coordinate, reconstruct the other seven as zero, and generalize quotient-aware clipping and AdamW absorption to noncontiguous omitted coordinates.
mechanism: Retained-coordinate rotation in the value-bias quotient
evidence_used: The 1589-parameter model retaining value coordinates 0 and 1 achieved 99.73%, whereas the 1588-parameter model retaining only coordinate 0 collapsed to 73.26%; retaining coordinate 1 instead directly tests the coordinate implicated by that regression.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1588, "training_steps": 4999}

RECENT RESULT
hypothesis: Omitting the final retained value-bias degree will produce a 1587-parameter model with at least 99% accuracy because all eight value-bias coordinates have the same projection-bias-absorbable role and the existing optimizer preserves their full-coordinate updates.
change: Store only the eight query-bias parameters, reconstruct the entire value bias as zero, and let the existing quotient-aware clipping and AdamW logic infer and absorb all eight omitted value coordinates.
mechanism: Complete value-bias absorption into the attention projection bias
evidence_used: Retaining only value coordinate 1 achieved 99.81% at 1588 parameters, while six-coordinate omission achieved 99.73%; this directly motivates testing the complete value-bias quotient after the retained-coordinate rotation restored accuracy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1587, "training_steps": 4999}



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
