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
verified_results: {"accuracy": 0.9976, "parameters": 1545, "training_steps": 4999}
prior_hypothesis: Removing one `fc1` input-weight scalar will reduce the model from 1,546 to 1,545 parameters while retaining at least 99% accuracy, because zero-mean LayerNorm outputs make one weight direction per MLP neuron functionally redundant while all eight learned `ln2` scales remain trainable.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing all eight value-projection bias parameters will reduce the model from 1,547 to 1,539 parameters while retaining at least 99% accuracy, because causal attention weights sum to one, making the value bias an input-independent offset exactly representable by the retained mean-free output-projection bias.
change: Retain only the learned query bias in `qkv`; reconstruct zero key and value biases during the forward pass while leaving all attention weights and the output-projection bias unchanged.
mechanism: Attention value/output affine-bias quotient
evidence_used: The 1,547-parameter design achieved 99.88% accuracy after completely quotienting the analogous pre-MLP affine bias, while multi-query weight sharing failed at 6.64%; this patch removes only a mathematically redundant affine pathway without sharing load-bearing representation weights.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7651, "parameters": 1539, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing only the vocabulary-common direction of the eight-dimensional value bias will reduce the model from 1,547 to 1,546 parameters while retaining at least 99% accuracy, because the retained seven balanced directions preserve value-bias optimization and the omitted constant attention output is absorbable by the output-projection bias.
change: Reparameterize the value-projection bias with a seven-dimensional orthonormal mean-free basis while retaining all query biases and reconstructing the full value bias during attention.
mechanism: Balanced single-mode value/output bias quotient
evidence_used: Removing all eight value-bias parameters fell to 76.51%, despite their affine redundancy, indicating that the optimization pathway is useful; this minimal balanced quotient removes one parameter while restoring seven of the eight pathways.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.109, "parameters": 1546, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one attention output-projection bias coordinate while retaining all eight value biases will reduce the model from 1,547 to 1,546 parameters and maintain at least 99% accuracy, because the full-rank value-to-output projection lets the retained value bias represent the omitted constant output direction.
change: Use a mean-free attention output projection with six learned bias coordinates instead of seven, leaving its weights and the complete query/value bias pathway unchanged.
mechanism: Value-preserving attention affine-bias quotient
evidence_used: Removing all value biases reached only 76.51%, and their balanced seven-parameter reconstruction reached 10.9%, showing that the value-bias optimization pathway is load-bearing despite the affine redundancy. The verified 1,547-parameter model restores that pathway and reaches 99.88%, motivating removal from the opposite side of the quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7367, "parameters": 1546, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one `ln2` scale coordinate at one will reduce the model from 1,547 to 1,546 parameters while retaining at least 99% accuracy, because the following unconstrained `fc1` weight column can absorb that scale and the initialized function remains unchanged.
change: Store seven learned scale coordinates in the bias-free pre-MLP LayerNorm and reconstruct the eighth as a fixed identity scale.
mechanism: Pre-MLP scale/weight quotient
evidence_used: The 1,547-parameter design reached 99.88% after successfully quotienting all `ln2` bias coordinates into `fc1`; this tests the analogous downstream-affine redundancy one coordinate at a time while preserving the full MLP preactivation function family.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.3343, "parameters": 1546, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the remaining learned final-LayerNorm bias coordinate at its initialized zero will reduce the model from 1,547 to 1,546 parameters while retaining at least 99% accuracy, because the fixed common offset preserves the token-row-mean pathway that already supports eliminating the other seven bias parameters.
change: Replace the final LayerNorm’s single learned bias coordinate with only its existing fixed common bias, leaving all attention, scale, embedding, and training pathways unchanged.
mechanism: Complete final-LayerNorm bias quotient
evidence_used: The verified 1,547-parameter model reaches 99.88% accuracy with seven of eight final-LayerNorm bias parameters already removed; extending that successful quotient is more targeted than modifying the attention-bias or LayerNorm-scale pathways whose recent one-parameter reductions failed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1546, "training_steps": 4999}

RECENT RESULT
hypothesis: Centering a second positional content coordinate across sequence positions will reduce the model from 1,546 to 1,545 parameters while retaining at least 99% accuracy, because its omitted position-common component is representable by the vocabulary-common component of the tied token embedding and is softmax-null at the output.
change: Reparameterize the final two positional content coordinates with the existing mean-free position basis instead of only the final coordinate, leaving their position-varying capacity and all training settings unchanged.
mechanism: Second token/position translation-gauge quotient
evidence_used: The current 1,546-parameter model achieves 99.93% accuracy with one position-common mode already removed; extending that successful gauge quotient preserves positional variation, unlike the previously noted failures from pruning positional representation channels.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7143, "parameters": 1545, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing one `fc1` input-weight scalar will reduce the model from 1,546 to 1,545 parameters while retaining at least 99% accuracy, because zero-mean LayerNorm outputs make one weight direction per MLP neuron functionally redundant while all eight learned `ln2` scales remain trainable.
change: Constrain one `fc1` row to have a fixed final coordinate, canonicalize its fresh initialization to preserve the initial function, and leave the other eleven rows unchanged.
mechanism: Single-row pre-MLP LayerNorm/weight quotient
evidence_used: Fixing an `ln2` scale coordinate failed at 33.43%, showing that scale optimization is load-bearing; this patch preserves every scale and instead removes one scalar from the opposite side of the same exact LayerNorm/linear quotient. The successful complete `ln2` bias quotient further supports targeting redundancy at this interface.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976, "parameters": 1545, "training_steps": 4999}



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
