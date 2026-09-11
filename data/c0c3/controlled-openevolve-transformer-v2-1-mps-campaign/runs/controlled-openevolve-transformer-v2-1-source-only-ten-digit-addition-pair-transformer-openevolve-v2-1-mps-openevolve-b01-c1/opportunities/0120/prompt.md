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
verified_results: {"accuracy": 1.0, "parameters": 1528, "training_steps": 4999}
prior_hypothesis: Fixing `ln2` scale coordinate two at one will reduce the model from 1,529 to 1,528 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the eleven trainable `fc1` coefficients using that coordinate.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing a second position-common mode will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because the quotient preserves initial inputs exactly by transferring the removed common component into the tied token embedding.
change: Mean-center the final two position-content coordinates instead of only the final coordinate, and transfer the newly removed initialization component into the matching non-decayed token coordinate.
mechanism: Second position-common-mode quotient with initialization transfer
evidence_used: The 1,534-parameter design achieves 99.91% with one position-common mode already removed. The failed inverse token-centering experiment moved a common component into decayed position parameters; this patch instead extends the successful positional constraint and moves its component into the token parameters already exempted from weight decay.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6981999999999999, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Mean-centering positional content coordinate one and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy.
change: Represent the second positional content coordinate in the existing position-mean-free basis, reconstruct it during the forward pass, and transfer its initialization mean into token embeddings so initial token-plus-position inputs are preserved.
mechanism: Second-coordinate position-common-mode quotient
evidence_used: The 1,534-parameter design achieved 99.91% with one positional common mode removed. Removing the adjacent penultimate mode failed at 69.82%, while second-coordinate constraints succeeded for both `fc1` rows five and six; this motivates testing the same exact positional quotient on the optimization-favorable second coordinate, with its common component transferred into already non-decayed token parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1533, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing one learned four-dimensional key encoder across both attention heads will reduce the model from 1,533 to 1,501 parameters while retaining at least 99% accuracy, because head-specific queries and values can still retrieve different operands through a common learned positional-address space.
change: Replace the independent per-head key projections with one learned key projection broadcast across both heads, while retaining separate query and value projections and preserving the original full-QKV initialization draw and variance.
mechanism: Shared-key multi-head causal attention
evidence_used: The 1,533-parameter design succeeds, but repeated isolated gauge removals are optimization-brittle, and rank-five token compression lost essential token geometry. This challenges the shared assumption that each head needs an independent key dictionary: fixed-format addition plausibly needs a common address representation, while distinct queries and values preserve head specialization.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2168, "parameters": 1501, "training_steps": 4999}

RECENT RESULT
hypothesis: Mean-centering positional content coordinate zero and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,533 to 1,532 parameters while retaining at least 99% accuracy.
change: Add a third mean-free positional coordinate, reconstruct it from the position basis, and transfer its initialization mean into token embeddings so initial token-plus-position inputs remain unchanged.
mechanism: First-coordinate positional common-mode quotient
evidence_used: The second-coordinate positional quotient reached 99.84%, while the inverse first-coordinate token-centering quotient reached 93.48% despite transferring its common component into decayed positional parameters; applying the successful transfer direction to this untested positional coordinate is the most targeted next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6271, "parameters": 1532, "training_steps": 4999}

RECENT RESULT
hypothesis: Mean-centering positional content coordinate two and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,533 to 1,532 parameters while retaining at least 99% accuracy.
change: Represent the third positional content coordinate in the existing position-mean-free basis, reconstruct it during the forward pass, and transfer its initialization mean into token embeddings to preserve initial token-plus-position inputs.
mechanism: Third-coordinate positional common-mode quotient
evidence_used: The second-coordinate positional quotient achieved 99.84%, whereas equivalent constraints on coordinates zero and four failed, demonstrating coordinate-specific optimization behavior; coordinate two is the closest untested positional mode to the successful coordinate.
result: the patch search text did not match the source

RECENT RESULT
hypothesis: Mean-centering positional content coordinate two and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,533 to 1,532 parameters while retaining at least 99% accuracy.
change: Represent positional content coordinate two in the existing position-mean-free basis, reconstruct it during the forward pass, and transfer its initialization mean into token embeddings so initial token-plus-position inputs remain unchanged.
mechanism: Third-coordinate positional common-mode quotient
evidence_used: The second-coordinate positional quotient achieved 99.84%, while equivalent constraints on coordinates zero and four failed; coordinate two is the closest untested positional mode, and its previous proposed test was malformed because the patch did not match the source.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1532, "training_steps": 4999}

RECENT RESULT
hypothesis: Mean-centering positional content coordinate three and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,532 to 1,531 parameters while retaining at least 99% accuracy.
change: Represent positional content coordinate three in the position-mean-free basis, reconstruct it during the forward pass, and transfer its initialization mean into token embeddings to preserve initial token-plus-position inputs.
mechanism: Fourth-coordinate positional common-mode quotient
evidence_used: Positional coordinate two achieved 100% accuracy after coordinate one achieved 99.84%, while coordinates zero and four failed; coordinate three is the closest untested positional mode to the two successful adjacent coordinates.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1531, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the equal-weight common mode of positional content coordinates zero and four will reduce the model from 1,531 to 1,530 parameters while retaining at least 99% accuracy, because it preserves a trainable orthogonal common mode and exactly preserves initialized token-plus-position inputs.
change: Reparameterize the two remaining positional coordinates into mean-free variation plus one shared common component, transferring the removed balanced component into the corresponding non-decayed token coordinates.
mechanism: Balanced residual positional common-mode quotient
evidence_used: Individual removal of coordinate zero reached 62.71% and coordinate four reached 69.82%, while coordinates one through three successfully supported the same quotient; a balanced constraint tests whether the two resistant coordinates need a common degree of freedom without fixing either coordinate outright.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7489, "parameters": 1530, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate seven at its initialized value of one will reduce the model from 1,531 to 1,530 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the corresponding trainable `fc1` coefficients.
change: Retain seven learned `ln2` scale coordinates and reconstruct the final coordinate as one during the forward pass.
mechanism: Downstream-absorbed final LayerNorm scale
evidence_used: The 1,531-parameter design reached 99.94% accuracy while `fc1` already omits its final input coordinate in seven of twelve rows; the scale is unused in those rows and multiplicatively redundant with the five remaining trainable final-coordinate weights.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1530, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate one at its initialized value of one will reduce the model from 1,530 to 1,529 parameters while retaining at least 99% accuracy, because its effect remains absorbable by the ten trainable `fc1` coefficients that use that coordinate.
change: Retain learned `ln2` scales for coordinates zero and two through six, while reconstructing coordinates one and seven as ones during the forward pass.
mechanism: Downstream-absorbed second LayerNorm scale
evidence_used: Fixing coordinate seven achieved 99.96% accuracy at 1,530 parameters; coordinate one is the next-sparsest `fc1` input coordinate because the successful row-five and row-six quotients already omit it.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1529, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate two at one will reduce the model from 1,529 to 1,528 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the eleven trainable `fc1` coefficients using that coordinate.
change: Retain learned `ln2` scales for coordinates zero and three through six, reconstructing coordinates one, two, and seven as ones.
mechanism: Downstream-absorbed third-coordinate LayerNorm scale
evidence_used: Fixing `ln2` coordinates seven and one successively preserved 99.96% and 99.99% accuracy; coordinate two is among the next-sparsest `fc1` input coordinates and has also shown favorable optimization behavior in the successful positional-coordinate quotient.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1528, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate three at one will reduce the model from 1,528 to 1,527 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the eleven trainable `fc1` coefficients using that coordinate.
change: Retain learned `ln2` scales for coordinates zero and four through six, reconstructing coordinates one, two, three, and seven as ones.
mechanism: Downstream-absorbed fourth-coordinate LayerNorm scale
evidence_used: Fixing `ln2` coordinates seven, one, and two successively preserved 99.96%, 99.99%, and 100% accuracy; coordinate three is one of the next-sparsest `fc1` inputs and also exhibited favorable optimization in the successful positional-coordinate quotient.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7415, "parameters": 1527, "training_steps": 4999}

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
