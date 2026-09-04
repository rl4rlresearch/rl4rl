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
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1525, "training_steps": 4999}
prior_hypothesis: Removing the position-common component of the fifth retained positional coordinate will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because its input contribution can be transferred exactly to the corresponding token-embedding coordinate and remains softmax-null under the tied output projection.

## Recent verification evidence

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

RECENT RESULT
hypothesis: Removing all eight learned value-bias coordinates will reduce the model from 1,528 to 1,520 parameters while retaining at least 99% accuracy, because softmax-normalized attention turns value bias into a position-independent offset that the trainable output-projection bias can represent directly.
change: Challenge the assumption that each attention head needs a learned value offset: retain all head-specific query, key, and value weights and the query bias, but fix value bias to zero and represent learned constant attention offsets only after head mixing.
mechanism: Post-mixing attention offset quotient
evidence_used: The 1,528-parameter design achieved 100% accuracy, whereas sharing key weights collapsed accuracy to 21.68%, indicating that head-specific addressing is load-bearing. This patch leaves that addressing untouched and removes an exact redundancy: with zero attention dropout, every attention row sums to one, so the value bias always contributes only through the existing mean-free projection bias.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6756, "parameters": 1520, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate zero at one will reduce the model from 1,528 to 1,527 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the eleven trainable `fc1` coefficients using that coordinate.
change: Retain learned `ln2` scales only for coordinates three through six, reconstructing coordinates zero, one, two, and seven as ones.
mechanism: Downstream-absorbed first-coordinate LayerNorm scale
evidence_used: Fixing coordinates seven, one, and two retained 99.96%, 99.99%, and 100% accuracy. Coordinate zero is the remaining `fc1` input with the same next-lowest usage count as failed coordinate three, making it the most informative coordinate-specific test before attempting fully dense coordinates.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5516, "parameters": 1527, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing only the final value-bias coordinate at zero will reduce the model from 1,528 to 1,527 parameters while retaining at least 99% accuracy, because its position-independent contribution can be learned by the attention output bias without the optimization disruption caused by removing all eight value-bias coordinates simultaneously.
change: Retain all query-bias coordinates and the first seven value-bias coordinates, reconstructing the final value-bias coordinate as zero during attention.
mechanism: Single-coordinate post-mixing attention offset quotient
evidence_used: Removing all eight value-bias coordinates reduced accuracy to 67.56%, showing that the full quotient is optimization-brittle despite its exact post-mixing redundancy; successful one-coordinate-at-a-time LayerNorm reductions show that isolating a single redundant coordinate can preserve training behavior.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4475, "parameters": 1527, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate four at one will reduce the model from 1,528 to 1,527 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.
change: Retain learned `ln2` scales for coordinates zero, three, five, and six, reconstructing coordinates one, two, four, and seven as ones.
mechanism: Fully connected LayerNorm-scale absorption
evidence_used: Fixing coordinates seven, one, and two preserved at least 99.96% accuracy, while failed coordinates zero and three are each structurally omitted by one `fc1` row. Coordinate four is the first untested scale used by every `fc1` row, making downstream absorption least constrained.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1527, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate five at one will reduce the model from 1,527 to 1,526 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.
change: Retain learned `ln2` scales only for coordinates zero, three, and six, reconstructing coordinates one, two, four, five, and seven as ones.
mechanism: Downstream-absorbed sixth-coordinate LayerNorm scale
evidence_used: Fixing coordinate four—the first fully connected `fc1` input coordinate tested—preserved 99.97% accuracy at 1,527 parameters, while failures occurred on coordinates zero and three, which are each omitted by one row. Coordinate five has the same fully connected downstream structure as successful coordinate four.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1526, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln2` scale coordinate six at one will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.
change: Retain learned `ln2` scales only for coordinates zero and three, reconstructing coordinates one, two, and four through seven as ones.
mechanism: Fully connected final LayerNorm-scale absorption
evidence_used: Successively fixing fully connected coordinates four and five preserved 99.97% and 99.82% accuracy; coordinate six has the same full downstream connectivity, unlike failed coordinates zero and three, which are each omitted by an `fc1` row.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.665, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing `ln1` scale coordinate seven at one will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because all query, key, and value rows have trainable coefficients that can absorb its scale.
change: Retain seven learned `ln1` scales and reconstruct the final scale as one during the forward pass.
mechanism: Downstream-absorbed attention LayerNorm scale
evidence_used: Fixing `ln2` coordinate seven preserved 99.96% accuracy despite sparse downstream use; `ln1` coordinate seven is used by every dense QKV row, and its bias is already fixed at zero, making absorption less constrained.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.408, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying `ln2` scale coordinate six to the mean of the load-bearing scales at coordinates zero and three will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because coordinate six remains trainable instead of being fixed at one.
change: Retain independent `ln2` scales for coordinates zero and three, and reconstruct coordinate six as their arithmetic mean.
mechanism: Adaptive shared LayerNorm scale
evidence_used: Fixing coordinate six at one fell to 66.5%, whereas fixing coordinates four and five retained 99.97% and 99.82%; this suggests the final scale needs adaptive training dynamics, which parameter sharing preserves while testing whether it needs an independent degree of freedom.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9773000000000001, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the position-common component of the fifth retained positional coordinate will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because its input contribution can be transferred exactly to the corresponding token-embedding coordinate and remains softmax-null under the tied output projection.
change: Represent positional coordinate four with a mean-free basis, transfer its initialized common component to token embeddings, and retain coordinate zero as the sole dense positional coordinate.
mechanism: Fifth positional common-mode quotient
evidence_used: The verified 1,526-parameter design reached 99.82% accuracy while already quotienting four positional common modes by this mechanism; unlike the failed LayerNorm-scale reductions, this removes another instance of the same input-sum redundancy.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1525, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the position-common component of the sole remaining dense positional coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its constant input contribution can be transferred exactly to the matching token-embedding coordinate.
change: Represent positional coordinate zero in the mean-free position basis, retain its removed common component in a buffer, and add that component to token embeddings at initialization.
mechanism: Complete positional common-mode quotient
evidence_used: Removing the common component of positional coordinate four preserved 99.88% accuracy at 1,525 parameters; this applies the same exact input-sum reparameterization to the only positional coordinate still storing a common mode.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9388, "parameters": 1524, "training_steps": 4999}

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
