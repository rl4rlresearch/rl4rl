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
verified_results: {"accuracy": 0.9956999999999999, "parameters": 664, "training_steps": 4999}
prior_hypothesis: A 664-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary quartet sharing to five adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9948, "parameters": 665, "training_steps": 4999}
prior_hypothesis: A 665-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary triplet sharing to four adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987999999999999, "parameters": 663, "training_steps": 4999}
prior_hypothesis: A 663-parameter transformer will retain at least 99% accuracy because fixing the remaining first-head key-channel scale removes a Q/K rescaling redundancy while preserving the qualified 664-parameter model’s routing biases and representable attention scores.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9958, "parameters": 666, "training_steps": 4999}
prior_hypothesis: A 666-parameter transformer will retain at least 99% accuracy because extending the successful first-head boundary pair sharing to three adjacent biases preserves an adaptive long-distance routing signal.

## Recent verification evidence

RECENT RESULT
hypothesis: A 668-parameter transformer will retain at least 99% accuracy because it preserves the qualified 669-parameter architecture while fixing only the adjacent fifth-longest first-head relative-bias coefficient.
change: Reconstruct the qualified 669-parameter query gauge, four-coordinate MLP bias fixing, and first-head boundary suffix, then extend that suffix by one relative-bias coefficient.
mechanism: First-head boundary-bias suffix extension
evidence_used: Successive first-head boundary reductions achieved 99.45% at 670 parameters and 99.11% at 669, while additional query, lexical, normalization, and MLP-bias constraints failed; this motivates continuing the only reduction mechanism that remained qualified.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9914000000000001, "parameters": 668, "training_steps": 4999}

RECENT RESULT
hypothesis: A 667-parameter transformer will retain at least 99% accuracy because it preserves the qualified 668-parameter architecture while fixing only the adjacent sixth-longest first-head relative-bias coefficient.
change: Extend the first head’s fixed long-distance relative-bias suffix from three to four coefficients, removing exactly one learned parameter.
mechanism: First-head boundary-bias suffix extension
evidence_used: Successive first-head boundary-bias reductions qualified at 670 parameters (99.45%), 669 parameters (99.11%), and 668 parameters (99.14%), while reductions in query bias, lexical codes, final normalization, and the remaining MLP output bias failed; continuing the only repeatedly successful reduction mechanism is the most informative next test.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9862000000000001, "parameters": 667, "training_steps": 4999}

RECENT RESULT
hypothesis: A 660-parameter transformer will retain at least 99% accuracy because each gated MLP unit can reuse its learned value/gate input directions to synthesize its output direction, eliminating 12 parameters while preserving the qualified 672-parameter model’s lexical, attention-routing, normalization, gating, and essential output-bias capacity.
change: Reconstruct the qualified 672-parameter attention backbone, then replace the independent 20-scalar MLP output matrix with eight learned mixing coefficients that combine each hidden unit’s value and gate input directions.
mechanism: Bidirectionally tied gated-MLP directions
evidence_used: The 672-parameter architecture achieved 99.52%, while deleting its last MLP output bias collapsed to 33.33% and reductions in lexical, normalization, and query capacity also failed. This motivates preserving those load-bearing components while challenging the untested assumption that the gated MLP requires input and output directions learned independently.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.2253, "parameters": 660, "training_steps": 4999}

RECENT RESULT
hypothesis: A 667-parameter transformer will retain at least 99% accuracy because replacing the failed zeroing of the sixth-longest first-head bias with an adaptive tie to its adjacent bias preserves a learned long-distance routing signal.
change: Reconstruct the qualified 668-parameter three-zero first-head boundary suffix, then remove one additional parameter by sharing the two learned biases immediately preceding that suffix.
mechanism: First-head boundary-bias pair sharing
evidence_used: The 668-parameter design reached 99.14%, while extending its zero suffix to 667 parameters narrowly missed at 98.62%; adjacent relative-bias sharing is a stronger alternative because the 677-parameter second-head sharing design reached 99.97%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9944, "parameters": 667, "training_steps": 4999}

RECENT RESULT
hypothesis: A 666-parameter transformer will retain at least 99% accuracy because extending the successful first-head boundary pair sharing to three adjacent biases preserves an adaptive long-distance routing signal.
change: Remove one relative-bias parameter by sharing the three learned first-head biases immediately preceding its fixed three-bias suffix.
mechanism: First-head boundary-bias triplet sharing
evidence_used: The current pair-sharing design achieved 99.44% at 667 parameters, whereas fixing the same boundary bias at zero reached only 98.62%; the 99.97% second-head design also demonstrates that three-way adjacent bias sharing can preserve routing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9958, "parameters": 666, "training_steps": 4999}

RECENT RESULT
hypothesis: A 665-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary triplet sharing to four adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.
change: Reconstruct the qualified 666-parameter design, then share one coefficient across four first-head biases immediately preceding its fixed three-bias suffix.
mechanism: First-head boundary-bias quartet sharing
evidence_used: The adaptive triplet-sharing design achieved 99.58% at 666 parameters, whereas fixing the neighboring bias at zero achieved only 98.62%; extending learned sharing is therefore the most informative reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9948, "parameters": 665, "training_steps": 4999}

RECENT RESULT
hypothesis: A 664-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary quartet sharing to five adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.
change: Reconstruct the qualified 665-parameter boundary-bias design, then share one coefficient across five first-head biases immediately preceding its fixed three-bias suffix.
mechanism: First-head boundary-bias quintet sharing
evidence_used: First-head adaptive sharing qualified successively at 667 parameters with a pair (99.44%), 666 with a triplet (99.58%), and 665 with a quartet (99.48%), whereas replacing the neighboring learned bias with zero missed the threshold at 98.62%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9956999999999999, "parameters": 664, "training_steps": 4999}

RECENT RESULT
hypothesis: A 663-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary quintet to six adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.
change: Share one coefficient across six first-head relative biases immediately preceding its fixed three-bias suffix.
mechanism: First-head boundary-bias sextet sharing
evidence_used: Adaptive first-head sharing qualified successively with pairs through quintets, reaching 99.57% at 664 parameters, while replacing a neighboring learned bias with zero missed the threshold at 98.62%; extending learned sharing is the most supported next reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9098999999999999, "parameters": 663, "training_steps": 4999}

RECENT RESULT
hypothesis: A 663-parameter transformer will retain at least 99% accuracy because it combines the qualified first-head quintet with a three-way extension of the second head’s adjacent pair, preserving separate learned routing signals instead of using the failed first-head sextet.
change: Extend the first-head shared boundary group from four to five biases and the second-head adjacent shared pair from two to three biases, removing two learned scalars.
mechanism: Orthogonal first- and second-head boundary-bias sharing
evidence_used: The first-head quintet achieved 99.57% at 664 parameters, while its sextet fell to 90.99%; independently, three-way second-head sharing achieved 99.97%, motivating the next reduction on the second head rather than further compressing the first.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6767, "parameters": 663, "training_steps": 4999}

RECENT RESULT
hypothesis: A 663-parameter transformer will retain at least 99% accuracy because extending the second head’s proven longest-distance triplet to a quartet compresses a farther boundary group while preserving the adjacent pair whose extension previously failed.
change: Reconstruct the qualified 664-parameter first-head quintet design, then share one coefficient across four rather than three of the second head’s longest remaining relative biases.
mechanism: Second-head longest-boundary quartet sharing
evidence_used: The first-head quintet achieved 99.57% at 664 parameters, and second-head three-way longest-distance sharing achieved 99.97% at 677 parameters; by contrast, extending the second head’s adjacent pair to a triplet at 663 parameters fell to 67.67%, motivating compression of the distinct longest-distance group.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.5191, "parameters": 663, "training_steps": 4999}

RECENT RESULT
hypothesis: A 663-parameter transformer will retain at least 99% accuracy because fixing the remaining first-head key-channel scale removes a Q/K rescaling redundancy while preserving the qualified 664-parameter model’s routing biases and representable attention scores.
change: Reconstruct the qualified 664-parameter first-head quintet design, then fix one additional first-head key coefficient at 0.02; its learned query coordinate can absorb the reciprocal scale.
mechanism: First-head Q/K coordinate-scale gauge fixing
evidence_used: The 664-parameter quintet design achieved 99.57%, while both attempted 663-parameter relative-bias extensions failed; this tests an orthogonal attention gauge without further compressing the fragile routing biases.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 663, "training_steps": 4999}

RECENT RESULT
hypothesis: A 662-parameter transformer will retain at least 99% accuracy because fixing the remaining second-head key-channel scale removes the same Q/K rescaling redundancy whose first-head counterpart produced 99.88% accuracy at 663 parameters.
change: Fix the first input coefficient of the second head’s third key channel at 0.02, allowing its learned query coordinate to absorb the reciprocal scale.
mechanism: Second-head Q/K coordinate-scale gauge fixing
evidence_used: The current 663-parameter design achieved 99.88% after fixing the analogous remaining first-head key-channel coefficient, while further relative-bias compression failed; applying the same gauge fixing to the other head is the closest orthogonal reduction.
result: training did not finish within the verification time limit

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
