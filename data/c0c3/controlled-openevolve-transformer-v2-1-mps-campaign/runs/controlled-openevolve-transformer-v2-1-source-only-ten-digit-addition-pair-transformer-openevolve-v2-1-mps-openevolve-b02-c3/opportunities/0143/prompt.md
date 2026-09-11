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
verified_results: {"accuracy": 0.9987999999999999, "parameters": 662, "training_steps": 4999}
prior_hypothesis: A 662-parameter transformer will achieve at least 99% accuracy because fixing the last key-channel scale preserves the 663-parameter model’s representable attention scores, while final-only validation removes the runtime bottleneck that prevented both prior 662-parameter attempts from finishing.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9948, "parameters": 665, "training_steps": 4999}
prior_hypothesis: A 665-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary triplet sharing to four adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987999999999999, "parameters": 663, "training_steps": 4999}
prior_hypothesis: A 663-parameter transformer will retain at least 99% accuracy because fixing the remaining first-head key-channel scale removes a Q/K rescaling redundancy while preserving the qualified 664-parameter model’s routing biases and representable attention scores.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9956999999999999, "parameters": 664, "training_steps": 4999}
prior_hypothesis: A 664-parameter transformer will retain at least 99% accuracy because extending the qualified first-head boundary quartet sharing to five adjacent biases preserves an adaptive long-distance routing signal while removing one learned scalar.

## Recent verification evidence

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

RECENT RESULT
hypothesis: A 661-parameter transformer will retain at least 99% accuracy because the first head’s V/output GL(3) symmetry allows its remaining three learned value-frame coefficients to be fixed as a full-rank canonical frame while the learned output projection absorbs the basis change.
change: Complete the partially fixed first-head 3×3 value frame by fixing its two remaining diagonal coefficients at 0.02 and its remaining off-diagonal coefficient at zero, removing exactly three learned QKV parameters without compressing routing biases or MLP capacity.
mechanism: Canonical fixed value frame with learned output decoding
evidence_used: The current 664-parameter model achieved 99.57%, and the 663-parameter first-head Q/K gauge fixing achieved 99.88%, showing that attention-coordinate redundancies can be removed safely. Conversely, tying functionally distinct MLP directions collapsed to 22.53%, motivating an exact V/output basis gauge rather than another functional-capacity tie.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 662-parameter transformer will achieve at least 99% accuracy because the second head’s remaining key-channel scale is an exact Q/K rescaling redundancy, while vectorized weight reconstruction addresses the prior attempt’s verification-time failure without changing model function.
change: Reconstruct the qualified 663-parameter first-head-quintet design, fix the analogous remaining second-head key coefficient at 0.02, and replace iterative QKV assembly with an equivalent single scatter operation.
mechanism: Second-head Q/K coordinate-scale gauge fixing with vectorized QKV reconstruction
evidence_used: Fixing the first-head key-channel scale yielded 99.88% at 663 parameters; the analogous 662-parameter attempt timed out rather than producing negative accuracy evidence, while further relative-bias compression failed.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A 662-parameter transformer will achieve at least 99% accuracy because fixing the last key-channel scale preserves the 663-parameter model’s representable attention scores, while final-only validation removes the runtime bottleneck that prevented both prior 662-parameter attempts from finishing.
change: Reconstruct the qualified first-head-quintet design, fix the remaining second-head key scale at 0.02, and evaluate only the final trained state so both required checkpoints are written within the time limit.
mechanism: Complete per-head Q/K scale gauge with final-only validation
evidence_used: The analogous first-head gauge fixing achieved 99.88% at 663 parameters; both second-head attempts timed out without producing negative accuracy evidence, so reducing validation overhead directly tests the same exact attention symmetry.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 662, "training_steps": 4999}



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
