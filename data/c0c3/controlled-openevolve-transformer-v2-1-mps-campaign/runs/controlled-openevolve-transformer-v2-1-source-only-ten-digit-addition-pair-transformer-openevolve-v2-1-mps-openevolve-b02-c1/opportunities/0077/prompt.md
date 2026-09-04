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
verified_results: {"accuracy": 0.9993000000000001, "parameters": 1002, "training_steps": 4999}
prior_hypothesis: Reducing both heads’ query/key score rank from three to two will lower the model from 1,022 to 1,002 parameters while retaining at least 99% accuracy, because rank-three routing achieved 99.85% and the evidence identifies the independently learned relative-position biases—not high-rank content scores—as the load-bearing attention pathway.

## Recent verification evidence

RECENT RESULT
hypothesis: Sharing the final LayerNorm’s last two learned bias coordinates will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because it preserves an adaptive offset where tying that coordinate to the fixed reference failed.
change: Store six final LayerNorm bias coordinates and reconstruct the seventh learned coordinate from its adjacent predecessor; leave the eighth coordinate as the common-mode reference.
mechanism: Learned adjacent final-normalization bias sharing
evidence_used: Removing all final LayerNorm biases collapsed accuracy to 15.64%, and tying one bias coordinate to the fixed reference reached only 19.67%, showing this pathway needs learned offsets; learned adjacent relative-bias sharing succeeded repeatedly, including the current 99.68% design, motivating an adaptive tie instead of deletion.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8253, "parameters": 1039, "training_steps": 4999}

RECENT RESULT
hypothesis: Removing the single hidden-layer MLP bias will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because prior removal of the larger seven-parameter MLP residual bias retained 99.95%, whereas further compression of positional and final-normalization pathways failed.
change: Make the one-unit MLP’s first projection bias-free while explicitly consuming the removed constructor’s bias initialization draw to preserve the successful initialization stream.
mechanism: Bias-free scalar MLP gate
evidence_used: Removing the seven-parameter MLP residual bias retained 99.95% accuracy, while head-0 positional sharing, head-1 quintuplet sharing, and adjacent final-LayerNorm bias sharing all fell below threshold; this makes the remaining scalar MLP bias the best-supported one-parameter ablation outside load-bearing pathways.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.1759, "parameters": 1039, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final LayerNorm’s last two learned scale coordinates will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because both coordinates remain adaptively scaled while all load-bearing normalization offsets remain independent.
change: Store six final LayerNorm scale coordinates and reconstruct the seventh from its adjacent predecessor; retain the eighth coordinate as the fixed residual-scale reference.
mechanism: Learned adjacent final-normalization scale sharing
evidence_used: Adjacent sharing of a final LayerNorm bias failed at 82.53%, showing that coordinate-specific offsets are load-bearing, while positional and scalar-MLP-bias ablations also failed; this motivates testing the still-uncompressed normalization-scale pathway with an adaptive tie rather than deleting or fixing a coordinate.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1039, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final LayerNorm’s last three learned scale coordinates will reduce the model from 1,039 to 1,038 parameters while retaining at least 99% accuracy, because sharing the last two scales achieved 99.95% and this removes only one adjacent scale degree of freedom.
change: Store five final LayerNorm scale coordinates and reconstruct the sixth and seventh learned coordinates from the final stored scale, while retaining the eighth coordinate as the fixed residual-scale reference.
mechanism: Learned adjacent final-normalization scale triplet sharing
evidence_used: The immediately prior adjacent scale-sharing result achieved 99.95% at 1,039 parameters, whereas adjacent final-normalization bias sharing reached only 82.53%; this supports extending compression in the demonstrably tolerant scale pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9684, "parameters": 1038, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing head 1’s fifth boundary bias as the midpoint of its learned neighbors will reduce the model from 1,039 to 1,038 parameters while retaining at least 99% accuracy, because it preserves a distinct transition value that direct quintuplet sharing eliminated.
change: Shorten head 1’s relative-bias parameter by one coordinate and interpolate the removed pre-boundary bias between the preceding independent bias and the successful shared quadruplet.
mechanism: Learned relative-boundary interpolation
evidence_used: Head 1’s adaptive boundary quadruplet achieved 99.68%, while extending equality to a quintuplet collapsed to 47.11%; this indicates the removed transition needs distinction, motivating learned interpolation rather than equality.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9967, "parameters": 1038, "training_steps": 4999}

RECENT RESULT
hypothesis: Decoupling each attention head’s query/key score rank from four to three will reduce the model from 1,038 to 1,022 parameters while retaining at least 99% accuracy, because positional routing remains independently learned and four-dimensional value transport remains intact.
change: Represent each head’s content-dependent attention scores with three learned query/key factors while preserving four-dimensional values, projections, relative-bias tables, gauge-aware virtual optimization, initialization stream, and decoding behavior.
mechanism: Rank-three content routing with rank-four value transport
evidence_used: Replacing independent relative-bias tables with affine positional pointers failed at 0%, identifying flexible positional routing as load-bearing, while the current independent-table model reaches 99.67%. The earlier rank-three proposal could not be verified, so whether full-rank content scores are necessary remains untested; this patch implements the decoupling consistently through initialization, factor reconstruction, optimization, and score scaling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1022, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing both heads’ query/key score rank from three to two will lower the model from 1,022 to 1,002 parameters while retaining at least 99% accuracy, because rank-three routing achieved 99.85% and the evidence identifies the independently learned relative-position biases—not high-rank content scores—as the load-bearing attention pathway.
change: Use two learned query/key factors per attention head while preserving four-dimensional values, relative-bias tables, score-factor gauge optimization, initialization behavior, and generic autoregressive decoding.
mechanism: Rank-two content routing with rank-four value transport
evidence_used: The current rank-three content-routing model reached 99.85% at 1,022 parameters after reducing 16 parameters from rank four, while affine replacement of the relative-position tables failed at 0%; this motivates the next direct rank ablation without altering the demonstrated positional-routing mechanism.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1002, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing both heads’ query/key score rank from two to one will lower the model from 1,002 to 978 parameters while retaining at least 99% accuracy, because rank-two routing achieved 99.93% and independently learned relative-position biases remain the load-bearing routing pathway.
change: Use one learned query/key factor per attention head while preserving four-dimensional values, relative-bias tables, gauge-aware optimization, initialization behavior, and generic autoregressive decoding.
mechanism: Rank-one content routing with rank-four value transport
evidence_used: Rank-three routing achieved 99.85% and rank-two routing improved to 99.93%, whereas replacing the independent relative-position tables with affine positional pointers produced 0% accuracy; this supports directly testing whether content routing can be compressed another rank without disturbing positional routing.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.4436, "parameters": 978, "training_steps": 4999}

RECENT RESULT
hypothesis: Reconstructing the shared final LayerNorm scale pair as the midpoint between the preceding learned scale and the fixed residual-scale reference will reduce the model from 1,002 to 1,001 parameters while retaining at least 99% accuracy.
change: Store five final LayerNorm scale coordinates and derive the sixth and seventh coordinates from a learned-to-reference midpoint; leave all other model and training behavior unchanged.
mechanism: Reference-anchored final-scale interpolation
evidence_used: Sharing the last two learned scales achieved 99.95%, while extending equality across three scales reached 96.84%; midpoint reconstruction preserves the successful pair sharing while avoiding the failed flat triplet.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.6056, "parameters": 1001, "training_steps": 4999}

RECENT RESULT
hypothesis: Interpolating head 0’s third-farthest relative bias between its learned fourth-farthest bias and the fixed zero boundary will reduce parameters from 1,002 to 1,001 while retaining at least 99% accuracy.
change: Remove one head-0 relative-bias parameter and reconstruct it as half of the neighboring learned boundary value, preserving rank-two content routing and all other behavior.
mechanism: Reference-anchored head-0 boundary interpolation
evidence_used: Directly sharing head 0’s third- and fourth-farthest biases failed at 59.01%, while learned boundary interpolation in head 1 retained 99.67%; interpolation preserves a distinct transition that equality removes.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.337, "parameters": 1001, "training_steps": 4999}

RECENT RESULT
hypothesis: Reducing only head 1’s query/key score rank from two to one will lower the model from 1,002 to 990 parameters while retaining at least 99% accuracy, because rank one failed only when imposed on both heads and head 1’s positional pathway tolerated substantially more compression than head 0’s.
change: Preserve rank-two content routing in head 0 while using rank-one routing in head 1, with head-specific score scaling, initialization, and gauge-aware virtual optimization.
mechanism: Asymmetric rank-one/rank-two content routing
evidence_used: Two rank-two heads achieved 99.93%, whereas reducing both heads to rank one collapsed to 44.36%; additionally, head 1 tolerated learned boundary sharing and interpolation while analogous head-0 boundary compression failed, motivating an asymmetric ablation that preserves head 0’s capacity.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8837999999999999, "parameters": 990, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing head 1’s remaining pre-boundary bias with two evenly spaced interpolants will reduce the model from 1,002 to 1,001 parameters while maintaining at least 99% accuracy, because it preserves distinct transition values instead of extending the failed equality constraint.
change: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct the two preceding plateau-transition biases at one-third and two-thirds between neighboring learned endpoints.
mechanism: Two-step learned head-1 boundary interpolation
evidence_used: Head 1’s single learned boundary interpolation retained 99.67%, while extending the plateau by direct equality collapsed to 47.11%; this motivates lengthening the successful smooth transition without compressing the sensitive head-0 pathway.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9831, "parameters": 1001, "training_steps": 4999}



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
