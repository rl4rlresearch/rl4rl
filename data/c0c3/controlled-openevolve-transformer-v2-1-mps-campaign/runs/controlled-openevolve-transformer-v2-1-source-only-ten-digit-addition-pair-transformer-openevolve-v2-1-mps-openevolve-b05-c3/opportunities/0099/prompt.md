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
verified_results: {"accuracy": 0.9983, "parameters": 1255, "training_steps": 4999}
prior_hypothesis: Fixing the already-qualified head phase and temperature and tying the final two shared-kernel lag logits will reduce the model from 1,258 to 1,255 parameters while retaining at least 99% accuracy.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9984999999999999, "parameters": 1252, "training_steps": 4999}
prior_hypothesis: Reusing the final LayerNorm’s learned mean scale for the sole pre-attention LayerNorm scale will reduce the qualified four-terminal-tie model from 1,253 to 1,252 parameters while retaining at least 99% accuracy, because it preserves adaptive scale conditioning that was lost in the failed fully fixed `ln1` design.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9991, "parameters": 1253, "training_steps": 4999}
prior_hypothesis: Extending the qualified three-coordinate terminal lag tie to four coordinates will reduce the model from 1,254 to 1,253 learned parameters while retaining at least 99% accuracy, because the remaining shared lag kernel preserves nearly all routing capacity.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9994, "parameters": 1256, "training_steps": 4999}
prior_hypothesis: Fixing the relative head phase at its evenly spaced initialization will retain at least 99% accuracy while reducing the model from 1,257 to 1,256 learned parameters, because the unrestricted learned lag kernel can adapt its routing while the fixed nonzero phase preserves head specialization.

## Recent verification evidence

RECENT RESULT
hypothesis: Extending the verified six-coordinate `ln2` scale quotient to seven coordinates will reduce the model from 1,280 to 1,279 learned parameters while retaining at least 99% accuracy, because `fc1` can absorb the additional fixed channel scale while one adaptive scale remains.
change: Represent `ln2` with one learned scale and seven fixed unit scales, preserving the learned attention, gauge-aware optimizer, checkpoints, and decoding interface.
mechanism: Seven-coordinate pre-MLP LayerNorm-scale quotient
evidence_used: The six-coordinate `ln2` quotient achieved 99.87% accuracy with 1,280 parameters, and the analogous seven-coordinate `ln1` quotient achieved 99.96%; this directly motivates testing the next single-coordinate reduction without removing all adaptive scales.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1279, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final adaptive `ln2` scale will reduce the qualified 1,279-parameter model to 1,278 parameters while retaining at least 99% accuracy, because `fc1` can absorb every channel scale.
change: Represent `ln2` with a fixed unit-scale buffer and no learned scale parameters, leaving the learned attention, MLP, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Complete pre-MLP LayerNorm-scale absorption
evidence_used: The seven-coordinate `ln2` quotient achieved 99.97% accuracy with 1,279 parameters, and every preceding incremental quotient exceeded 99%, directly motivating removal of the sole remaining adaptive scale.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1278, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the final adaptive `ln1` scale will reduce the qualified model from 1,278 to 1,277 learned parameters while retaining at least 99% accuracy, because the following value map can absorb every channel scale.
change: Replace the sole learned `ln1` scale with a fixed unit-scale buffer while preserving the learned attention, optimizer, checkpoints, and decoding interface.
mechanism: Complete pre-attention LayerNorm-scale absorption
evidence_used: The current complete `ln2` scale absorption achieved 99.94% accuracy with 1,278 parameters, and the analogous seven-coordinate `ln1` quotient previously achieved 99.96%; this motivates removing its sole remaining adaptive scale.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9771, "parameters": 1277, "training_steps": 4999}

RECENT RESULT
hypothesis: A full-resolution learned lag kernel shared through per-head learned cyclic shifts and temperatures will retain at least 99% accuracy with 1,251 parameters, because the heads need distinct routing locations but may not need independently learned kernel shapes.
change: Replace the old assumption that each head requires an unrelated dense lag table with one unrestricted discrete kernel whose learned phase and temperature produce distinct head routes, reducing the attention routing parameters from 60 to 32.
mechanism: Phase-shared discrete lag attention
evidence_used: Dense stationary lag routing achieved 99.97%, showing discrete lag resolution is load-bearing; the signed-table design achieved only 0.15%, showing opposite logits are too restrictive. Learned translation and scaling preserve discrete resolution while avoiding that failed complementary-sign constraint.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995, "parameters": 1259, "training_steps": 4999}

RECENT RESULT
hypothesis: Combining the verified phase-shared lag kernel with complete `ln2` scale absorption will reduce the qualified 1,259-parameter design by one learned parameter while retaining at least 99% accuracy.
change: Replace independent per-head lag tables with one learned full-resolution kernel plus a learned relative phase and temperature, and make every `ln2` scale fixed at one.
mechanism: Phase-shared lag attention with complete pre-MLP scale absorption
evidence_used: Phase-shared discrete lag attention achieved 99.95% accuracy with 1,259 parameters, while complete `ln2` scale absorption independently achieved 99.94%; this directly tests whether the two successful reductions compose.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1258, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the relative head temperature in the verified phase-shared, fully scale-absorbed design will retain at least 99% accuracy while reducing the model from 1,258 to 1,257 learned parameters, because the learned cyclic phase still provides distinct head routing.
change: Adopt phase-shared discrete lag attention and complete `ln2` scale absorption, while fixing both heads to the shared kernel temperature.
mechanism: Equal-temperature phase-shared lag attention
evidence_used: Phase-shared lag attention combined with complete `ln2` scale absorption achieved 99.98% accuracy with 1,258 parameters; removing only its single relative-temperature parameter isolates whether phase separation alone supplies the necessary head specialization.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9993000000000001, "parameters": 1257, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the relative head phase at its evenly spaced initialization will retain at least 99% accuracy while reducing the model from 1,257 to 1,256 learned parameters, because the unrestricted learned lag kernel can adapt its routing while the fixed nonzero phase preserves head specialization.
change: Replace the learned relative-shift parameter with a fixed buffer, leaving the learned lag kernel, causal attention, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Fixed-spacing phase-shared lag attention
evidence_used: Equal-temperature phase-shared attention achieved 99.93% accuracy with 1,257 parameters, showing that distinct cyclic phase routing remains sufficient after removing head-specific temperature; fixing only the remaining scalar phase is the most direct next boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9994, "parameters": 1256, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing the already-qualified head phase and temperature and tying the final two shared-kernel lag logits will reduce the model from 1,258 to 1,255 parameters while retaining at least 99% accuracy.
change: Use the qualified fixed half-period head spacing and common temperature, then constrain the two terminal cyclic-kernel coordinates to share the gauge-fixed value.
mechanism: Fixed-spacing phase sharing with tied terminal lag logits
evidence_used: Fixed-spacing phase-shared attention achieved 99.94% accuracy with 1,256 parameters; its remaining unrestricted 29-coordinate lag kernel has substantial accuracy margin, motivating a one-coordinate terminal-lag tie as the next boundary test.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1255, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified two-way terminal lag tie to three coordinates will reduce the model from 1,255 to 1,254 learned parameters while retaining at least 99% accuracy, because the unrestricted remaining lag kernel preserves nearly all routing capacity.
change: Adopt fixed `ln2` scales, fixed half-period head spacing, and common head temperature, then tie the final three cyclic-kernel logits to the gauge-fixed value.
mechanism: Three-way terminal lag-logit tying
evidence_used: The 1,255-parameter design with two tied terminal lag logits achieved 99.83% accuracy; its 0.83-point margin above the requirement motivates the next isolated one-coordinate tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9982, "parameters": 1254, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified three-coordinate terminal lag tie to four coordinates will reduce the model from 1,254 to 1,253 learned parameters while retaining at least 99% accuracy, because the remaining shared lag kernel preserves nearly all routing capacity.
change: Replace independent per-head lag tables with the qualified fixed-spacing shared cyclic kernel and constrain its final four logits to the gauge-fixed value.
mechanism: Four-way terminal lag-logit tying
evidence_used: The three-way terminal tie achieved 99.82% accuracy with 1,254 parameters, leaving a 0.82-point margin above the requirement and directly motivating the next isolated one-coordinate tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9991, "parameters": 1253, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified four-coordinate terminal lag tie to five coordinates will reduce the model from 1,253 to 1,252 learned parameters while retaining at least 99% accuracy, because the remaining shared lag kernel preserves sufficient routing capacity.
change: Constrain the final five cyclic-kernel logits to the gauge-fixed value, leaving the learned attention, optimizer, checkpoints, and decoding interface unchanged.
mechanism: Five-way terminal lag-logit tying
evidence_used: The four-way terminal tie achieved 99.91% accuracy with 1,253 parameters, and each preceding one-coordinate extension also exceeded 99%, motivating the next isolated reduction.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9163, "parameters": 1252, "training_steps": 4999}

RECENT RESULT
hypothesis: Reusing the final LayerNorm’s learned mean scale for the sole pre-attention LayerNorm scale will reduce the qualified four-terminal-tie model from 1,253 to 1,252 parameters while retaining at least 99% accuracy, because it preserves adaptive scale conditioning that was lost in the failed fully fixed `ln1` design.
change: Adopt the verified four-coordinate terminal lag tie and replace the separate one-parameter `ln1` scale with a differentiable scale shared from the existing final LayerNorm weights.
mechanism: Shared adaptive pre-attention scale
evidence_used: Four terminal lag ties achieved 99.91% at 1,253 parameters, while five ties collapsed to 91.63%; separately, fixing the last adaptive `ln1` scale reached only 97.71%, motivating a non-routing reduction that preserves learned scale adaptation through parameter sharing.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1252, "training_steps": 4999}



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
