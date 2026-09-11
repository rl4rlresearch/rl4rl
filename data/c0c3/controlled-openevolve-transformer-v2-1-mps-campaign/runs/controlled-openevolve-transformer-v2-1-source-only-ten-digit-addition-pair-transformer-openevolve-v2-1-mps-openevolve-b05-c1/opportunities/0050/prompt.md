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
verified_results: {"accuracy": 1.0, "parameters": 1372, "training_steps": 49999}
prior_hypothesis: Sharing one MLP output-bias coordinate with the learned attention query offset will reduce the model from 1,373 to 1,372 parameters while retaining at least 99% accuracy within 50,000 steps, because both successful 1,373- and 1,376-parameter designs already use bias coordinates for multiple roles without changing their zero initialization.

## Recent verification evidence

RECENT RESULT
hypothesis: Removing the eight-parameter attention output bias will reduce the model from 1,384 to 1,376 parameters while retaining at least 99% accuracy within the proven 52,000-step schedule.
change: Disable only the bias of the standard attention output linear layer, preserving the terminal LayerNorm, one-neuron MLP, training budget, and fast built-in operators.
mechanism: Bias-free attention output projection
evidence_used: The current model achieved 99.99% accuracy at 52,000 steps. Previous 1,376-parameter attempts changed the terminal LayerNorm and timed out, so removing an attention projection bias instead provides an informative eight-parameter reduction without the implicated LayerNorm path or added reconstruction operations.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Constraining each query-bias vector to one learned scalar will reduce the model from 1,384 to 1,378 parameters while retaining at least 99% accuracy after 52,000 steps.
change: Replace the eight-coordinate query bias with one broadcast scalar per attention head, preserving the existing fast linear and LayerNorm paths.
mechanism: Per-head gauge-fixed query bias
evidence_used: The current model reached 99.99% accuracy at 52,000 steps, while recent reductions that altered linear or LayerNorm kernels timed out; query/key coordinates have a rotational gauge, so fixing each head’s query-bias direction removes six redundant degrees of freedom without removing its learned content-independent attention term.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Decoupling each head’s two-dimensional query/key routing space from its four-dimensional value space will reduce the model from 1,384 to 1,324 parameters while retaining at least 99% accuracy after 52,000 steps; batch size 256 and less frequent validation will keep verification within the time limit.
change: Replace full-width queries and keys with learned two-dimensional per-head address codes while preserving full-width values, residuals, tied embeddings, and the essential one-neuron MLP; halve the training batch and validate every 1,000 steps.
mechanism: Two-dimensional attention-routing bottleneck with full-width values
evidence_used: The current full-width model reached 99.99%, and a rank-six positional factorization reached 99.98%, indicating that positional routing tolerates a lower-dimensional learned representation. In contrast, removing the one-neuron MLP collapsed accuracy to 73.76%, so this patch preserves the nonlinear and value-processing capacity while testing whether full query/key width is the load-bearing assumption.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: A per-head scalar query bias will reduce the model from 1,384 to 1,378 parameters while retaining at least 99% accuracy at 52,000 steps; eliminating basis multiplications, per-example hash checks, and intermediate validation will allow verification to finish.
change: Represent mean-zero LayerNorm outputs by their first seven coordinates, constrain query bias to one scalar per head, accelerate exact holdout exclusion, and validate only at the final positive step.
mechanism: Zero-copy normalized-input chart with gauge-fixed query bias
evidence_used: The prior 1,378-parameter query-bias model timed out without contrary accuracy evidence, while the 1,384-parameter model reached 99.99%; repeated timeouts motivate preserving the proven training budget while removing avoidable runtime overhead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1378, "training_steps": 51999}

RECENT RESULT
hypothesis: Fixing each head’s scalar query offset to one will reduce the model from 1,378 to 1,376 parameters while retaining at least 99% accuracy at 52,000 steps, because inverse query/key rescaling can absorb any nonzero per-head scalar without reducing attention expressivity.
change: Remove the two learned per-head query-bias parameters and replace their broadcast with a fixed unit offset, preserving all optimized operators and training settings.
mechanism: Gauge-fixed unit query offset
evidence_used: The current gauge-fixed scalar-query model achieved 100% accuracy with 1,378 parameters; unlike earlier 1,376-parameter attempts that changed LayerNorm or projection kernels and timed out, this removes only a query/key scaling redundancy from the verified implementation.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Sharing one learned query-offset scalar across both heads will reduce the model from 1,378 to 1,377 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Replace the two per-head query-bias parameters with one broadcast learned scalar initialized at zero, and shorten training by 2,000 steps for runtime margin.
mechanism: Shared trainable query-offset gauge
evidence_used: The 1,378-parameter scalar-query model achieved 100% accuracy at 52,000 steps, while fixing the offsets to one timed out; retaining a trainable zero-initialized offset preserves the successful optimization path, and independent query/key weights can absorb head-specific rescaling.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1377, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing the functionally redundant common-mode attention projection bias as the learned query offset will reduce the model from 1,377 to 1,376 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Remove the standalone query-offset parameter and derive its zero-initialized learned value from the mean attention projection bias.
mechanism: Attention-output gauge reuse
evidence_used: The 1,377-parameter model achieved 100% accuracy at 50,000 steps, whereas fixing the offset to one changed its successful zero-initialized optimization path and timed out. The projection bias’s common-mode component is erased by downstream LayerNorms, so it can encode the query offset without reducing effective model capacity.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1376, "training_steps": 49999}

RECENT RESULT
hypothesis: Representing each positional embedding with seven learned coordinates and a fixed-zero eighth coordinate will reduce the model by `INPUT_LEN` parameters while retaining at least 99% accuracy within 50,000 steps, because position-dependent common-mode shifts are erased by every LayerNorm.
change: Reduce positional embeddings from eight to seven learned coordinates and pad the omitted coordinate with zero during the forward pass.
mechanism: Fixed-last-coordinate positional quotient
evidence_used: The current 1,376-parameter model achieved 100% accuracy at 50,000 steps, and a more restrictive rank-six positional factorization previously achieved 99.98%; this seven-coordinate chart removes only the exact LayerNorm-invariant common mode and adds one small padding operation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.696, "parameters": 1353, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing the LayerNorm-invisible common mode of the MLP output bias as its hidden-unit bias will reduce the model from 1,376 to 1,375 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Remove the standalone `fc1` bias and derive its zero-initialized learned value from the mean of `fc2.bias`.
mechanism: MLP output-gauge reuse
evidence_used: The current 1,376-parameter model achieved 100% accuracy after successfully reusing the attention-output bias common mode; the MLP output is likewise followed by LayerNorm, making its bias common mode functionally redundant and available for the same reuse.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1375, "training_steps": 49999}

RECENT RESULT
hypothesis: Removing the LayerNorm-invisible common mode of the one-neuron MLP’s output weight will reduce the model from 1,375 to 1,374 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Emit seven learned MLP output coordinates and append a fixed-zero eighth coordinate, while retaining the independently learned hidden bias as a scalar.
mechanism: MLP activation-dependent output gauge quotient
evidence_used: The 1,375-parameter model achieved 99.99% accuracy at 50,000 steps. Unlike the failed positional quotient, this isolates a single redundant output-weight direction; the MLP branch’s activation-dependent common-mode output is erased by downstream LayerNorm.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1374, "training_steps": 49999}

RECENT RESULT
hypothesis: Reusing the mean of the seven-coordinate MLP output bias as the hidden-unit bias will reduce the model from 1,374 to 1,373 parameters while retaining at least 99% accuracy within 50,000 steps.
change: Remove the standalone hidden-bias parameter and derive its zero-initialized value from `fc2.bias`.
mechanism: Shared MLP bias coordinate
evidence_used: The 1,375-parameter model achieved 99.99% accuracy while deriving its hidden bias from the mean of `fc2.bias`; the current 1,374-parameter model achieved 100%, so this directly tests whether that successful sharing remains sufficient after quotienting the MLP output weight.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1373, "training_steps": 49999}

RECENT RESULT
hypothesis: Sharing one MLP output-bias coordinate with the learned attention query offset will reduce the model from 1,373 to 1,372 parameters while retaining at least 99% accuracy within 50,000 steps, because both successful 1,373- and 1,376-parameter designs already use bias coordinates for multiple roles without changing their zero initialization.
change: Replace the seven-parameter `fc2` bias with six dedicated coordinates plus the attention projection-bias mean as the seventh coordinate, while preserving the same seven-coordinate MLP output, derived hidden bias, and optimized `F.linear` computation.
mechanism: Cross-branch bias-coordinate sharing
evidence_used: The 1,373-parameter model achieved 100% accuracy after reusing the MLP output-bias mean as its hidden bias, and the 1,376-parameter model achieved 100% after reusing the attention output-bias mean as its query offset; sharing one coordinate across these proven learned-bias mechanisms is the smallest direct next reduction.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1372, "training_steps": 49999}

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
