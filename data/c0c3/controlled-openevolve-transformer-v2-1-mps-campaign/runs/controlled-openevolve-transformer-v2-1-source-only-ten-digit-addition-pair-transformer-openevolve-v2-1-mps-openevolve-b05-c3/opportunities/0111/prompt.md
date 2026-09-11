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
verified_results: {"accuracy": 0.998, "parameters": 1246, "training_steps": 4999}
prior_hypothesis: Extending the verified six-way final-scale tie to seven channels will reduce the model from 1,247 to 1,246 learned parameters while retaining at least 99% accuracy, because two learned final-scale values and the shared adaptive pre-attention scale remain.

REFERENCE DESIGN 1
verified_results: {"accuracy": 0.9990000000000001, "parameters": 1130, "training_steps": 4999}
prior_hypothesis: Tying the final two bias coordinates will reduce the qualified 1,131-parameter model to 1,130 parameters while retaining at least 99% accuracy, because the adaptive normalization scale is preserved and only one latent output-bias degree of freedom is removed.

REFERENCE DESIGN 2
verified_results: {"accuracy": 0.9987999999999999, "parameters": 1245, "training_steps": 4999}
prior_hypothesis: Extending the verified seven-way final-scale tie to all eight channels will reduce the best qualified model from 1,246 to 1,245 parameters while retaining at least 99% accuracy, because the shared scale remains learned and continues to condition both final normalization and pre-attention normalization.

REFERENCE DESIGN 3
verified_results: {"accuracy": 0.9984000000000001, "parameters": 1250, "training_steps": 4999}
prior_hypothesis: Extending the qualified final-LayerNorm tie from two channels to three will reduce the model from 1,251 to 1,250 learned parameters while retaining at least 99% accuracy, because six independent final scales and the shared adaptive `ln1` signal remain.

## Recent verification evidence

RECENT RESULT
hypothesis: Combining the qualified four-terminal lag tie and shared adaptive `ln1` scale with one tied pair of final-LayerNorm scales will produce a 1,251-parameter model while retaining at least 99% accuracy, because seven independent final scales remain and prior LayerNorm-scale reductions were substantially more tolerant than the failed fifth routing-logit tie.
change: Adopt the verified 1,252-parameter design, then reconstruct the final LayerNorm from seven learned scales by sharing its last learned scale across the final two channels.
mechanism: Shared adaptive pre-attention scale with a two-way final-LayerNorm scale tie
evidence_used: The four-terminal-tie plus shared adaptive `ln1` design achieved 99.85% with 1,252 parameters, while fixing all `ln2` scales achieved 99.94% and seven-coordinate `ln1` quotienting achieved 99.96%; these results motivate removing one non-routing scale degree of freedom instead of extending the terminal lag tie that collapsed to 91.63%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9976999999999999, "parameters": 1251, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the assumed full-rank eight-coordinate token table with learned seven-dimensional token codes and a shared learned basis will reduce the model from 1,256 to 1,200 parameters while retaining at least 99% accuracy.
change: Factorize the tied input/output embedding into a learned rank-seven codebook and learned 7×8 basis, initialize it as a stable seven-channel embedding, and train both factors with regular AdamW.
mechanism: Learned rank-seven tied token manifold
evidence_used: The 1,256-parameter fixed-spacing design achieved 99.94% accuracy, while every available design retains a full-rank tied embedding and concentrates reductions in routing or normalization; this tests that unchallenged representation assumption with substantially more potential progress.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0718, "parameters": 1199, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified final-LayerNorm tie from two channels to three will reduce the model from 1,251 to 1,250 learned parameters while retaining at least 99% accuracy, because six independent final scales and the shared adaptive `ln1` signal remain.
change: Reconstruct the eight-channel final LayerNorm from six learned scales by sharing its last learned scale across the final three channels.
mechanism: Three-way final-LayerNorm scale tying
evidence_used: The current two-way final-scale tie achieved 99.77% accuracy with 1,251 parameters, while complete `ln2` scale absorption also retained 99.94%; this motivates another isolated non-routing scale tie instead of the fifth lag-logit tie that collapsed to 91.63%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984000000000001, "parameters": 1250, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified three-way final-scale tie to four channels will reduce the model from 1,250 to 1,249 parameters while retaining at least 99% accuracy, because five independent final scales and the shared adaptive pre-attention signal remain.
change: Add a final LayerNorm reconstructed from five learned scales with its last four channel scales tied, and derive the shared `ln1` scale from the reconstructed full scale vector.
mechanism: Four-way final-LayerNorm scale tying
evidence_used: The three-way final-scale tie achieved 99.84% accuracy with 1,250 parameters, after the two-way tie achieved 99.77%; this supports testing one further isolated final-scale tie instead of reducing the routing kernel whose fifth terminal tie collapsed to 91.63%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9984999999999999, "parameters": 1249, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified four-way final-scale tie to five channels will reduce the model from 1,249 to 1,248 learned parameters while retaining at least 99% accuracy, because four independent final scales and the shared adaptive pre-attention scale remain.
change: Adopt the qualified shared adaptive `ln1` scale, reconstruct the final LayerNorm from four learned scales with its last five channel scales tied, and leave attention, training, checkpointing, and decoding unchanged.
mechanism: Five-way final-LayerNorm scale tying
evidence_used: The four-way final-scale tie achieved 99.85% accuracy with 1,249 parameters after the two-way and three-way ties also exceeded 99%, directly motivating one further isolated final-scale tie.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1248, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the qualified five-way final-scale tie to six channels will reduce the model from 1,248 to 1,247 learned parameters while retaining at least 99% accuracy, because three independent final scales and the shared adaptive pre-attention scale remain.
change: Adopt the qualified four-terminal lag tie and shared adaptive `ln1` scale, then reconstruct the final LayerNorm from three learned scales with its last six channel scales tied.
mechanism: Six-way final-LayerNorm scale tying
evidence_used: The five-way final-scale tie achieved 99.83% accuracy with 1,248 parameters, after the preceding two-way through four-way ties also exceeded 99%; this motivates one further isolated scale tie while avoiding the fifth terminal lag tie that collapsed to 91.63%.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1247, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified six-way final-scale tie to seven channels will reduce the model from 1,247 to 1,246 learned parameters while retaining at least 99% accuracy, because two learned final-scale values and the shared adaptive pre-attention scale remain.
change: Reconstruct the eight-channel final LayerNorm from two learned scales by sharing its last learned scale across the final seven channels.
mechanism: Seven-way final-LayerNorm scale tying
evidence_used: Six-way final-scale tying achieved 99.83% accuracy with 1,247 parameters, and every preceding extension from two-way through five-way also exceeded 99%, motivating one further isolated scale tie while preserving the verified four-terminal attention kernel.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.998, "parameters": 1246, "training_steps": 4999}

RECENT RESULT
hypothesis: Extending the verified seven-way final-scale tie to all eight channels will reduce the best qualified model from 1,246 to 1,245 parameters while retaining at least 99% accuracy, because the shared scale remains learned and continues to condition both final normalization and pre-attention normalization.
change: Reconstruct the final LayerNorm from one learned scalar shared across all eight channels, leaving the verified four-terminal attention kernel and training process unchanged.
mechanism: Single adaptive final-LayerNorm scale
evidence_used: Seven-way final-scale tying achieved 99.80% accuracy with 1,246 parameters, and every preceding extension also exceeded 99%; this motivates the final isolated scale tie while preserving the adaptive signal whose complete removal previously underperformed.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9987999999999999, "parameters": 1245, "training_steps": 4999}

RECENT RESULT
hypothesis: Constraining only the final-bias common mode enables an exact per-token embedding row-shift quotient, reducing the qualified 1,245-parameter design to `1245 - vocab_size` parameters while retaining at least 99% accuracy because the learned shared final/pre-attention scale remains intact.
change: Use seven-coordinate token embeddings with a fixed last channel, reconstruct a zero-sum seven-coordinate final bias, retain one learned scalar final scale, and train both quotient representations with the existing gauge-aware optimizer.
mechanism: Per-token embedding row-shift quotient with zero-sum final bias
evidence_used: The single adaptive final scale achieved 99.88% accuracy at 1,245 parameters, while removing the adaptive scale previously underperformed; this preserves that scale and instead removes one final-bias common mode, which makes every token-specific embedding row shift invisible to both the pre-LayerNorm network and tied output head.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9981, "parameters": 1131, "training_steps": 4999}

RECENT RESULT
hypothesis: Starting from the qualified 1,131-parameter quotient design, deriving its sole final/pre-attention scale from the learned mean MLP input bias will reduce the model to 1,130 parameters while retaining at least 99% accuracy, because normalization remains adaptively scaled rather than fixed.
change: Adopt the verified per-token embedding row-shift and zero-sum final-bias quotients, then remove the standalone normalization-scale parameter by reconstructing a positive scale from the existing learned MLP bias.
mechanism: MLP-bias-sourced adaptive normalization with per-token embedding quotient
evidence_used: The embedding/final-bias quotient achieved 99.81% accuracy with 1,131 parameters, while eliminating the last adaptive pre-attention scale reached only 97.71%; this motivates removing its storage through sharing while preserving a learned adaptive signal.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.8817, "parameters": 1130, "training_steps": 4999}

RECENT RESULT
hypothesis: Tying the final two bias coordinates will reduce the qualified 1,131-parameter model to 1,130 parameters while retaining at least 99% accuracy, because the adaptive normalization scale is preserved and only one latent output-bias degree of freedom is removed.
change: Reconstruct the zero-sum final LayerNorm bias from six learned coordinates with its final two coordinates tied, leaving attention, training, gauge-aware optimization, checkpoints, and decoding unchanged.
mechanism: Two-way final-LayerNorm bias tying
evidence_used: The current quotient design achieved 99.81% accuracy with 1,131 parameters, while sourcing its adaptive scale elsewhere collapsed to 88.17%; this motivates preserving the scale and testing an isolated final-bias tie instead.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9990000000000001, "parameters": 1130, "training_steps": 4999}

RECENT RESULT
hypothesis: Replacing the two independent four-feature value maps in the qualified 1,130-parameter quotient model with one learned map shared across the shifted attention heads will produce 1,102 parameters while retaining at least 99% accuracy, because the heads can reuse token features while their distinct routing patterns and output-projection slices preserve their roles.
change: Adopt the verified per-token embedding and tied zero-sum final-bias quotients, then broadcast one learned four-dimensional value representation to both attention heads instead of learning separate value encoders.
mechanism: Shared cross-head value encoder
evidence_used: The 1,130-parameter quotient design achieved 99.90%, whereas rank-seven token factorization collapsed to 7.18% and a fifth routing-logit tie collapsed to 91.63%; this motivates preserving token geometry and routing while challenging the shared assumption that shifted heads need independent value feature extractors.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.06309999999999999, "parameters": 1102, "training_steps": 4999}



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
