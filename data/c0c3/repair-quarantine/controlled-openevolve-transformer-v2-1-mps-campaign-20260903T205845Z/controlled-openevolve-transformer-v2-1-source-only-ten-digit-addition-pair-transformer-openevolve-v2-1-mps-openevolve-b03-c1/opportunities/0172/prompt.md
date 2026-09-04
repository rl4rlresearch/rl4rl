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
verified_results: {"accuracy": 0.9989, "parameters": 606, "training_steps": 4999}
prior_hypothesis: Fixing an adaptive four-column key basis to identity in each attention head will reduce the verified 638-parameter model to 606 parameters while retaining at least 99% accuracy, because the transformation preserves every initialized attention score and the model’s full learned query/key function class.

## Recent verification evidence

RECENT RESULT
hypothesis: Replacing four independently learned harmonic amplitudes with their exact initialized spectral template plus one learned global gain will reduce the verified model from 606 to 603 parameters while retaining at least 99% accuracy.
change: Preserve the initial positional representation exactly, freeze its relative harmonic amplitudes, and learn only a shared additive positional-strength adjustment.
mechanism: Shared-gain fixed-spectrum positional encoding
evidence_used: The 606-parameter model reached 0.9989 accuracy using fixed harmonic position codes and learned relative-distance routing, while the failed 604-parameter experiment altered the fragile lexical representation; this instead tests whether independent positional amplitudes are necessary while leaving lexical computation unchanged.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.565, "parameters": 603, "training_steps": 4999}

RECENT RESULT
hypothesis: Sharing the final head-specific relative-bias column will reduce the verified 606-parameter model to 605 parameters while retaining at least 99% accuracy, because the initialized function is unchanged and the following nine distance biases already generalize with cross-head sharing.
change: Shorten the head-specific relative-bias table by one column and restore the removed boundary column as one learned scalar shared across both attention heads.
mechanism: Boundary relative-distance bias sharing
evidence_used: The current 606-parameter design achieved 0.9989 accuracy; the failed 603- and 604-parameter changes altered positional or lexical representations, so a one-parameter reduction within the already successful bias-sharing scheme is the most conservative informative test.
result: the implementation could not be verified

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
mechanism: [missing mechanism]
evidence_used: [missing evidence]
result: no usable edit was produced

RECENT RESULT
hypothesis: Replacing the 30-parameter partially shared distance lookup with 20 learned head-specific DCT coefficients will reduce the model from 606 to 596 parameters while retaining at least 99% accuracy, because ten smooth modes per head preserve flexible causal routing without constraining the fragile lexical or absolute-position representations.
change: Parameterize each attention head’s gauge-fixed relative bias as a learned ten-mode cosine expansion over the currently active distance range, while retaining the eleven fixed-zero long-distance endpoints.
mechanism: Spectral relative-distance routing
evidence_used: The 606-parameter model reached 0.9989 accuracy with extensive sharing and fixed endpoints in its relative-bias table, indicating that unconstrained per-distance biases are not all essential. The 603-parameter fixed-spectrum positional experiment fell to 0.565, so this patch leaves all four independently learned absolute-position amplitudes and the lexical path unchanged and instead tests a different, fully learned spectral routing mechanism.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.0524, "parameters": 616, "training_steps": 4999}

RECENT RESULT
hypothesis: Fixing one token-radius coordinate will reduce the verified 606-parameter model to 605 parameters while retaining at least 99% accuracy, because a reciprocal rescaling of all token radii, the shared lexical projection, and the final lexical bias makes this coordinate functionally redundant while preserving the initialized model exactly.
change: Store the first initialized token radius as a fixed buffer and learn only the remaining radii, reconstructing the complete radius vector during the forward pass.
mechanism: Lexical global-scale gauge fixing
evidence_used: The 606-parameter design achieved 0.9989 accuracy, while the failed 604-parameter lexical experiment changed the lexical representation itself; this patch instead removes only an exact continuous scale redundancy without changing its initialization or representable learned functions.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.9504, "parameters": 605, "training_steps": 4999}



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
