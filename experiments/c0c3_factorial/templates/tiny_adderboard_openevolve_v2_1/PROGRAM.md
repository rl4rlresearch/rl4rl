# Optimize a transformer for four-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two integers from 0 through 9999.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% exact-answer accuracy under fixed verification.
Every submitted implementation is trained from a fresh initialization.
The starting design has 1,012 parameters and uses 400 optimizer steps.

## Learned-model requirement

The model must contain and use learned causal self-attention and map token
inputs to token logits. Learned positional attention, content attention,
different token representations, embeddings, layers, widths, heads, parameter
sharing, optimizers, losses, and training schedules may all be changed.

`TRAINING_STEPS` is editable: 400 is the starting value, not a required count
or a ceiling. There is no protected training ladder. Verification performs
exactly that many optimizer updates, subject to the common wall-clock timeout.
`LR_SCHEDULE_STEPS`, batch size, and clipping are also editable. The schedule
horizon is independent of the number of updates.

Pair tokens are only the starting representation. You may replace
`encode_inputs`, `encode_targets`, and `decode_targets`, change token order,
vocabulary, grouping, sequence lengths, and `OUTPUT_TOKENS`, and coordinate
those changes with the model. Operand encoders receive two arrays of four
decimal digits (least significant first). Target encoders receive five answer
digits during training only. Target decoders receive only generated answer
tokens and must return five predicted decimal digits in least-significant-first
order. Decoders never receive the operands. Codecs must be reversible formatting;
they must not solve addition. The protected decoder repeatedly calls the model
and appends its argmax token.

Do not implement a sum solver, carry propagation, arithmetic answer tables,
finite-state addition transitions, fixed answer rules, or a disguised solver
in tokenization, model execution, training, or saved weights. Ordinary digit
formatting is permitted. Do not use dummy parameters or post-training weight
surgery. Learned attention must participate meaningfully in prediction.

Verification owns disjoint training/public/holdout data, fresh initialization,
optimizer-step accounting, parameter counting, checkpoint writing, and scoring.
It writes `checkpoints/best.pt` and positive-step `checkpoints/last.pt` from the
trained model. Keep source unchanged during evaluation. Holdout results are
post-search only. All architectural choices are open within this learned
transformer task and its common resource budget.

## Work boundaries

{task_contract}

{framework_contract}

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, or any surrounding repository. Do not run
training or verification yourself and do not generate hidden alternatives.
Return one patch for one implementation; verification happens after you finish.

## Available designs

{design_context}

## Recent verification evidence

{recent_outcomes}

{proposal_guidance_section}

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

Start each block with `<<<<<<< SEARCH`, put the exact existing lines next, use
`=======` as the divider, then replacement lines, and end with `>>>>>>> REPLACE`.
Every SEARCH section must be nonempty and match exactly once after earlier
blocks. All blocks must apply to the editable source and together produce one
implementation. Do not paste whole files, logs, or progress reports outside it.
