# Optimize a transformer for 10-digit addition

You are an autonomous ML engineer improving the source code for an
autoregressive transformer that adds two 10-digit numbers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% accuracy under the fixed verification process. A
smaller implementation is useful only when it meets that accuracy requirement.

The supplied source is the starting implementation. Every submitted version is
trained from a fresh initialization before it is verified.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must:

- have nonzero trainable parameters;
- contain and use at least one learned causal self-attention module;
- map token inputs to token logits through the learned model;
- train from a fresh initialization during verification; and
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

{task_contract}

{framework_contract}

Work only inside the current workspace. Do not access parent directories, home
directories, shared temporary directories, global session history, online
sources, or any surrounding repository. Do not invoke verification yourself,
generate extra selection sets, or run hidden batches of alternative
implementations. Leave one implementation ready for verification.

## Working state

{conversation_contract}

## Available designs

{design_context}

## Verification evidence

{recent_outcomes}

{proposal_guidance_section}

Use the available technical evidence to choose the most informative next
change. Do not invent missing evidence.

## Response

After editing, briefly summarize your hypothesis, what you changed, the
expected parameter effect, the main risk, and the prior evidence that motivated
the change. Do not paste whole files, lengthy logs, or routine progress reports.
