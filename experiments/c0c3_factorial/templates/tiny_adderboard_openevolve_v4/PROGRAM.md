# Optimize a learned transformer for four-digit addition

You are an autonomous ML engineer improving a learned autoregressive
transformer that adds two four-digit nonnegative integers.

## Goal

Minimize the actual number of deduplicated learned model parameters while
maintaining at least 99% exact-answer accuracy. A smaller implementation is
useful only when it meets that accuracy requirement. Every verification starts
from a fresh initialization.

The protected verifier trains one uninterrupted trajectory and measures public
accuracy at bounded step checkpoints. It stops at the first checkpoint reaching
99%, or at the common terminal step. You may edit the checkpoint list in
`EVALUATION_LADDER`; the verifier accepts only an increasing literal list within
its bounds and always includes the common terminal step. You may also change the
model architecture, optimizer, loss, batch size, gradient handling, schedule,
and other contents of `train.py`. Keep the four required function interfaces.

## Learned-model requirement

Produce a smaller trained autoregressive transformer, not a hand-coded addition
program. The submitted implementation must have nonzero trainable parameters,
contain and use at least one learned causal self-attention module, map token
inputs to token logits through that learned model, and train from a fresh
initialization. Exact accuracy must materially depend on the learned attention
path.

Do not implement or embed decimal arithmetic, carry propagation, place-value
rules, digit lookup tables, finite-state addition transitions, fixed answer
rules, or input-dependent Python logic that directly computes the sum. Do not
hide such a solver in model generation, token processing, training, or fixed
weights. Do not add dummy or zero-length parameters to disguise a fixed
algorithm as a learned model.

The protected verifier owns pair generation, hash-disjoint training, public,
and sealed-holdout partitions, fresh initialization, the training loop,
autoregressive decoding, exact accuracy, parameter counting, device selection,
and evaluator timing. Do not attempt to inspect, replace, or bypass them.

## Work boundaries

{task_contract}

{framework_contract}

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, external data, pretrained weights, or any
surrounding repository. Do not run training or verification yourself and do not
generate hidden alternatives. Return one patch for one implementation;
verification happens after you finish.

## Available designs

{design_context}

## Recent verification evidence

{recent_outcomes}

{proposal_guidance_section}

Use the available technical evidence to choose the most informative next
change. Treat unsuccessful or malformed work as evidence when a useful
technical reason is provided. Do not invent missing evidence.

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
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.
