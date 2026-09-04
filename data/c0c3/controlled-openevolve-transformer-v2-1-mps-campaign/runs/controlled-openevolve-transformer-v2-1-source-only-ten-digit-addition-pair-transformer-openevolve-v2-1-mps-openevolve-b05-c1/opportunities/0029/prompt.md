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
verified_results: {"accuracy": 0.9998999999999999, "parameters": 1384, "training_steps": 51999}
prior_hypothesis: Reducing `d_ff` from 2 to 1 will lower the model from 1,400 to 1,384 parameters while retaining at least 99% accuracy when trained for 52,000 steps.

## Recent verification evidence

RECENT RESULT
hypothesis: Reducing `d_ff` from 10 to 9 will lower the model from 1,528 to 1,512 parameters while retaining at least 99% accuracy when trained for 20,000 steps.
change: Use a 9-unit feed-forward layer and extend the default training budget from 16,000 to 20,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Successive reductions from `d_ff=12` to 11 and then 10 achieved 99.87%, 99.93%, and 99.98% with progressively longer training, supporting the next one-neuron reduction with another 4,000-step allowance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1512, "training_steps": 19999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 9 to 8 will lower the model from 1,512 to 1,496 parameters while retaining at least 99% accuracy when trained for 24,000 steps.
change: Use an 8-unit feed-forward layer and extend the default training budget from 20,000 to 24,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: The quotient-space model remained successful through successive `d_ff` reductions from 12 to 9 with 4,000 additional steps per neuron removed; most recently, `d_ff=9` achieved 100% accuracy at 20,000 steps, motivating the next boundary test with the same convergence allowance.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1496, "training_steps": 23999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 8 to 7 will lower the model from 1,496 to 1,480 parameters while retaining at least 99% accuracy when trained for 28,000 steps.
change: Use a 7-unit feed-forward layer and extend the default training budget from 24,000 to 28,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Successive `d_ff` reductions from 12 through 8 all exceeded 99% accuracy when paired with 4,000 additional training steps per removed neuron; most recently, `d_ff=8` achieved 99.99% at 24,000 steps with 1,496 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1480, "training_steps": 27999}

RECENT RESULT
hypothesis: Replacing independent eight-dimensional position vectors with arbitrary six-dimensional learned position codes and a shared learned projection will reduce parameters while retaining at least 99% accuracy after 40,000 steps.
change: Replace the full positional table (`8L` parameters) with a rank-six factorization (`6L + 48` parameters), orthogonally initialized and variance-scaled, while preserving the successful attention and MLP widths.
mechanism: Learned low-rank absolute-position manifold
evidence_used: The current 1,480-parameter model reached 100% accuracy, whereas fixed sinusoidal positions reached 0%; this suggests positional information is load-bearing but does not establish that every position needs an independent eight-dimensional vector. Free per-position codes preserve learned absolute addressing, and prior quotient-space models show that additional training can overcome slower convergence from constrained parameterizations.
result: met the accuracy requirement but was not a strict improvement
reported_values: {"accuracy": 0.9998, "parameters": 1482, "training_steps": 39999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 7 to 6 will lower the model from 1,480 to 1,464 parameters while retaining at least 99% accuracy when trained for 32,000 steps.
change: Use a 6-unit feed-forward layer and extend the default training budget from 28,000 to 32,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Every successive reduction from `d_ff=12` through `d_ff=7` exceeded 99% accuracy when paired with 4,000 additional steps per removed neuron; most recently, `d_ff=7` achieved 100% accuracy at 28,000 steps.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 1.0, "parameters": 1464, "training_steps": 31999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 6 to 5 will lower the model from 1,464 to 1,448 parameters while retaining at least 99% accuracy when trained for 36,000 steps.
change: Use a 5-unit feed-forward layer and extend the default training budget from 32,000 to 36,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Every successive `d_ff` reduction from 12 through 6 exceeded 99% accuracy when paired with 4,000 additional training steps per removed neuron; most recently, `d_ff=6` achieved 100% accuracy at 32,000 steps with 1,464 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9995999999999999, "parameters": 1448, "training_steps": 35999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 5 to 4 will lower the model from 1,448 to 1,432 parameters while retaining at least 99% accuracy when trained for 40,000 steps.
change: Use a 4-unit feed-forward layer and extend the default training budget from 36,000 to 40,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Every successive `d_ff` reduction from 12 through 5 met the accuracy requirement with 4,000 additional steps per removed neuron; most recently, `d_ff=5` achieved 99.96% accuracy at 36,000 steps with 1,448 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9997, "parameters": 1432, "training_steps": 39999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 4 to 3 will lower the model from 1,432 to 1,416 parameters while retaining at least 99% accuracy when trained for 44,000 steps.
change: Use a 3-unit feed-forward layer and extend the default training budget from 40,000 to 44,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Every successive `d_ff` reduction from 12 through 4 met the accuracy requirement with 4,000 additional training steps per removed neuron; most recently, `d_ff=4` achieved 99.97% accuracy at 40,000 steps with 1,432 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998, "parameters": 1416, "training_steps": 43999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 3 to 2 will lower the model from 1,416 to 1,400 parameters while retaining at least 99% accuracy when trained for 48,000 steps.
change: Use a two-unit feed-forward layer and extend the default training budget from 44,000 to 48,000 steps.
mechanism: Progressive MLP narrowing with extended convergence
evidence_used: Every successive `d_ff` reduction from 12 through 3 met the accuracy requirement with 4,000 additional training steps per removed neuron; most recently, `d_ff=3` achieved 99.98% accuracy at 44,000 steps with 1,416 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9983, "parameters": 1400, "training_steps": 47999}

RECENT RESULT
hypothesis: Reducing `d_ff` from 2 to 1 will lower the model from 1,400 to 1,384 parameters while retaining at least 99% accuracy when trained for 52,000 steps.
change: Use a one-unit feed-forward layer and extend the default training budget from 48,000 to 52,000 steps.
mechanism: Single-neuron residual MLP with extended convergence
evidence_used: Every successive `d_ff` reduction from 12 through 2 met the accuracy requirement with 4,000 additional training steps per removed neuron; most recently, `d_ff=2` achieved 99.83% accuracy at 48,000 steps with 1,400 parameters.
result: met the accuracy requirement and became an available design
reported_values: {"accuracy": 0.9998999999999999, "parameters": 1384, "training_steps": 51999}

RECENT RESULT
hypothesis: Removing the final one-neuron MLP will reduce the model from 1,384 to 1,360 learned parameters while retaining at least 99% accuracy after 56,000 training steps.
change: Remove the MLP branch from the transformer block and extend training by 4,000 steps.
mechanism: Attention-only residual decoder
evidence_used: The one-neuron MLP model achieved 99.99% accuracy at 52,000 steps, and every preceding one-neuron width reduction succeeded with a 4,000-step allowance; this motivates testing the structural boundary where attention alone performs the computation.
result: did not meet the accuracy requirement
reported_values: {"accuracy": 0.7376, "parameters": 1360, "training_steps": 55999}

RECENT RESULT
hypothesis: Centering positional features, vocabulary embeddings, and residual-branch outputs removes 42 functionally redundant parameters, reducing the model from 1,384 to 1,342 parameters while retaining at least 99% accuracy after 60,000 steps.
change: Replace full positional and token embeddings with learned mean-zero parameterizations, constrain attention and MLP residual outputs to the mean-zero feature subspace, preserve the essential one-neuron MLP, and extend training to 60,000 steps.
mechanism: Exact quotienting of LayerNorm- and softmax-invariant parameter directions
evidence_used: The 1,384-parameter one-neuron MLP model reached 99.99%, while removing that MLP collapsed accuracy to 73.76%; this motivates retaining its learned nonlinearity and instead removing only exact invariances induced by LayerNorm and softmax.
result: training did not finish within the verification time limit



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
