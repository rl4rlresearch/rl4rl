# Improve a small sensor-sequence classifier

You are an autonomous ML engineer improving a learned classifier for six
activities from 128 time steps of nine inertial sensor channels.

## Goal

Increase validation accuracy and reduce inference multiply-accumulate operations
(MACs) per example. These are two separate objectives. A new design is eligible
for retention when no earlier valid evaluated design is at least as accurate and
at most as expensive, with a strict improvement in one objective. Exact objective
ties keep the earlier design. Previously discarded designs also count in this
comparison; their source is not restored or made available.

There is no minimum accuracy gate. The available designs are a bounded working
memory, not necessarily the complete current nondominated frontier. When memory
is full, an eligible child replaces its selected parent. Progress reporting uses
the dominated area of all evaluated designs, not a weighted candidate score.

## Model and training interface

Edit `train.py`: model architecture, initialization, optimizer, loss and learning
rate schedule. `build_model()` returns a CPU PyTorch model mapping `[batch,128,9]`
to six finite logits. `BATCH_SIZE` is an integer from 16 to 512.
`build_optimizer(model,total_steps)`, `training_loss(model,features,labels,step,total_steps)`
and `after_optimizer_step(optimizer,step,total_steps)` define training.
`GRAD_CLIP_NORM` controls gradient clipping. All models start from fresh weights.

Verification owns training data, person-disjoint splits, training-only channel
normalization, exactly 50,000 training-example exposures, checkpoints and scoring.
Only final model predictions are scored. There is no validation-driven early
stopping, pretrained state, or access to validation labels in training.

The model may have up to 100,000 learned parameters and 2,000,000 inference MACs
per example. Counted primitives are standard `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`,
`nn.RNN`, `nn.GRU`, `nn.LSTM` and their cell variants. Both directions and all
layers are counted. Biases, normalization, activation, pooling and elementwise
gate operations are excluded from MACs; MACs are not measured device latency.
Use these standard modules for affine computation; uncounted functional affine
operations, projected LSTMs, and overridden primitive implementations are rejected.
BatchNorm, LayerNorm, GroupNorm, pooling and ordinary tensor reshaping are allowed.
Bidirectional recurrence is allowed because the complete input window is available.

Use only torch, math and typing imports. Do not access files, network, other
processes, pretrained weights or runtime introspection. Do not alter thread limits,
protected data, evaluator behavior or source during verification. Avoid cached
training examples, hard-coded label rules and unregistered learned state.

## Work boundaries

{task_contract}

{framework_contract}

The editable source and reference source are supplied below. Do not run training
or verification yourself. Submit one implementation; verification runs afterward.

## Available designs

{design_context}

## Recent verification evidence

{recent_outcomes}

{proposal_guidance_section}

## Response

Return `MECHANISM:`, `HYPOTHESIS:`, `INTENDED_EDIT:` and `EVIDENCE:` metadata lines,
then one or more exact SEARCH/REPLACE blocks for the single implementation.
Each starts with `<<<<<<< SEARCH`, then exact existing source, `=======`, the
replacement source, and `>>>>>>> REPLACE`. Every nonempty SEARCH must match
exactly once after preceding blocks. Keep whole files, logs and progress reports
outside the response. Ground the change in the available technical evidence.
