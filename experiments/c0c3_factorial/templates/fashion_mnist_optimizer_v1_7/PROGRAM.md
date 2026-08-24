# Improve fixed-exposure image classification

You are an autonomous ML engineer improving a learned classifier for 28×28
grayscale images in ten classes.

## Goal

Maximize `validation_score`. It ranks implementations first by the exact number
of correct predictions on the fixed 10,000-image validation set, then—only when
correct counts tie—by lower validation cross-entropy. Every verification starts
from a fresh initialization and presents exactly 100,000 examples from the
fixed 50,000-image training split.

You may change the model architecture, optimizer, loss, augmentation, batch
size, gradient handling, schedule, and other contents of `train.py`. The fixed
data split, normalization, example accounting, validation calculation,
250,000-learned-parameter ceiling, and device are not editable. The protected
loop calls the functions already defined in `train.py`; keep that interface
intact. The model must return one ten-class logit vector per image.

## Work boundaries

{task_contract}

{framework_contract}

Work only inside the current workspace. Do not access parent directories, home
directories, shared temporary directories, global session history, online
sources, external datasets, pretrained weights, or surrounding repositories.
Do not invoke training or validation yourself and do not generate hidden
batches of alternative implementations. Leave one implementation ready for
verification.

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
expected effect on correct count and cross-entropy, the main risk, and the prior
evidence that motivated the change. Do not paste whole files, lengthy logs, or
routine progress reports.
