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

Maximize validation_score. No additional accuracy threshold.
Editable source files: train.py.
Results reported after each verification: validation_score, validation_correct, validation_accuracy, validation_cross_entropy, parameters, examples_processed, optimizer_steps, training_seconds, batch_size.

Propose changes through exact SEARCH/REPLACE blocks. The patching interface applies them to the supplied editable source.

The editable source and any reference source are included below. Do not access
parent directories, home directories, shared temporary directories, global
session history, online sources, external datasets, pretrained weights, or any
surrounding repository. Do not run training or validation yourself and do not
generate hidden alternatives. Return one patch for one implementation;
verification happens after you finish.

## Available designs

The current editable design and the qualified reference designs below are available as technical evidence. Edit only the current workspace.

CURRENT DESIGN
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245151, "training_seconds": 51.79995895805769, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2081755744934082, "validation_score": 9279.413847134932}
prior_hypothesis: Sharing one 90%-max mixture coefficient across both pooling stages at the verified 2.0e-3 learning rate will exceed 9,284 correct predictions by retaining the successful anti-aliasing effect while avoiding the harmful pooling-granularity increase seen with channelwise coefficients.

REFERENCE DESIGN 1
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 51.86431899992749, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20815952377319336, "validation_score": 9284.413852633003}
prior_hypothesis: Replacing hard max pooling with a 90%-max learnable mixture will exceed 9,283 correct predictions by reducing translation aliasing while preserving the baseline’s edge-selective behavior.

REFERENCE DESIGN 2
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 51.639847041806206, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.208310994720459, "validation_score": 9281.413800753437}
prior_hypothesis: Independent 90%-max pooling gates trained at half the backbone learning rate with beta2=0.96 will exceed 9,284 correct predictions by permitting stage specialization while limiting drift from the empirically best initialization.

REFERENCE DESIGN 3
verified_results: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245222, "training_seconds": 41.75457075005397, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2084277919769287, "validation_score": 9281.413760758665}
prior_hypothesis: Extending the 9,284-correct mixed-pooling design from one mixture coefficient per stage to one per feature channel will exceed 9,284 correct by letting edge-selective and region-selective channels choose different pooling behavior with only 72 additional parameters.

## Recent verification evidence

RECENT RESULT
hypothesis: Expanding the classifier hidden width from 140 to 144 at the verified 2.0e-3 peak and beta2=0.96 will exceed 9,283 correct predictions by using the remaining parameter budget to preserve more discriminative information from the established spatial representation.
change: Restore the best verified learning rate and widen only the classifier’s hidden layer, LayerNorm, and output projection, increasing parameters from 245,150 to approximately 249,298.
mechanism: Proven-feature classifier bottleneck expansion
evidence_used: The unchanged backbone achieved the best result of 9,283 correct at 2.0e-3 and beta2=0.96; auxiliary covariance and multi-scale features regressed, motivating additional capacity in the proven classifier path instead of introducing another representation mechanism.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 249298, "training_seconds": 57.40010291687213, "validation_accuracy": 0.9239, "validation_correct": 9239, "validation_cross_entropy": 0.21131550674438476, "validation_score": 9239.41277437399}

RECENT RESULT
hypothesis: Replacing hard max pooling with a 90%-max learnable mixture will exceed 9,283 correct predictions by reducing translation aliasing while preserving the baseline’s edge-selective behavior.
change: Add a two-parameter mixed-pooling module and use it at both downsampling stages; preserve all other architecture, optimization, loss, augmentation, schedule, and TTA settings.
mechanism: Learnable max–average mixed pooling
evidence_used: The 9,283-correct baseline depends heavily on translation augmentation and TTA, while changing the translation kernel or probability-space marginalization regressed. This motivates improving translation stability inside the representation with minimal added capacity.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 51.86431899992749, "validation_accuracy": 0.9284, "validation_correct": 9284, "validation_cross_entropy": 0.20815952377319336, "validation_score": 9284.413852633003}

RECENT RESULT
hypothesis: Adding three learned residual prototypes per class to the 9,284-correct mixed-pooling design will exceed 9,284 correct predictions by modeling class-specific appearance modes without disturbing the proven shared representation or base linear classifier.
change: Restore the best verified mixed pooling and beta2=0.96, then replace each class’s single affine score with a preserved base score plus the strongest of three learned residual evidence templates.
mechanism: Residual multi-prototype maxout class scoring
evidence_used: Learnable mixed pooling achieved the best result at 9,284 correct, while widening the shared classifier fell to 9,239; this challenges the load-bearing assumption that added capacity should be shared and instead allocates the remaining parameter budget to conditional, class-specific prediction paths.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Extending the 9,284-correct mixed-pooling design from one mixture coefficient per stage to one per feature channel will exceed 9,284 correct by letting edge-selective and region-selective channels choose different pooling behavior with only 72 additional parameters.
change: Restore beta2=0.96 and replace both hard max-pooling layers with 90%-max channelwise learnable mixtures.
mechanism: Channelwise learnable max–average pooling
evidence_used: Scalar mixed pooling improved the best verified result to 9,284 correct with lower cross-entropy, while larger shared-capacity additions regressed; a channelwise extension preserves that successful mechanism with minimal capacity and runtime overhead.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245222, "training_seconds": 41.75457075005397, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.2084277919769287, "validation_score": 9281.413760758665}

RECENT RESULT
hypothesis: Sharing one 90%-max mixture coefficient across both pooling stages at the verified 2.0e-3 learning rate will exceed 9,284 correct predictions by retaining the successful anti-aliasing effect while avoiding the harmful pooling-granularity increase seen with channelwise coefficients.
change: Replace both hard max pools with one shared learnable max–average pool and restore the accuracy-optimal 2.0e-3 optimizer and schedule peak.
mechanism: Stage-tied learnable max–average pooling
evidence_used: Two stage-specific scalar mixtures achieved the best result of 9,284 correct, whereas expanding them to 72 channelwise coefficients fell to 9,281; this tests whether stronger sharing improves the successful mechanism’s statistical stability.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245151, "training_seconds": 51.79995895805769, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.2081755744934082, "validation_score": 9279.413847134932}

RECENT RESULT
hypothesis: EMA-averaging the final 40% of the 9,284-correct mixed-pooling model’s trajectory will exceed 9,284 correct predictions by reducing late-step parameter and BatchNorm noise without changing the proven representation.
change: Register an AdamW post-step hook that averages parameters and floating-point buffers with decay 0.98 from 60% training progress, then installs the averaged checkpoint after the final update.
mechanism: Late-training exponential checkpoint averaging
evidence_used: Scalar mixed pooling is the best design at 9,284 correct, while added capacity, stronger consistency regularization, and smaller batches regressed or timed out; checkpoint averaging preserves that design and targets optimization variance instead.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 48.283058542059734, "validation_accuracy": 0.9259, "validation_correct": 9259, "validation_cross_entropy": 0.21012696838378905, "validation_score": 9259.413179784488}

RECENT RESULT
hypothesis: Initializing independent stage-level mixtures halfway between the 9,284-correct 90%-max design and hard max pooling will exceed 9,284 correct by retaining anti-aliasing while preserving more edge-selective evidence during early optimization.
change: Replace channelwise pooling coefficients with one independent coefficient per stage and initialize each mixture to 95% max pooling; preserve all other settings.
mechanism: Stage-specific 95%-max learnable mixed pooling
evidence_used: Stage-specific 90%-max pooling achieved the best result at 9,284 correct, while channelwise coefficients fell to 9,281 and a shared coefficient fell to 9,279, identifying stage-level granularity as best and motivating a local interpolation toward hard max.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 60.55497175012715, "validation_accuracy": 0.9268, "validation_correct": 9268, "validation_cross_entropy": 0.20886294555664062, "validation_score": 9268.413611817483}

RECENT RESULT
hypothesis: Giving the two stage-specific 90%-max mixture logits twice the backbone learning rate will exceed 9,284 correct predictions by allowing faster stage specialization within the fixed 522-step budget.
change: Restore the best verified scalar mixed-pooling architecture and beta2=0.96, isolate its two mixture logits in a 4.0e-3 optimizer group, and preserve the established 2.0e-3 cosine schedule for all other parameters.
mechanism: Accelerated stage-specific mixed-pooling adaptation
evidence_used: Independent stage-level mixtures reached 9,284 correct, outperforming a shared coefficient at 9,279 and channelwise coefficients at 9,281; this indicates stage-specific adaptation is useful while finer granularity is harmful.
result: training did not finish within the verification time limit

RECENT RESULT
hypothesis: Independent pooling stages initialized at 87.5% max will exceed 9,284 correct predictions by modestly strengthening the anti-aliasing that made 90%-max pooling successful, while retaining stage specialization.
change: Restore independent scalar pooling gates and lower their initialization from 90% to 87.5% max pooling; preserve all other settings.
mechanism: Stage-specific 87.5%-max learnable mixed pooling
evidence_used: Independent 90%-max pooling achieved the best 9,284 correct, whereas moving toward hard max with 95% fell to 9,268 and sharing the gate fell to 9,279, motivating a local search in the opposite direction without sacrificing stage independence.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 46.28965483303182, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20842137985229492, "validation_score": 9279.413762954162}

RECENT RESULT
hypothesis: Initializing the first pooling stage at 87.5% max and the second at 92.5% max will exceed 9,284 correct predictions by strengthening early anti-aliasing while preserving sharper semantic evidence later, without changing the successful mean pooling bias or parameter count.
change: Parameterize each scalar pooling gate’s initialization and assign complementary stage-specific max weights averaging 90%; preserve all other architecture, optimization, augmentation, loss, schedule, and TTA settings.
mechanism: Depth-asymmetric mixed-pooling initialization
evidence_used: Independent 90%-max stage gates achieved the best result of 9,284 correct, while tying them fell to 9,279 and moving both together to 87.5% or 95% fell to 9,279 and 9,268; this motivates testing depth-specific differentiation while keeping the aggregate initialization centered on the verified optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 49.43539558397606, "validation_accuracy": 0.9276, "validation_correct": 9276, "validation_cross_entropy": 0.2090347053527832, "validation_score": 9276.413553058308}

RECENT RESULT
hypothesis: Coupling the two pooling gates around the verified 90%-max optimum while learning only their stage-to-stage difference will exceed 9,284 correct predictions by retaining useful stage specialization without noisy common-mode drift.
change: Replace channelwise pooling coefficients with one shared contrast parameter that moves the two scalar stage gates in opposite directions around a fixed 90%-max center.
mechanism: Mean-preserving stage-contrast mixed pooling
evidence_used: Independent 90%-max scalar gates achieved 9,284 correct, while one tied gate achieved 9,279 and 72 channelwise gates achieved 9,281; moving both scalar initializations to either 87.5% or 95% also regressed, motivating stage differentiation constrained around the established optimum.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245151, "training_seconds": 58.149861792102456, "validation_accuracy": 0.9279, "validation_correct": 9279, "validation_cross_entropy": 0.20840261840820312, "validation_score": 9279.413769378172}

RECENT RESULT
hypothesis: Independent 90%-max pooling gates trained at half the backbone learning rate with beta2=0.96 will exceed 9,284 correct predictions by permitting stage specialization while limiting drift from the empirically best initialization.
change: Restore the best verified two-gate mixed-pooling architecture and beta2=0.96, while assigning the pooling logits a 0.5× learning-rate multiplier throughout the cosine schedule.
mechanism: Slow-timescale stage-specific mixed pooling
evidence_used: Independent 90%-max gates achieved 9,284 correct, while shared, channelwise, 87.5%, 95%, and asymmetric variants all regressed; this supports preserving stage independence and the 90% operating point while testing more conservative gate optimization.
result: improved the objective and became an available design
reported_values: {"batch_size": 192, "examples_processed": 100000, "optimizer_steps": 522, "parameters": 245152, "training_seconds": 51.639847041806206, "validation_accuracy": 0.9281, "validation_correct": 9281, "validation_cross_entropy": 0.208310994720459, "validation_score": 9281.413800753437}

## Direction

Before choosing the next change, step back from the current line of work. Think very critically about the assumptions you have made so far, and the assumptions shared by the available designs, especially if they have resulted in a lack of progress. Do a thorough analysis of the assumptions, and identify the load-bearing assumptions. Moving forward, make changes that challenge these assumptions, and try to test genuinely different learned computational mechanisms. Think critically about how your next change could make more progress than what you have been making so far, and implement that thinking into your changes. The change should alter how the classifier represents images or computes class predictions. Do not revisit a type of change that already failed unless the recent evidence identifies a specific reason the new version should behave differently. Prefer implementations that cleanly test the alternative, and state the old assumption and the new approach in the final summary. Use prior results to explain why the alternative is plausible and informative.

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
blocks have been applied. All blocks must apply. Together they must describe
one implementation ready for verification. The mechanism name is descriptive,
not chosen from a fixed list. Do not paste whole files, lengthy logs, or routine
progress reports outside the patch.
