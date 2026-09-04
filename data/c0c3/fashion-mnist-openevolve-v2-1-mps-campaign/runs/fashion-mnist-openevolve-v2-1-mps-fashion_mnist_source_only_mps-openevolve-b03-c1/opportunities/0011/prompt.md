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

The current editable design is provided. No reference design is available.

CURRENT DESIGN
verified_results: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 38.54385858308524, "validation_accuracy": 0.9122, "validation_correct": 9122, "validation_cross_entropy": 0.2498493896484375, "validation_score": 9122.40004820112}
prior_hypothesis: Exposing raw intensity, horizontal and vertical Sobel responses, and Laplacian detail will exceed 9,091 correct predictions because the fixed exposure budget no longer requires the shallow stem to learn these shape-sensitive primitives from pixels alone.

## Recent verification evidence

RECENT RESULT
hypothesis: A near-budget residual CNN trained with twice as many optimizer updates, mild translation/flip augmentation, and warmup-cosine scheduling will exceed the current 8,928 correct predictions.
change: Replace the 105,866-parameter baseline with a roughly 247,000-parameter residual model, use batch size 128, augment training images, reduce label smoothing, and adopt warmup-cosine AdamW.
mechanism: Residual multiscale CNN with exposure-efficient augmentation
evidence_used: The starting design reaches 89.28% accuracy after only 392 optimizer steps, motivating greater model capacity and more updates within the unchanged 100,000-example exposure budget.
result: the implementation could not be verified

RECENT RESULT
hypothesis: A 244,386-parameter batch-normalized CNN trained for roughly twice as many optimizer steps will exceed 8,928 correct predictions, while mild augmentation and flip-averaged evaluation reduce overfitting.
change: Use batch size 128, add two convolutional stages within the parameter ceiling, apply mild translation and horizontal-flip augmentation, use low-smoothing AdamW with warmup-cosine decay, and average original/flipped logits during evaluation.
mechanism: Exposure-efficient deep CNN with evaluation-time flip ensembling
evidence_used: The baseline achieves 8,928 correct with only 392 optimizer steps and 105,866 parameters; this motivates safely using the remaining parameter budget and doubling update frequency. The prior near-budget residual attempt produced no verifiable metric, so this patch uses a simpler sequential architecture.
result: improved the objective and became an available design
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 244386, "training_seconds": 24.701929665869102, "validation_accuracy": 0.905, "validation_correct": 9050, "validation_cross_entropy": 0.2664461006164551, "validation_score": 9050.394805590035}

RECENT RESULT
hypothesis: A second 72-channel convolution at 7×7, funded by shrinking the dense layer, will exceed 9,050 correct predictions under the unchanged training procedure.
change: Add spatial feature refinement before the final pooling stage and reduce the classifier width from 160 to 112, yielding 249,186 learned parameters.
mechanism: Dense-to-convolutional parameter reallocation
evidence_used: The simple sequential 244,386-parameter CNN reached 9,050 correct while the more complicated residual attempt was unverifiable; this preserves the verified structure while reallocating capacity from the dense head to image-specific spatial processing.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 128, "examples_processed": 100000, "optimizer_steps": 782, "parameters": 249186, "training_seconds": 27.796126666944474, "validation_accuracy": 0.903, "validation_correct": 9030, "validation_cross_entropy": 0.26525047073364255, "validation_score": 9030.39517867139}

RECENT RESULT
hypothesis: Halving the batch size while linearly scaling the learning rate will exceed 9,050 correct predictions by doubling stochastic optimizer updates without changing the verified architecture or augmentation.
change: Change batch size from 128 to 64 and scale both the initial and scheduled peak learning rates from 2.5e-4/2.5e-3 to 1.25e-4/1.25e-3.
mechanism: Smaller-batch exposure-efficient optimization
evidence_used: The verified 244,386-parameter design reached 9,050 correct with 782 steps, improving on the earlier 392-step design; this isolates whether still more updates improve fixed-exposure learning.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 57.5237520840019, "validation_accuracy": 0.905, "validation_correct": 9050, "validation_cross_entropy": 0.2646499664306641, "validation_score": 9050.395366317378}

RECENT RESULT
hypothesis: Averaging predictions across the original image, four one-pixel translations, and their horizontal flips will exceed 9,050 correct predictions by reducing spatial prediction variance without changing training.
change: Expand evaluation-time augmentation from two views to ten views matched to the translation and flip transformations used during training.
mechanism: Matched translation-and-flip test-time ensembling
evidence_used: Halving the batch size retained exactly 9,050 correct predictions while only improving cross-entropy, suggesting further optimizer updates have saturated; the trained model already sees translations and flips, making matched multi-view inference an orthogonal next test.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 45.195842042099684, "validation_accuracy": 0.9073, "validation_correct": 9073, "validation_cross_entropy": 0.25953308563232425, "validation_score": 9073.396972501718}

RECENT RESULT
hypothesis: Sampling an independent ±2-pixel translation for every training image will exceed 9,073 correct predictions by reducing correlated augmentation noise within each batch.
change: Replace the single batch-wide crop offset with vectorized per-image crop selection while preserving the augmentation range, batch size, architecture, and optimizer.
mechanism: Per-example translation diversification
evidence_used: Ten-view translation-and-flip inference improved the current design from 9,050 to 9,073 correct, showing translation variance remains consequential; training currently gives all 64 images the same translation, so independent offsets directly target that variance.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 44.937164624920115, "validation_accuracy": 0.9082, "validation_correct": 9082, "validation_cross_entropy": 0.25911584167480467, "validation_score": 9082.397104049882}

RECENT RESULT
hypothesis: Averaging all 50 views from the training augmentation distribution will exceed 9,082 correct predictions by further reducing translation-sensitive errors.
change: Replace the ten-view cross-shaped ensemble with every combination of ±2-pixel horizontal and vertical translation, each with its horizontal flip.
mechanism: Full-support translation-and-flip test-time ensembling
evidence_used: Ten-view translation-and-flip inference improved accuracy from 9,050 to 9,073, and per-image sampling across the full ±2 training range then improved it to 9,082; evaluating the complete matched transformation distribution directly extends both gains.
result: was valid but was not a strict improvement
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 53.59044679207727, "validation_accuracy": 0.9074, "validation_correct": 9074, "validation_cross_entropy": 0.26268745346069333, "validation_score": 9074.395980809526}

RECENT RESULT
hypothesis: Biasing training crops toward zero- and one-pixel translations while retaining occasional two-pixel shifts will exceed 9,082 correct predictions.
change: Replace uniform ±2-pixel offsets with independent triangular offsets formed by summing two uniform three-way samples.
mechanism: Center-weighted per-image translation augmentation
evidence_used: Ten-view inference using central and one-pixel translations reached 9,082 correct, whereas uniformly averaging the full ±2 support fell to 9,074, indicating that extreme transformations should receive less weight.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 59.7259305419866, "validation_accuracy": 0.9088, "validation_correct": 9088, "validation_cross_entropy": 0.25145248794555664, "validation_score": 9088.39953574332}

RECENT RESULT
hypothesis: Weighting the centered evaluation view 3:2 relative to each one-pixel cardinal translation will exceed 9,088 correct predictions by matching the successful triangular training-crop distribution.
change: Change the ten-view test-time ensemble from equal averaging to triangular-distribution weights while retaining the same views and horizontal flips.
mechanism: Center-weighted matched test-time ensembling
evidence_used: Center-weighted training translations improved accuracy from 9,082 to 9,088, while uniformly averaging the full translation support fell to 9,074; this motivates testing center bias at inference without changing training.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 244386, "training_seconds": 61.90971816610545, "validation_accuracy": 0.9091, "validation_correct": 9091, "validation_cross_entropy": 0.2513663063049316, "validation_score": 9091.39956325936}

RECENT RESULT
hypothesis: Exposing raw intensity, horizontal and vertical Sobel responses, and Laplacian detail will exceed 9,091 correct predictions because the fixed exposure budget no longer requires the shallow stem to learn these shape-sensitive primitives from pixels alone.
change: Challenge the raw-pixel-only representation assumption by adding three parameter-free differential channels before the existing CNN and expanding its first convolution from one to four inputs, for approximately 245,034 learned parameters.
mechanism: Fixed differential-basis image representation
evidence_used: Doubling optimizer updates left accuracy tied at 9,050 while increasingly refined spatial ensembling produced only incremental gains to 9,091, suggesting optimization and translation variance are no longer the main bottlenecks; a fixed edge/detail basis cleanly tests whether more sample-efficient image representation can provide the next gain.
result: improved the objective and became an available design
reported_values: {"batch_size": 64, "examples_processed": 100000, "optimizer_steps": 1564, "parameters": 245034, "training_seconds": 38.54385858308524, "validation_accuracy": 0.9122, "validation_correct": 9122, "validation_cross_entropy": 0.2498493896484375, "validation_score": 9122.40004820112}



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
