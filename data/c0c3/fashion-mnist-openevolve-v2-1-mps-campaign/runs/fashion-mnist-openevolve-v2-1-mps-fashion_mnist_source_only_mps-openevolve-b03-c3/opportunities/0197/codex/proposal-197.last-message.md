MECHANISM: Cosine-annealed classifier dropout

HYPOTHESIS: Decaying dropout from 0.10 to zero will exceed 9,284 correct predictions by retaining early regularization while making low-learning-rate fine-tuning match deterministic evaluation.

INTENDED_EDIT: Preserve the validated architecture, loss, augmentation, optimizer, TTA, and calibration while annealing the existing classifier dropout probability over training.

EVIDENCE: Architectural changes, smaller batches, EMA, and flip-consistency regularization all reduced validation_correct; this motivates a minimal schedule change that removes only late training-time noise from the otherwise validated computation.

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_probability = 0.05 * (1.0 + math.cos(math.pi * progress))
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_probability
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
>>>>>>> REPLACE