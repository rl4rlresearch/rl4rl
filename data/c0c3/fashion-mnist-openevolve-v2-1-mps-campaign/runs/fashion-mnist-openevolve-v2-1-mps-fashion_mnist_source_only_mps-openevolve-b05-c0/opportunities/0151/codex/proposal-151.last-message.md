MECHANISM: Modest cosine learning-rate rescaling

HYPOTHESIS: Increasing the peak learning rate by 4% to 2.6e-3 will exceed 9,260 correct predictions by improving optimization progress without the aggressive schedule distortion of a peak hold.

INTENDED_EDIT: Raise both AdamW’s initial learning rate and the cosine schedule’s base learning rate from 2.5e-3 to 2.6e-3, preserving all other settings.

EVIDENCE: The 5% warmup reduced early learning rates and lowered validation_correct from 9,260 to 9,250 despite improving cross-entropy, suggesting that slightly more optimization progress may favor the primary correctness objective.

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
    )
=======
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.6e-3, weight_decay=1e-4
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.6e-3 * multiplier
>>>>>>> REPLACE