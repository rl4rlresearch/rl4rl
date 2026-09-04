MECHANISM: Nonzero cosine learning-rate floor

HYPOTHESIS: Keeping 10% of the peak learning rate at the end of training will exceed 9,250 correct predictions by making the final optimizer updates productive while tail EMA controls their noise.

INTENDED_EDIT: Change the cosine schedule from decay-to-zero to decay-to-10%-of-peak, leaving the proven batch-32 model, loss, optimizer, and EMA unchanged.

EVIDENCE: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, indicating that additional effective updates help; smaller batches timed out, so increasing the usefulness of existing late updates is the next runtime-neutral test.

<<<<<<< SEARCH
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.1 + 0.9 * cosine
>>>>>>> REPLACE