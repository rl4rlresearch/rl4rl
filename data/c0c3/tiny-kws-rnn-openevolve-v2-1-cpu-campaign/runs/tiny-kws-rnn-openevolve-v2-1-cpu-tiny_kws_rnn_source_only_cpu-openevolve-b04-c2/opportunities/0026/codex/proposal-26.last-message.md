MECHANISM: Higher-update optimization at the 22-step frontier

HYPOTHESIS: Training the 22-step frames 4–25 model with batch size 32 will recover its 0.21-point accuracy shortfall and reach at least 85% validation accuracy while retaining approximately 682.1 million total inference MACs.

INTENDED_EDIT: Remove frame 26 from the qualified schedule and halve the training batch size from 64 to 32, doubling the approximate number of optimizer updates without changing inference architecture or cost.

EVIDENCE: At 27 steps, reducing batch size from 128 to 64 raised accuracy from 84.66% to 87.24%; the batch-64 22-step model already reached 84.79%, so applying the same higher-update strategy targets a much smaller deficit.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(4, available_frames - 5))
=======
        return list(range(4, available_frames - 6))
>>>>>>> REPLACE