MECHANISM: Exposure-compensated larger-batch optimization

HYPOTHESIS: Batch size 64 with linearly scaled learning rate and exposure-matched EMA updates will finish within the verification limit while matching or exceeding 9,258 correct predictions.

INTENDED_EDIT: Double the batch size and learning rate, and update the EMA every two steps so its sampling frequency per presented example remains unchanged.

EVIDENCE: The verified design required 3,126 optimizer steps and 62.8 training seconds, while every subsequent verification timed out; halving the step count directly targets that repeated failure while compensating the learning-rate and EMA dynamics for the larger batch.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.parameters(), lr=2.5e-3, weight_decay=1e-4
=======
        model.parameters(), lr=5.0e-3, weight_decay=1e-4
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 5.0e-3 * multiplier
>>>>>>> REPLACE

<<<<<<< SEARCH
            elif (completed_steps - ema_start) % 4 == 0:
=======
            elif (completed_steps - ema_start) % 2 == 0:
>>>>>>> REPLACE