MECHANISM: Larger-minibatch throughput with improved batch-normalization statistics

HYPOTHESIS: Increasing the batch size from 48 to 64 will provide enough training and evaluation throughput to avoid recent timeouts while preserving at least 9,252 correct predictions through less noisy gradients and batch-normalization estimates.

INTENDED_EDIT: Increase only the batch size, retaining the successful architecture, optimizer schedule, augmentation, late averaging, ten-view ensemble, and 1.10 calibration scale.

EVIDENCE: The available design reached 9,252 correct but required 76.9 training seconds, and numerous mathematically minor inference changes subsequently timed out; reducing optimizer steps from 2,084 to approximately 1,563 directly targets this recurring constraint with minimal algorithmic disruption.

<<<<<<< SEARCH
BATCH_SIZE = 48
=======
BATCH_SIZE = 64
>>>>>>> REPLACE