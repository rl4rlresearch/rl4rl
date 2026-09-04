MECHANISM: Tail-free divisor batching

HYPOTHESIS: Using batch size 125 will exceed 9,322 correct predictions by giving every training example equal optimizer-step weight and providing 800 uniform updates without materially increasing runtime.

INTENDED_EDIT: Change the batch size from 128 to 125, which exactly divides the 50,000-image split and eliminates both 80-example tail batches.

EVIDENCE: The top-four design achieved 9,322 correct with batch size 128, while weighting BatchNorm updates for the smaller tail batches left the hard-maximum design at 9,320; this suggests testing the remaining tail effect—disproportionate optimizer updates—by eliminating partial batches entirely.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 125
>>>>>>> REPLACE