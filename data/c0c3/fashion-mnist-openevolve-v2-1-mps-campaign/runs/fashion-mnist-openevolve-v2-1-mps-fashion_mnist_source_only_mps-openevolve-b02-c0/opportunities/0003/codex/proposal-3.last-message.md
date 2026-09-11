MECHANISM: Small-batch optimizer-step densification

HYPOTHESIS: Keeping the proven 216,346-parameter architecture and training recipe but doubling optimizer updates from roughly 782 to 1,563 will exceed 9,128 correct predictions by improving optimization under the fixed 100,000-example exposure.

INTENDED_EDIT: Reduce batch size from 128 to 64 while preserving the successful architecture, augmentation, optimizer, schedule, and flip-averaged inference.

EVIDENCE: The successful design combined an increase from 392 to 782 optimizer steps with a gain from 8,928 to 9,128 correct, whereas replacing its spatial head with a deeper residual global-pooling model fell to 8,985; this motivates retaining the proven model and testing increased update density.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE