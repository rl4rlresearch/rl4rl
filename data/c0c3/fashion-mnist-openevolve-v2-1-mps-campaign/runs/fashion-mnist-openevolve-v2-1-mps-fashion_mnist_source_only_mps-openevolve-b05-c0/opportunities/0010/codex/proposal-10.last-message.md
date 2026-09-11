MECHANISM: Higher update density from smaller minibatches

HYPOTHESIS: Reducing batch size from 128 to 64 will exceed 9,210 correct predictions by providing roughly twice as many optimizer updates within the fixed 100,000-example budget.

INTENDED_EDIT: Change only the training batch size, preserving the successful BatchNorm architecture, loss, cosine schedule, and tail EMA.

EVIDENCE: Moving from the original 392-update regime to the 782-update, batch-128 regime accompanied the large improvement from 8,928 to 9,168 correct; after BatchNorm raised this to 9,210, update density remains the clearest unisolated optimization variable.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE