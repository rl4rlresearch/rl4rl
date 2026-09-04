MECHANISM: Faster short-horizon Adam second-moment adaptation

HYPOTHESIS: Reducing AdamW’s second-moment decay from 0.99 to 0.98 will exceed 9,324 correct predictions by tracking gradient-scale changes more responsively during the 2,084-step warmup-and-cosine training horizon.

INTENDED_EDIT: Change only AdamW’s second beta from 0.99 to 0.98, preserving the successful architecture, augmentation, loss, schedule, and tail weight averaging.

EVIDENCE: Lowering beta2 from the default 0.999 to 0.99 improved correctness from 9,318 to 9,324 and cross-entropy from 0.20309 to 0.20116; continuing this successful optimization direction is the most informative next test.

<<<<<<< SEARCH
        betas=(0.9, 0.99),
=======
        betas=(0.9, 0.98),
>>>>>>> REPLACE