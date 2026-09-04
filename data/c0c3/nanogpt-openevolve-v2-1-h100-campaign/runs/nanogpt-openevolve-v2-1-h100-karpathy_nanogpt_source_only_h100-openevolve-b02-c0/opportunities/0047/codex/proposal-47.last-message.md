MECHANISM: Fine-grained local-attention window refinement

HYPOTHESIS: A 420-token short window will reduce val_bpb below 0.983635 by retaining slightly more throughput than 424 tokens while preserving more context than the worse 416-token design.

INTENDED_EDIT: Reduce all six short-attention windows from 424 to 420 tokens while preserving both full-context layers and every other setting.

EVIDENCE: The best result moved from 0.983765 at 416 tokens to 0.983635 at 424 tokens, while 432 tokens regressed to 0.983917; 420 tokens bisects the remaining lower-side bracket around the current optimum.

<<<<<<< SEARCH
        short_window = 53 * long_window // 256
=======
        short_window = 105 * long_window // 512
>>>>>>> REPLACE