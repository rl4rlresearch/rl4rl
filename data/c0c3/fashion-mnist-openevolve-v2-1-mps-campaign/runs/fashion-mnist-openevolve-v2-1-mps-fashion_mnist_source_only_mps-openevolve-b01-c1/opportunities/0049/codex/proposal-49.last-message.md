MECHANISM: Stronger hierarchical flip-pair supervision

HYPOTHESIS: Increasing flip-pair loss allocation from 25% to 50% will exceed 9,245 correct predictions by further aligning optimization with the flip-averaged validation ensemble while retaining individual-view supervision.

INTENDED_EDIT: Rebalance the intermediate view objective equally between individual-view and flip-pair cross-entropy; preserve the architecture, augmentation, six-view ensemble loss, cosine curriculum, and optimizer.

EVIDENCE: Introducing 25% flip-pair supervision improved the best result from 9,239 to 9,245 correct, while architecture, smoothing, augmentation, and EMA changes failed to improve; increasing that successful signal is the most direct next test.

<<<<<<< SEARCH
    view_loss = 0.75 * individual_loss + 0.25 * pair_loss
=======
    view_loss = 0.5 * individual_loss + 0.5 * pair_loss
>>>>>>> REPLACE