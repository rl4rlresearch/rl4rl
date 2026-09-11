MECHANISM: Reduced classifier dropout on fast maximum attention

HYPOTHESIS: Reducing classifier dropout from 0.10 to 0.05 will exceed 9,322 correct predictions by improving convergence during the fixed two-pass exposure budget while retaining mild regularization.

INTENDED_EDIT: Halve classifier dropout while preserving the reliable hard-maximum attention architecture, paired-view cross-entropy, optimizer, and flip ensemble.

EVIDENCE: Hard-maximum attention reached 9,320 correct in 75.3 seconds; the prior reduced-dropout trial used slower top-four saliency and timed out, so applying the parameter-neutral change to the faster verified backbone isolates its accuracy effect with better timing margin.

<<<<<<< SEARCH
            nn.Dropout(p=0.10),
=======
            nn.Dropout(p=0.05),
>>>>>>> REPLACE