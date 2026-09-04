MECHANISM: Reduced short-horizon head regularization

HYPOTHESIS: Reducing classifier dropout from 0.15 to 0.05 will exceed 9,280 correct predictions by improving feature utilization during the fixed two-pass exposure without the instability observed from a 1.2× learning-rate increase.

INTENDED_EDIT: Lower only the classifier-head dropout probability, preserving architecture size, runtime, augmentation, optimizer, schedule, and validation ensemble.

EVIDENCE: More optimizer updates improved validation_correct from 9,238 to 9,280, indicating that the short training horizon benefits from more effective optimization, while increasing learning rates reduced it to 9,274; weaker stochastic masking is a low-cost alternative for accelerating head convergence.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.05),
>>>>>>> REPLACE