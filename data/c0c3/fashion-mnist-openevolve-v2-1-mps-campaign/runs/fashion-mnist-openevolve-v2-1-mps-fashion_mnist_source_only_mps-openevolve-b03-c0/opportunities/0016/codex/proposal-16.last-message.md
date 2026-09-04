MECHANISM: Reduced stochastic head regularization

HYPOTHESIS: Lowering classifier dropout from 0.10 to 0.05 will exceed 9,316 correct predictions by improving short-horizon feature utilization while retaining the beneficial 0.04 label smoothing.

INTENDED_EDIT: Reduce the dense classifier’s dropout probability to 0.05; preserve the architecture, optimizer, schedule, and label smoothing.

EVIDENCE: Label smoothing improved correctness at 0.04 but regressed at 0.06, indicating excessive regularization is harmful; tuning the still-untested dropout strength isolates another source of regularization without disturbing the strongest configuration.

<<<<<<< SEARCH
            nn.Dropout(p=0.1),
=======
            nn.Dropout(p=0.05),
>>>>>>> REPLACE