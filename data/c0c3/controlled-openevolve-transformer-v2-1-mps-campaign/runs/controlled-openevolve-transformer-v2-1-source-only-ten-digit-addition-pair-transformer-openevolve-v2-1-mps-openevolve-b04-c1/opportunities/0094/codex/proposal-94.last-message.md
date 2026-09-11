MECHANISM: Query-bias rotational gauge fixing

HYPOTHESIS: Constraining the sole biased two-dimensional attention head to one query-bias coordinate will reduce the model from 872 to 871 parameters while retaining at least 99% accuracy, because a joint orthogonal rotation of that head’s query and key channels can align any query-bias vector with one axis without changing attention logits.

INTENDED_EDIT: Replace the two learned query-bias coordinates with one scalar coordinate; the existing padding keeps all other query-bias coordinates fixed at zero while preserving initialization and random-stream order.

EVIDENCE: The two-dimensional-query/key design reached 99.92% accuracy, and the current exact-quotient design reached 99.98% at 872 parameters; this removes only the remaining rotationally redundant direction rather than reducing attention capacity.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
=======
        # Joint Q/K rotations preserve attention logits, so the biased head's
        # two-coordinate query offset can be aligned with one fixed axis.
        self.q_bias = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE