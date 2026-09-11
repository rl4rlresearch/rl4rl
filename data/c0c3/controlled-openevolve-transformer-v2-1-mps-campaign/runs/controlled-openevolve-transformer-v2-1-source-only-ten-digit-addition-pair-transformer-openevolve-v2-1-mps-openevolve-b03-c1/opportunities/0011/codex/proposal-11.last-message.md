MECHANISM: Remove value-bias-representable attention output bias

HYPOTHESIS: Removing the eight zero-initialized attention output-projection bias parameters will reduce the model from 1,612 to 1,604 parameters while retaining at least 99% accuracy, because the retained value bias can supply a position-independent vector through the output projection.

INTENDED_EDIT: Construct the attention output projection normally to preserve initialization ordering, then remove only its learned bias.

EVIDENCE: The 1,612-parameter model achieved 99.96% accuracy. Although removing value bias failed, that result motivates preserving the optimization-critical value bias and testing the converse redundant parameter: the downstream projection bias.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)
        # Preserve construction order while removing the bias representable by
        # the retained value bias through this projection.
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE