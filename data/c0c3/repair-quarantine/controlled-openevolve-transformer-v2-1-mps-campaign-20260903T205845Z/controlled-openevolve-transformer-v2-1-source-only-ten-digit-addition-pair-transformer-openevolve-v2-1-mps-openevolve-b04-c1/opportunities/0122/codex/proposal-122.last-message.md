MECHANISM: Bias-free narrow Q/K attention

HYPOTHESIS: Fixing the single query-offset scalar at zero will reduce the verified model from 847 to 846 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the zero-initialized learned query bias with a nonpersistent zero buffer, preserving the forward computation at initialization and the random initialization stream.

EVIDENCE: The 847-parameter model achieved 99.94% accuracy, while reductions that constrained independent GLU projections failed; this instead tests one isolated attention scalar without altering the verified GLU capacity.

<<<<<<< SEARCH
        # Joint Q/K rotations preserve attention logits, so the biased head's
        # two-coordinate query offset can be aligned with one fixed axis.
        self.q_bias = nn.Parameter(torch.zeros(1))
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
=======
        # Use bias-free Q/K attention while preserving the established
        # tensor shape and constructor-time random initialization stream.
        self.register_buffer(
            "q_bias", torch.zeros(1), persistent=False
        )
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
>>>>>>> REPLACE