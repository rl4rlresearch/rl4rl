MECHANISM: Initialization-scale query-bias anchoring

HYPOTHESIS: Fixing the shared query-bias amplitude at 0.02 will produce a 1,413-parameter model with at least 99% accuracy, because any nonzero amplitude retains the query/key rescaling-equivalent function class while matching the projection initialization scale and avoiding the disruptive unit-amplitude initialization.

INTENDED_EDIT: Replace the learned shared query-bias scalar with a fixed, nonpersistent 0.02 buffer while preserving both attention heads, full query/key widths, and all other verified components.

EVIDENCE: The learned shared scalar reached 99.04% in the current 1,414-parameter design, while fixing it at one previously collapsed to 39.69%; testing a small nonzero anchor isolates whether that failure came from initialization scale rather than loss of model capacity.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(1))
=======
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
>>>>>>> REPLACE