MECHANISM: Shared trainable query-offset gauge

HYPOTHESIS: Sharing one learned query-offset scalar across both heads will reduce the model from 1,378 to 1,377 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Replace the two per-head query-bias parameters with one broadcast learned scalar initialized at zero, and shorten training by 2,000 steps for runtime margin.

EVIDENCE: The 1,378-parameter scalar-query model achieved 100% accuracy at 52,000 steps, while fixing the offsets to one timed out; retaining a trainable zero-initialized offset preserves the successful optimization path, and independent query/key weights can absorb head-specific rescaling.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(n_head))
=======
        self.q_bias = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias.view(1, self.n_head, 1, 1)
=======
        q = q + self.q_bias.view(1, 1, 1, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=52000)
=======
    p.add_argument("--train-steps", type=int, default=50000)
>>>>>>> REPLACE