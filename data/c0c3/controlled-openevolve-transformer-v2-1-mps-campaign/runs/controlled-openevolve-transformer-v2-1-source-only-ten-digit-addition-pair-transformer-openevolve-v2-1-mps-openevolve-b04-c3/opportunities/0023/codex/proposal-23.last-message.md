MECHANISM: Fixed-amplitude sinusoidal positional representation

HYPOTHESIS: Fixing the sinusoidal position amplitude at 0.02 will retain at least 99% accuracy after 75,000 steps while reducing the qualified 1,397-parameter design to 1,396 parameters.

INTENDED_EDIT: Remove the learned positional gain and add the existing fixed-amplitude sinusoidal buffer directly to token embeddings.

EVIDENCE: The learned-scale sinusoidal design achieved 99.69% accuracy at 74,999 steps; the prior fixed-amplitude attempt produced no accuracy evidence because 90,000-step training exceeded the time limit, so testing the same one-parameter reduction at the qualified 75,000-step budget is the closest controlled experiment.

<<<<<<< SEARCH
        # A generic Fourier basis replaces the independently learned vector at
        # every position. Attention learns how to use these positional features,
        # while one gain adapts their magnitude relative to token embeddings.
=======
        # A generic fixed-amplitude Fourier basis replaces the independently
        # learned vector at every position.
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
        self.pos_scale = nn.Parameter(torch.ones(()))
=======
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
=======
        x = self.token_emb(idx) + self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE