MECHANISM: Per-head relative-bias softmax gauge fixing

HYPOTHESIS: Fixing each head’s distance-zero relative bias to zero will reduce parameters from 1,588 to 1,586 while retaining at least 99% accuracy, because subtracting that bias from every causal attention logit leaves the softmax unchanged.

INTENDED_EDIT: Store only nonzero-distance relative biases and prepend a fixed zero reference bias during attention.

EVIDENCE: The rank-4 positional/relative-attention design achieved 99.97% accuracy at 1,588 parameters; unlike riskier capacity reductions, this removes one exact softmax-invariant degree of freedom per head.

<<<<<<< SEARCH
        # Each head learns which causal distances are useful, allowing operand
        # alignment to be represented directly in attention-score space.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge;
        # only relative differences between its distance biases are learned.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att + self.relative_bias[:, distance].unsqueeze(0)
=======
        relative_bias = torch.cat(
            (self.relative_bias.new_zeros(self.n_head, 1), self.relative_bias),
            dim=1,
        )
        att = att + relative_bias[:, distance].unsqueeze(0)
>>>>>>> REPLACE