MECHANISM: Disjoint adjacent tolerant-head bias sharing

HYPOTHESIS: A 676-parameter transformer will retain at least 99% accuracy because it preserves the qualified three-way longest-bias sharing while independently tying the next two shorter second-head biases, avoiding the destructive four-way merge.

INTENDED_EDIT: Reconstruct the qualified 677-parameter routing layout, then remove one parameter by adding a separate two-way adjacent bias tie in the tolerant second head.

EVIDENCE: The 677-parameter three-way-sharing design achieved 99.97%, while extending that same group to four positions collapsed to 50.28%; this motivates preserving the successful triple and testing a disjoint adjacent pair.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads; the second head's next four longest biases are also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 4)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases, shares
        # its three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_coeff = F.pad(self.relative_bias, (0, 4)).view(
            self.n_head, -1
        )
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE