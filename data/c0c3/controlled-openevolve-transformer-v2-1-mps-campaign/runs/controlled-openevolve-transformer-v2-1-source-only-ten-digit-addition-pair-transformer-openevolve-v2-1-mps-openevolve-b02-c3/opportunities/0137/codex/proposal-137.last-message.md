MECHANISM: Second-head longest-boundary quartet sharing

HYPOTHESIS: A 663-parameter transformer will retain at least 99% accuracy because extending the second head’s proven longest-distance triplet to a quartet compresses a farther boundary group while preserving the adjacent pair whose extension previously failed.

INTENDED_EDIT: Reconstruct the qualified 664-parameter first-head quintet design, then share one coefficient across four rather than three of the second head’s longest remaining relative biases.

EVIDENCE: The first-head quintet achieved 99.57% at 664 parameters, and second-head three-way longest-distance sharing achieved 99.97% at 677 parameters; by contrast, extending the second head’s adjacent pair to a triplet at 663 parameters fell to 67.67%, motivating compression of the distinct longest-distance group.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent triplet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 14)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quintet immediately preceding them. The second
        # head fixes its next six longest biases, shares its four longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 17)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 5
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 5 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(3),
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 7
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 7 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(5),
                self.relative_bias.new_zeros(3),
                second_head_bias[:-2],
                second_head_bias[-2:-1].expand(2),
                second_head_bias[-1:].expand(4),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE