MECHANISM: First-head boundary-bias triplet sharing

HYPOTHESIS: A 666-parameter transformer will retain at least 99% accuracy because extending the successful first-head boundary pair sharing to three adjacent biases preserves an adaptive long-distance routing signal.

INTENDED_EDIT: Remove one relative-bias parameter by sharing the three learned first-head biases immediately preceding its fixed three-bias suffix.

EVIDENCE: The current pair-sharing design achieved 99.44% at 667 parameters, whereas fixing the same boundary bias at zero reached only 98.62%; the 99.97% second-head design also demonstrates that three-way adjacent bias sharing can preserve routing.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent pair immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 13)
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 4
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 4 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(3),
=======
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
>>>>>>> REPLACE