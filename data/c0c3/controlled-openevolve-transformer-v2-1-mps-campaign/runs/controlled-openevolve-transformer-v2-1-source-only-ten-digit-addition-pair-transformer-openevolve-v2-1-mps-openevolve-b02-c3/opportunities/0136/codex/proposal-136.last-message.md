MECHANISM: Orthogonal first- and second-head boundary-bias sharing

HYPOTHESIS: A 663-parameter transformer will retain at least 99% accuracy because it combines the qualified first-head quintet with a three-way extension of the second head’s adjacent pair, preserving separate learned routing signals instead of using the failed first-head sextet.

INTENDED_EDIT: Extend the first-head shared boundary group from four to five biases and the second-head adjacent shared pair from two to three biases, removing two learned scalars.

EVIDENCE: The first-head quintet achieved 99.57% at 664 parameters, while its sextet fell to 90.99%; independently, three-way second-head sharing achieved 99.97%, motivating the next reduction on the second head rather than further compressing the first.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quartet immediately preceding them. The second
        # head fixes its next six longest biases, shares its three longest
        # remaining positions, and separately shares the adjacent pair
        # immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 15)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head fixes its next three longest distances and
        # shares the adjacent quintet immediately preceding them. The second
        # head fixes its next six longest biases and separately shares two
        # adjacent triplets immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 17)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 6
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 6 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias[:-1],
                first_head_bias[-1:].expand(4),
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
                second_head_bias[-2:-1].expand(3),
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
            )
        ).view(self.n_head, -1)
>>>>>>> REPLACE