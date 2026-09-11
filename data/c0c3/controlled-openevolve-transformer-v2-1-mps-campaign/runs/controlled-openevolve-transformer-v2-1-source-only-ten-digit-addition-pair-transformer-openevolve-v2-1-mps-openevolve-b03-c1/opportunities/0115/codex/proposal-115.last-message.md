MECHANISM: Residual query-bias rotational gauge fixing

HYPOTHESIS: Fixing one first-head query-bias coordinate will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy, because a rotation of the untouched third/fourth query-key feature pair can set that bias coordinate to zero without changing attention scores.

INTENDED_EDIT: Remove the third first-head query-bias scalar after initialization and reconstruct it as a fixed zero during attention.

EVIDENCE: Two query-key orthogonal gauge removals preserved attention scores and achieved 99.93% and 99.90% accuracy; this removes another rotational gauge coordinate while avoiding the additional learned-weight packing changes associated with later timeout results.

<<<<<<< SEARCH
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_flat[q_tail_coordinates],
                        k_flat[k_tail_coordinates],
                        v_flat[tail_coordinates],
                    )
                ).clone()
            )
=======
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_flat[q_tail_coordinates],
                        k_flat[k_tail_coordinates],
                        v_flat[tail_coordinates],
                    )
                ).clone()
            )

            # The untouched third/fourth feature plane of the first head has
            # a residual joint query-key rotation. Use it to fix the third
            # query-bias coordinate, which is already zero at initialization.
            self.qkv.bias = nn.Parameter(
                torch.cat(
                    (self.qkv.bias[:2], self.qkv.bias[3:])
                ).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
=======
        q_bias = torch.cat(
            (
                self.qkv.bias[:2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2:],
            )
        )
        q = F.linear(x, q_weight, q_bias)
        k = F.linear(x, k_weight)
>>>>>>> REPLACE