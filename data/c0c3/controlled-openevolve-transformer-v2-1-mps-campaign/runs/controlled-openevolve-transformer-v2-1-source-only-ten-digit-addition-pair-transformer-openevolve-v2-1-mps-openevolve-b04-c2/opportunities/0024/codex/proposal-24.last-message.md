MECHANISM: Bias-free attention with verified tail positional quotient

HYPOTHESIS: Removing the remaining zero-initialized query bias from the verified 1,621-parameter design will produce a 1,613-parameter model with at least 99% accuracy, because a learned pre-attention LayerNorm bias and query projection can jointly represent a constant query offset without changing the initial forward pass.

INTENDED_EDIT: Apply the verified value-bias removal and final-four positional quotient, then remove the query bias so the QKV projection is entirely bias-free.

EVIDENCE: The 1,621-parameter design removed all value-bias parameters and retained 99.93% accuracy. Unlike the failed positional and final-LayerNorm reductions, eliminating query bias preserves initialization exactly and tests a distinct affine redundancy through `ln1.bias` and the query weight.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the original parameter slot and initialization RNG consumption,
        # but store only the query and value biases. A shared key bias cancels
        # from every row of the attention softmax.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve constructor RNG consumption while omitting all QKV biases.
        # Key and value biases are functionally redundant, while a query
        # offset can be represented by the learned pre-attention LayerNorm
        # bias followed by the query projection.
        self.qkv.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        qkv = F.linear(x, self.qkv.weight, None)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final two positions, which have the shortest causal influence.
        self.compact_pos_count = 4
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
>>>>>>> REPLACE

<<<<<<< SEARCH
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-2:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-2].reshape(-1),
                )
            )
=======
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-2:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-2].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
>>>>>>> REPLACE