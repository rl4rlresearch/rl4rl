MECHANISM: Balanced per-head query-bias pruning

HYPOTHESIS: A 1,376-parameter transformer will retain at least 99% accuracy because the current 1,377-parameter design achieved 100%, while fixing the corresponding initially zero coordinate in the other attention head leaves three learned query-bias coordinates per head and preserves all query-key projections and relative-lag routing.

INTENDED_EDIT: Replace the seven-coordinate query bias with six learned coordinates, reconstructing the final coordinate of each query head as zero during every forward pass.

EVIDENCE: The current design reached 100% accuracy after pruning one initially zero query-bias coordinate, substantially outperforming prior 1,377-parameter projection, scale, and MLP reductions; distributing the same pruning mechanism symmetrically across both heads is the closest informative extension.

<<<<<<< SEARCH
        # The final coordinate starts at zero and remains fixed; the other
        # seven query-bias coordinates are learned normally.
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        # The final coordinate of each head starts at zero and remains fixed;
        # the other three query-bias coordinates per head are learned.
        self.q_bias = nn.Parameter(torch.zeros(d_model - n_head))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (self.q_bias, self.q_bias.new_zeros(1))
        )
        q = self.q_proj(x) + q_bias
=======
        learned_q_bias = self.q_bias.view(
            self.n_head, self.head_dim - 1
        )
        q_bias = torch.cat(
            (
                learned_q_bias,
                learned_q_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        ).reshape(-1)
        q = self.q_proj(x) + q_bias
>>>>>>> REPLACE