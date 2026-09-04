MECHANISM: Cross-head residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,607-parameter design by quotienting the first output-projection column of the second attention head will yield 1,606 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Parameterize projection columns 0, 1, and 4 in the seven-dimensional zero-sum basis, reconstruct their original ordering in the forward pass, and generalize quotient optimizer moments to matrix parameters.

EVIDENCE: Quotienting head 0’s first two projection columns achieved 99.96% at 1,607 parameters, while quotienting its third column scored 98.71%; testing the analogous first column of head 1 isolates a distinct cross-head allocation of the same exact LayerNorm-erased direction.

<<<<<<< SEARCH
            # Any uniform output-row component of a projection column only
            # adds a per-token common-mode residual shift. Subsequent
            # LayerNorms erase it, so quotient one column by that direction.
            compact_proj_column = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, 0]
            )
            remaining_proj_weight = proj_weight[:, 1:].clone()
        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_first_column = nn.Parameter(compact_proj_column)
        self.proj.weight = nn.Parameter(remaining_proj_weight)
=======
            # Quotient the first two columns of head zero and the analogous
            # first column of head one by their function-null uniform output
            # direction. Keep all other columns in their original order.
            selected_proj_weight = torch.cat(
                (
                    proj_weight[:, :2],
                    proj_weight[:, self.head_dim : self.head_dim + 1],
                ),
                dim=1,
            )
            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ selected_proj_weight
            )
            remaining_proj_weight = torch.cat(
                (
                    proj_weight[:, 2 : self.head_dim],
                    proj_weight[:, self.head_dim + 1 :],
                ),
                dim=1,
            ).clone()
        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_compact_columns = nn.Parameter(compact_proj_columns)
        self.proj.weight = nn.Parameter(remaining_proj_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = self.proj_bias_basis @ self.proj.bias
        proj_first_column = (
            self.proj_bias_basis @ self.proj_first_column
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (proj_first_column, self.proj.weight), dim=1
        )
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        compact_proj_columns = (
            self.proj_bias_basis @ self.proj_compact_columns
        )
        first_head_remaining = self.head_dim - 2
        proj_weight = torch.cat(
            (
                compact_proj_columns[:, :2],
                self.proj.weight[:, :first_head_remaining],
                compact_proj_columns[:, 2:],
                self.proj.weight[:, first_head_remaining:],
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_first_column, blk.attn.proj_bias_basis),
=======
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_compact_columns, blk.attn.proj_bias_basis),
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in gauge_params
    ]
=======
    gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(
                (basis.size(0), *param.shape[1:])
            ),
            "exp_avg_sq": basis.new_zeros(
                (basis.size(0), *param.shape[1:])
            ),
        }
        for param, basis in gauge_params
    ]
>>>>>>> REPLACE