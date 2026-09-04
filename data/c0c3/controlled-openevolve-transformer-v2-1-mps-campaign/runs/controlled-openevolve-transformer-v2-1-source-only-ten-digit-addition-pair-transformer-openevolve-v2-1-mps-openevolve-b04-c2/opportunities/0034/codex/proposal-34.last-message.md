MECHANISM: Attention-output weight common-mode gauge quotient

HYPOTHESIS: Compacting all eight columns of the attention output-projection weight into the seven-dimensional zero-sum residual subspace will reduce the verified 1,598-parameter model to 1,590 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store the attention output-projection weight in compact output coordinates, reconstruct it in the forward pass, and preserve virtual eight-coordinate AdamW dynamics.

EVIDENCE: Compacting all 12 terminal-MLP output-weight columns with the same downstream-LayerNorm symmetry achieved 99.83% accuracy at 1,598 parameters; an attention-output common-mode shift likewise leaves the intervening MLP unchanged through `ln2` and is removed by the final LayerNorm.

<<<<<<< SEARCH
    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-3:] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-3].clone())
        self.k_compact = nn.ParameterList(
            [nn.Parameter(row.clone()) for row in compact_key]
        )
        self.v_weight = nn.Parameter(v_weight.clone())
        del self.qkv
=======
    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-3:] @ self.key_basis
            compact_proj = (
                self.key_basis.transpose(0, 1) @ self.proj.weight.detach()
            )
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-3].clone())
        self.k_compact = nn.ParameterList(
            [nn.Parameter(row.clone()) for row in compact_key]
        )
        self.v_weight = nn.Parameter(v_weight.clone())
        self.proj.weight = nn.Parameter(compact_proj.clone())
        del self.qkv
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_weight = self.key_basis @ self.proj.weight
        y = F.linear(y, proj_weight, self.proj.bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.mlp.fc2.weight, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (compact, blk.attn.key_basis)
        for blk in model.blocks
        for compact in blk.attn.k_compact
    ]
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.mlp.fc2.weight, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.attn.proj.weight, blk.attn.key_basis) for blk in model.blocks
    ] + [
        (compact, blk.attn.key_basis)
        for blk in model.blocks
        for compact in blk.attn.k_compact
    ]
>>>>>>> REPLACE