MECHANISM: Energy-matched fixed head-slot residual routing

HYPOTHESIS: Replacing dense attention output mixing with distinct residual slots and one learned gain per head will reduce the model from 1,107 to 1,046 parameters while retaining at least 99% accuracy, because preserving route identity—not dense post-attention mixing—is load-bearing.

INTENDED_EDIT: Compress the initialized observable output maps into two energy-matched learned gains, remove the dense projection and its bias before training, and concatenate the independently routed head contexts directly into disjoint residual channels.

EVIDENCE: Summing the routed contexts before a shared projection collapsed accuracy to 0.01%, showing that head identity is essential. Conversely, the current shared-value model reached 99.91%, indicating that both routes can use the same semantic value representation. Fixed disjoint slots preserve that identity while testing whether the 63-parameter dense mixer is unnecessary.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias and independent query/key projections while
        # sharing one zero-mean value readout across the routing heads.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(self.proj.bias[:-1].clone())
=======
        # Construct the original affines to preserve the baseline RNG stream.
        # After global initialization, compress the dense output projection
        # into learned per-head gains and keep the routed contexts in distinct
        # residual-channel slots.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(self.proj.bias[:-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )
=======
    def compress_proj_to_head_gains(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            head_maps = centered.view(
                d_model, self.n_head, self.head_dim
            )
            gain = torch.linalg.vector_norm(
                head_maps, dim=(0, 2)
            ) / math.sqrt(self.head_dim)
            self.proj_gain = nn.Parameter(gain.clone())
            self.proj = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_weight = self.proj_basis @ self.proj.weight
        proj_bias = self.proj_basis @ self.proj.bias
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y * self.proj_gain.view(1, self.n_head, 1, 1)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
=======
            block.attn.gauge_fix_qkv()
            block.attn.compress_proj_to_head_gains()
            block.mlp.gauge_fix_fc1()
>>>>>>> REPLACE