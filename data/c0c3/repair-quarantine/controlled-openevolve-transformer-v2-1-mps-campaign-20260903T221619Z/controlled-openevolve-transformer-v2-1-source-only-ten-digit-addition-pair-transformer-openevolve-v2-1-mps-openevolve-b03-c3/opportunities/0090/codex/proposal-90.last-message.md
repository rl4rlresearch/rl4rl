MECHANISM: Content-independent learned relative-lag attention

HYPOTHESIS: A 1,270-parameter transformer will retain at least 99% accuracy because learned per-head lag distributions can route the fixed-offset operands without content-dependent query/key scores, while learned values, projections, and the MLP still perform token-dependent computation.

INTENDED_EDIT: Replace query/key attention with two learned causal relative-lag routing heads over a shared learned value stream, and absorb all first-LayerNorm scales into that sole value projection.

EVIDENCE: The 1,388-parameter relative-lag design reached 99.22% despite already sharing keys and values; this suggests its load-bearing head distinction is learned lag routing. The patch directly tests the alternative to the old assumption that a separate 104-parameter content-addressing path is also necessary.

<<<<<<< SEARCH
        # Keep a distinct learned query for each head, but share one learned
        # key/value representation across the query heads.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        # Each head learns a causal distribution over relative lags and
        # applies it to one shared learned value representation. This makes
        # positional routing, rather than content matching, the attention
        # mechanism while preserving token-dependent attended values.
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
        # Singleton head dimensions broadcast the shared learned key/value
        # stream across the independently parameterized query heads.
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
=======
        bsz, seqlen, d_model = x.shape
        v = self.v_proj(x).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        lags = (
            positions.unsqueeze(1) - positions.unsqueeze(0)
        ).clamp_min(0)
        full_relative_bias = torch.stack(
            [
                torch.cat((stored, stored.new_zeros(1)))
                for stored in self.relative_bias
            ]
        )
        if torch.is_grad_enabled():
            full_relative_bias.retain_grad()
            self.full_relative_bias = full_relative_bias
        att = att + full_relative_bias[:, lags].unsqueeze(0)
=======
        positions = torch.arange(seqlen, device=x.device)
        lags = (
            positions.unsqueeze(1) - positions.unsqueeze(0)
        ).clamp_min(0)
        full_relative_bias = torch.stack(
            [
                torch.cat((stored, stored.new_zeros(1)))
                for stored in self.relative_bias
            ]
        )
        if torch.is_grad_enabled():
            full_relative_bias.retain_grad()
            self.full_relative_bias = full_relative_bias
        att = full_relative_bias[:, lags].unsqueeze(0).expand(
            bsz, -1, -1, -1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FullyAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The final two ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
    attention_scales = [
        torch.ones(
            2, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
=======
    # With lag-only routing, v is the sole consumer of ln1, so all eight
    # scales can live in optimizer-coordinate state while v stores their
    # products with its input columns.
    attention_scales = [
        torch.ones(
            8, device=device, dtype=blk.attn.v_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(blk.attn.v_proj.weight[:, -8:])
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -2:],
                    blk.attn.k_proj.weight.grad[:, -2:],
                    blk.attn.v_proj.weight.grad[:, -2:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            ).detach()
=======
            effective_grad = (
                blk.attn.v_proj.weight.grad[:, -8:].detach().clone()
            )
            effective_weight = (
                blk.attn.v_proj.weight[:, -8:].detach()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # These effective columns are updated below through their ambient
        # weight/scale factorizations rather than by the ordinary optimizer.
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        # These effective columns are updated below through their ambient
        # weight/scale factorizations rather than by the ordinary optimizer.
        for blk in model.blocks:
            blk.attn.v_proj.weight.grad[:, -8:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                effective_weight = (
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
                q_end = blk.attn.q_proj.out_features
                k_end = q_end + blk.attn.k_proj.out_features
                blk.attn.q_proj.weight[:, -2:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -2:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -2:].copy_(
                    effective_weight[k_end:]
                )
=======
                effective_weight = (
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
                blk.attn.v_proj.weight[:, -8:].copy_(
                    effective_weight
                )
>>>>>>> REPLACE