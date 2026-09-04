MECHANISM: Per-head relative-logit softmax gauge fixing

HYPOTHESIS: A 1,386-parameter model will retain at least 99% accuracy because the verified 1,388-parameter relative-lag design reached 99.22%, while removing one additive constant from each head’s lag logits is an exact softmax symmetry with ambient AdamW dynamics preserved.

INTENDED_EDIT: Store each head’s learned relative-lag vector with its final coordinate fixed to zero, reconstruct the full vectors during attention, and optimize both omitted coordinates through the existing gauge-aware clipping and AdamW path.

EVIDENCE: The current learned relative-lag design achieved 99.22% at 1,388 parameters, and prior positional/output gauge reductions repeatedly preserved accuracy when omitted coordinates retained full ambient optimizer dynamics.

<<<<<<< SEARCH
        # Each head learns which causal lags are useful, providing positional
        # routing without injecting a full vector for every absolute position.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len)
        )
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
=======
        # Additive constants in each head's lag logits cancel in softmax.
        self.relative_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 1))
                for _ in range(n_head)
            ]
        )
        self.full_relative_bias = None
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att + self.relative_bias[:, lags].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
=======
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
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # three attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [
        model.token_emb.weight,
    ]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # relative-logit, attention-output, and terminal-output gauges.
    gauge_params = [
        model.token_emb.weight,
    ]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.relative_bias)
        gauge_params.extend(blk.attn.proj.weight_prefix)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
=======
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_grad.detach()
                for full_grad in blk.attn.full_relative_bias.grad
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
>>>>>>> REPLACE