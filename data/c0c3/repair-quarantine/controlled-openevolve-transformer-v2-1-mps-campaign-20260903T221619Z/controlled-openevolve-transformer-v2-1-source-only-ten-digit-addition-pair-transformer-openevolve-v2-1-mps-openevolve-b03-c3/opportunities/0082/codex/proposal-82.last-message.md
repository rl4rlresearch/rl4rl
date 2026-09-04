MECHANISM: Relative-lag attention with a seventh residual-output gauge

HYPOTHESIS: A 1,385-parameter model will retain at least 99% accuracy because the verified 1,386-parameter relative-lag design achieved 99.50%, while removing the common output component of a seventh `fc2` column is the same exact pre-LayerNorm scalar-shift symmetry already used successfully on six columns.

INTENDED_EDIT: Replace absolute positional embeddings with gauge-fixed learned per-head relative-lag logits, then extend the terminal projection’s ambient-AdamW output gauge from six to seven columns.

EVIDENCE: The gauge-fixed relative-lag design reached 99.50% with 1,386 parameters, substantially outperforming other positional compression approaches; its terminal projection already removes six equivalent column-wise output shifts, motivating one controlled additional reduction.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and six weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(6)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 6)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and seven weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(7)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 7)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 6:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 7:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        # One additive constant per head is omitted because softmax is
        # invariant to shifting every valid lag logit equally.
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
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
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
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 6:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 7:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Optimize omitted gauge coordinates in their full ambient spaces.
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
        model.pos_emb.third,
        model.pos_emb.fourth,
        model.pos_emb.fifth,
        model.pos_emb.sixth,
        model.pos_emb.seventh,
    ]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
=======
    # Optimize omitted gauge coordinates in their full ambient spaces.
    gauge_params = [
        model.token_emb.weight,
    ]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.relative_bias)
        gauge_params.extend(blk.attn.proj.weight_prefix)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
            model.pos_emb.full_third.grad.detach(),
            model.pos_emb.full_fourth.grad.detach(),
            model.pos_emb.full_fifth.grad.detach(),
            model.pos_emb.full_sixth.grad.detach(),
            model.pos_emb.full_seventh.grad.detach(),
        ]
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
        ]
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