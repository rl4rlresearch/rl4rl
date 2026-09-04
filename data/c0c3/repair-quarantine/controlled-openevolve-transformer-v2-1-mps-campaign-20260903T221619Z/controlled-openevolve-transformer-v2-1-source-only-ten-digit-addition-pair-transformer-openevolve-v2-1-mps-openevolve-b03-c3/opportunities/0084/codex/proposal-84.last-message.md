MECHANISM: Ninth terminal-output scalar-shift gauge

HYPOTHESIS: A 1,383-parameter relative-lag model will retain at least 99% accuracy because the verified 1,384-parameter design achieved 99.73%, while removing the common output component of a ninth `fc2` column uses the same exact pre-final-LayerNorm symmetry already verified for eight columns.

INTENDED_EDIT: Replace absolute position embeddings with gauge-fixed per-head relative-lag logits and extend the qualified terminal projection gauge from eight to nine columns, preserving omitted coordinates through clipping and ambient AdamW updates.

EVIDENCE: The 1,384-parameter relative-lag design reached 99.73% after extending the terminal gauge to eight columns; a ninth column is the smallest controlled reduction using the identical symmetry.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and six output-shift column gauges removed."""

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
    """Linear layer with bias and nine output-shift column gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(9)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 9)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 6:])
=======
        self.weight_rest.copy_(raw_weight[:, 9:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        # One additive constant per head is omitted because softmax is
        # invariant to shifting all valid relative-lag logits equally.
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
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 6:])
=======
                module.weight_rest.copy_(raw_weight[:, 9:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
        model.pos_emb.third,
    ]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
=======
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