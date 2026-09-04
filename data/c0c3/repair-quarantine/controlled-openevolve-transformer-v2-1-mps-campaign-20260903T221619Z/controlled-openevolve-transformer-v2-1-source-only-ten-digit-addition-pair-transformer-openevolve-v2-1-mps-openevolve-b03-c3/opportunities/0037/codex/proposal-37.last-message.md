MECHANISM: Attention output-bias shift gauge

HYPOTHESIS: Gauge-fixing the all-ones component of the attention output bias will reduce the qualified 1,542-parameter model to 1,541 parameters while retaining at least 99% accuracy, because the resulting scalar residual shift is erased by both the second pre-norm and final LayerNorm.

INTENDED_EDIT: Reproduce the verified shared key/value, bias-free value, bias-free `ln1` design and represent the attention projection bias with seven learned differences, preserving its full eight-coordinate AdamW dynamics.

EVIDENCE: The shared-key/value, bias-free-value, bias-free-`ln1` design achieved 99.91% accuracy at 1,542 parameters; four analogous terminal output-direction gauges remained qualified, while the failed tied-embedding gauge motivates testing this distinct local invariance.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with its shift-invariant bias scalar removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_bias = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_bias.retain_grad()
            self.full_bias = full_bias
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # Preserve independent learned queries while broadcasting one learned
        # key/value representation across both attention maps.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedTerminalLinear):
=======
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedTerminalLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and four terminal-weight gauge vectors.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
=======
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-bias, terminal-bias, and four terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.attn.proj.bias)
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
=======
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
>>>>>>> REPLACE