MECHANISM: Shared pre-LayerNorm output-scale gauge anchoring

HYPOTHESIS: Fixing one shared pre-LayerNorm scale coefficient at its initialization value will reduce the verified model from 1,570 to 1,569 learned parameters while retaining at least 99% accuracy after 21,000 training steps.

INTENDED_EDIT: Store seven learned coefficients for the shared `ln1`/`ln2` scale and reconstruct the eighth as a fixed one during both normalizations, preserving the initial function and random stream.

EVIDENCE: The two-row `fc1` anchor reached 99.98% at 1,570 parameters, while extending it to a third row reached only 29.83% and an initialization-preserving retry reached 57.91%; this tests a different exact redundancy without further constraining the sensitive MLP rows.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.weight = self.ln1.weight
        self.ln2.bias = self.ln1.bias
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.weight = self.ln1.weight
        self.ln2.bias = self.ln1.bias
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    @staticmethod
    def _norm(x: torch.Tensor, layer: nn.LayerNorm) -> torch.Tensor:
        weight = torch.cat((layer.weight, layer.weight.new_ones(1)))
        return F.layer_norm(
            x,
            layer.normalized_shape,
            weight,
            layer.bias,
            layer.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self._norm(x, self.ln1))
        x = x + self.mlp(self._norm(x, self.ln2))
        return x
>>>>>>> REPLACE