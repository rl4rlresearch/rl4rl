MECHANISM: Optimizer-preserving attention residual gauge quotient

HYPOTHESIS: Removing the eight LayerNorm-invisible common-mode weights from the attention output projection will reduce the model from 1,446 to 1,438 parameters while retaining at least 99% accuracy when full-width initialization and virtual AdamW moments are preserved.

INTENDED_EDIT: Store seven output rows for the attention projection, reconstruct an eighth zero row, retain its full-width shared value/output bias, and train the omitted row through the existing quotient-aware optimizer.

EVIDENCE: Quotient-aware optimization let the seven-row MLP projection reach 99.75% after direct reparameterizations failed at 93.33% and 12.75%; the earlier combined attention/MLP reduction lacked this optimizer-preserving treatment, while the current 1,446-parameter model provides 99.93% accuracy headroom.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedResidualProjection(nn.Module):
    """Residual projection modulo its all-ones output direction."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Preserve the RNG stream of the removed full-width Linear constructor.
        discarded_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        torch.empty(out_features).uniform_(-bound, bound)

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight.copy_(full_weight[:-1] - full_weight[-1:])
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 0, 0, 1))
        return F.linear(x, weight, self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = GaugeFixedResidualProjection(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, GaugeFixedEmbedding):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedLMHead):
            module.embedding.initialize_from_full_normal()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        if isinstance(module, GaugeFixedEmbedding):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedLMHead):
            module.embedding.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedResidualProjection):
            module.initialize_from_full_normal()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
=======
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.proj.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
>>>>>>> REPLACE