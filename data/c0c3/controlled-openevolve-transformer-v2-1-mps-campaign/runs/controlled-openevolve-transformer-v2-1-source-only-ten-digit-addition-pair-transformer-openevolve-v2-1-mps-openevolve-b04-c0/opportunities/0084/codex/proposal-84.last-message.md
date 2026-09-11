MECHANISM: Query-key basis gauge anchoring

HYPOTHESIS: Fixing `qkv.weight[0,0]` at zero will reduce the model to 1585 parameters while retaining at least 99% accuracy, because an invertible query/key basis transformation can absorb one query-coordinate constraint without reducing the attention score functions representable by the head.

INTENDED_EDIT: Replace the QKV linear layer with an otherwise identical learned projection whose first query weight is fixed at zero, while preserving initialization random-number consumption and all 191 remaining learned QKV weights.

EVIDENCE: The 1586-parameter design achieved 99.90%, while eliminating another LayerNorm gain through detached sharing reached only 98.53%; this tests a separate attention redundancy while preserving both empirically sensitive gains.

<<<<<<< SEARCH
    def position(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.pos_weight())


class CausalSelfAttention(nn.Module):
=======
    def position(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.pos_weight())


class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with one query weight fixed by query-key basis gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 1))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = AttentionWeightAnchoredLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SharedAnchorEmbeddings):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, AttentionWeightAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight_rest.copy_(weight.flatten()[1:])
        elif isinstance(module, SharedAnchorEmbeddings):
>>>>>>> REPLACE