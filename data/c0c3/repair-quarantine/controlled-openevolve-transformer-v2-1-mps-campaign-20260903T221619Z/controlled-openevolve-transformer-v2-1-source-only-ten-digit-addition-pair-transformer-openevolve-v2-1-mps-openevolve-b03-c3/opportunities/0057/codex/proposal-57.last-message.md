MECHANISM: Fifth terminal-column output-shift gauge with ambient AdamW

HYPOTHESIS: A 1,533-parameter model will retain at least 99% accuracy because the qualified 1,534-parameter tied-embedding/five-scale design already achieved 99.90%, while the fifth `fc2` column’s common-output component is exactly erased by the final LayerNorm and its full AdamW dynamics remain preserved.

INTENDED_EDIT: Reproduce the qualified tied-token gauge and five absorbed `ln2` scales, then remove the common-output component from a fifth terminal projection column and include it in the existing ambient gauge optimizer.

EVIDENCE: Reference Design 2 achieved 99.90% accuracy at 1,534 parameters using the tied-token gauge, five-scale absorption, and four terminal-column gauges; extending that already-qualified exact terminal symmetry by one column is the smallest controlled reduction.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class GaugeFixedPositionEmbedding(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class GaugeFixedTokenEmbedding(nn.Module):
    """Tied embedding with its global scalar-shift gauge removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 1.0) -> None:
        raw = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(raw, mean=0.0, std=std)
        flat = raw.reshape(-1)
        self.weight.copy_(flat[:-1] - flat[-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (self.weight, self.weight.new_zeros(1))
        ).view(self.num_embeddings, self.embedding_dim)
        if torch.is_grad_enabled():
            full_weight.retain_grad()
        self.full_weight = full_weight
        return F.embedding(idx, full_weight)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight)


class TiedTokenProjection(nn.Module):
    """Parameter-free output view of the learned token embedding."""

    def __init__(self, embedding: GaugeFixedTokenEmbedding):
        super().__init__()
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings
        object.__setattr__(self, "embedding", embedding)

        scratch = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.project(x)


class GaugeFixedPositionEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and five weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(5)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 5:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
class FourFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with four scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(4)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
=======
class FiveFixedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with five scales absorbed by following columns."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = FourFixedScaleLayerNorm(cfg.d_model)
=======
        self.ln2 = FiveFixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
=======
        self.cfg = cfg
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Parameter-free output view preserves input/output weight tying.
        self.lm_head = TiedTokenProjection(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTokenEmbedding):
            module.reset_parameters(std=0.02)
        elif isinstance(module, TiedTokenProjection):
            module.embedding.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            module.reset_parameters(std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
=======
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 5:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.pos_emb.first]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and five
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The last four ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(4, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -4:])
        for blk in model.blocks
    ]
=======
    # The last five ln2 scales are redundant with the corresponding fc1
    # columns. Keep their factorization only as optimizer-coordinate state,
    # while the model stores and uses the deduplicated effective columns.
    absorbed_scales = [
        torch.ones(5, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -5:])
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -4:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -4:].detach()
                / virtual_scale.unsqueeze(0)
            )
=======
        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -5:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -5:].detach()
                / virtual_scale.unsqueeze(0)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -4:].zero_()
=======
        for blk in model.blocks:
            blk.mlp.fc1.weight.grad[:, -5:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -4:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
=======
                blk.mlp.fc1.weight[:, -5:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
>>>>>>> REPLACE