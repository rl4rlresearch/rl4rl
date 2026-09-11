MECHANISM: Global tied-token common-shift quotient

HYPOTHESIS: Applying the qualified three-active-query-bias construction and removing the exact global common-shift redundancy from the tied token matrix will produce 1,596 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Adopt the verified two-parameter query-bias mean reconstruction, represent the tied token matrix with one globally omitted coefficient, and preserve full-space initialization, AdamW moments, weight decay, and gradient clipping for the new gauge.

EVIDENCE: Reference Design 3 achieved 99.89% accuracy at 1,597 parameters; unlike failed capacity ablations and additional key/MLP quotients, a common scalar added to every tied token-matrix entry only shifts residual channels and all output logits uniformly, making it an orthogonal exact symmetry.

<<<<<<< SEARCH
class GaugeFixedEmbedding(nn.Embedding):
=======
class GaugeFixedTokenEmbedding(nn.Embedding):
    """Tied token matrix represented modulo one global common shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight, self.weight.new_zeros(1)))
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GaugeTiedLinear(nn.Linear):
    """Output projection dynamically tied to a gauge-fixed embedding."""

    def __init__(self, embedding: GaugeFixedTokenEmbedding):
        super().__init__(
            embedding.embedding_dim,
            embedding.num_embeddings,
            bias=False,
        )
        self.weight = None
        object.__setattr__(self, "embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class GaugeFixedEmbedding(nn.Embedding):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and four query biases remain.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
=======
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Two coordinates generate three active query biases, while
        # key/value biases and the remaining query coordinates stay at zero.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
=======
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.mean().unsqueeze(0))
        )
        bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # Weight tying with the dynamically reconstructed input embeddings.
        self.lm_head = GaugeTiedLinear(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedTokenEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                flat = full.reshape(-1)
                module.weight.copy_(flat[:-1] - flat[-1])
        elif isinstance(module, GaugeTiedLinear):
            # Reproduce the baseline tied-head initialization draw, then select
            # the equivalent gauge whose final flattened coordinate is zero.
            embedding = module.embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                flat = full.reshape(-1)
                embedding.weight.copy_(flat[:-1] - flat[-1])
        elif isinstance(module, GaugeFixedEmbedding):
>>>>>>> REPLACE

<<<<<<< SEARCH
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    key_gauges: List[
=======
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    token_parameter: torch.nn.Parameter,
    key_gauges: List[
>>>>>>> REPLACE

<<<<<<< SEARCH
        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())
=======
        if (
            parameter is position_parameter
            or parameter is token_parameter
        ):
            total_sq.add_(grad.sum(dim=-1).square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_parameter = model.pos_emb.weight
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    excluded = {id(position_parameter)}
=======
    position_parameter = model.pos_emb.weight
    token_parameter = model.token_emb.weight
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    excluded = {id(position_parameter), id(token_parameter)}
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizers = [
=======
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    token_optimizer = GaugeAdamW(
        token_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizers = [
>>>>>>> REPLACE

<<<<<<< SEARCH
    all_optimizers = [optimizer, position_optimizer, *key_optimizers]
=======
    all_optimizers = [
        optimizer,
        position_optimizer,
        token_optimizer,
        *key_optimizers,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
                model,
                position_parameter,
                key_gauges,
=======
                model,
                position_parameter,
                token_parameter,
                key_gauges,
>>>>>>> REPLACE

<<<<<<< SEARCH
        position_optimizer.step()
        optimizer.step()
=======
        position_optimizer.step()
        token_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE