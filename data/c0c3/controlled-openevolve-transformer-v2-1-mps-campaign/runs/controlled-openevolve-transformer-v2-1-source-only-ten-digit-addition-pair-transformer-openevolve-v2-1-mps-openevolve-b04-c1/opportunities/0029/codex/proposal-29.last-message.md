MECHANISM: Tied-embedding global-shift gauge fixing

HYPOTHESIS: Removing the single global all-ones degree of freedom from the tied token embedding will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the shift is erased from inputs by LayerNorm and adds only a vocabulary-wide common logit at the output.

INTENDED_EDIT: Represent the tied token embedding in an orthonormal zero-mean basis with one fewer learned coordinate, and use its materialized weight for both input lookup and output logits.

EVIDENCE: The first-LayerNorm gauge fixing retained 99.92% accuracy at 1607 parameters, whereas cuts to query capacity and projection weights failed. Although the positional-shift gauge failed, this tests the distinct tied-embedding symmetry while preserving isotropic quotient initialization and all attention and MLP capacity.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class ShiftGaugeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        size = num_embeddings * embedding_dim
        self.coordinates = nn.Parameter(torch.empty(size - 1))

        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coordinates, mean=0.0, std=0.02)

    def materialized_weight(self) -> torch.Tensor:
        flat = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, flat) / self.reflector_norm_sq
        flat = flat - self.reflector * projection
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.materialized_weight())
>>>>>>> REPLACE