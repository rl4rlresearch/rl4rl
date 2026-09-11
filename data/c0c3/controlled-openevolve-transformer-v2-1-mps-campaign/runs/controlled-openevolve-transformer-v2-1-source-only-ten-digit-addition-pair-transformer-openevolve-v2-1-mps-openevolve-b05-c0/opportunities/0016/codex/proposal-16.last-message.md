MECHANISM: Quotient-aware tied-embedding gauge fixing

HYPOTHESIS: Removing the global common-shift degree of freedom from the tied token embedding will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy, because the shift is canceled at transformer inputs by LayerNorm and adds only a vocabulary-wide constant to output logits, while virtual full-coordinate AdamW preserves the successful optimization dynamics.

INTENDED_EDIT: Store all but one coordinate of the tied token embedding, reconstruct its final coordinate as zero, preserve gauge-equivalent initialization and RNG ordering, and optimize it alongside the attention output bias with virtual full-coordinate moments and clipping.

EVIDENCE: The 1634-parameter attention-bias quotient reached 99.85% only with virtual full-coordinate AdamW; this patch applies that successful optimizer treatment to a distinct exact gauge while leaving all optimization-sensitive query, value, normalization, and MLP parameters intact.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedEmbedding(nn.Module):
    """Embedding with its global all-entries shift fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight.reshape(-1)[-1].clone()
        fixed = (full_weight - anchor).reshape(-1)[:-1].clone()
        self.weight = nn.Parameter(fixed)

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings,
            self.embedding_dim,
        )

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight.reshape(-1)[-1].clone()
        self.weight.copy_((full_weight - anchor).reshape(-1)[:-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


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

        self.apply(self._init_weights)
=======
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Consume the same constructor-time draws as the former tied Linear.
        _ = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # The former tied lm_head reinitialized the shared weight last.
        self._init_weights(self.token_emb)
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
        if isinstance(module, GaugeFixedEmbedding):
            full_weight = torch.empty(
                module.num_embeddings,
                module.embedding_dim,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [block.attn.proj_bias for block in model.blocks]
=======
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
>>>>>>> REPLACE