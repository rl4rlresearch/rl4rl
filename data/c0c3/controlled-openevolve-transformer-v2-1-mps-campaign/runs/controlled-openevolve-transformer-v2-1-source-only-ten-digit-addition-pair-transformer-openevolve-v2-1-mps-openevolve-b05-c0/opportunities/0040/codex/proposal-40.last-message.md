MECHANISM: Per-position residual-stream shift quotient

HYPOTHESIS: Fixing one all-coordinate shift per positional embedding row will reduce the model by `max_seq_len` learned parameters while retaining at least 99% accuracy, because with zero dropout each shift passes unchanged through residual connections and is removed by every subsequent LayerNorm.

INTENDED_EDIT: Replace the positional embedding with row-wise gauge-fixed parameters and train each row using the existing virtual-coordinate AdamW and gradient-clipping logic.

EVIDENCE: The 1622-parameter model achieved 99.92% using the same virtual-coordinate optimizer for successful exact shift quotients; unlike the failed final value-bias and MLP-bias reductions, every positional-row shift is independently invisible to the pre-LayerNorm blocks and final LayerNorm.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedPositionEmbedding(nn.Module):
    """Positional embedding with each row's residual-stream shift fixed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match nn.Embedding's constructor-time random-number consumption.
        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        fixed = full_weight - full_weight[:, -1:]
        self.weight = nn.ParameterList(
            nn.Parameter(row[:-1].clone()) for row in fixed
        )

    def full_weight(self) -> torch.Tensor:
        return torch.stack(
            [F.pad(row, (0, 1)) for row in self.weight],
            dim=0,
        )

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        fixed = full_weight - full_weight[:, -1:]
        for parameter, row in zip(self.weight, fixed):
            parameter.copy_(row[:-1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len,
            cfg.d_model,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
=======
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            reference = module.weight[0]
            full_weight = torch.empty(
                module.num_embeddings,
                module.embedding_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
=======
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
        *model.pos_emb.weight,
    ]
>>>>>>> REPLACE