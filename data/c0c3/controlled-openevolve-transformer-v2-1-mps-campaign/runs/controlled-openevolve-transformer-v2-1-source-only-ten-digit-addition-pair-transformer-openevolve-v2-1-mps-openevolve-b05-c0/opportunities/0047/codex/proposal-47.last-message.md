MECHANISM: Single-position residual-stream shift quotient

HYPOTHESIS: Fixing the all-coordinate shift of positional row 0 will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because the shift is removed by every subsequent LayerNorm and its virtual AdamW coordinate preserves full-row optimization geometry.

INTENDED_EDIT: Replace positional row 0 with seven gauge-fixed coordinates, retain ordinary parameters for all other rows, and optimize the reduced row through the existing virtual-coordinate AdamW and gradient-clipping path.

EVIDENCE: The 1621-parameter MLP gauge reached 99.93%, proving another exact one-coordinate quotient can train successfully; the 1599-parameter positional experiment removed all 23 row shifts simultaneously and reached 73.41%, motivating the more informative test of one isolated positional-row quotient.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedPositionEmbedding(nn.Module):
    """Position embedding with row 0's all-coordinate shift fixed to zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Match nn.Embedding's constructor-time random-number consumption.
        source = nn.Embedding(num_embeddings, embedding_dim)
        full_weight = source.weight.detach()
        anchor = full_weight[0, -1].clone()
        self.first_weight = nn.Parameter(
            (full_weight[0, :-1] - anchor).clone()
        )
        self.rest_weight = nn.Parameter(full_weight[1:].clone())

    def full_weight(self) -> torch.Tensor:
        first = F.pad(self.first_weight, (0, 1)).unsqueeze(0)
        return torch.cat((first, self.rest_weight), dim=0)

    @torch.no_grad()
    def reset_from_full_(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[0, -1].clone()
        self.first_weight.copy_(full_weight[0, :-1] - anchor)
        self.rest_weight.copy_(full_weight[1:])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedQKV(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedPositionEmbedding(
            cfg.max_seq_len,
            cfg.d_model,
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
=======
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedPositionEmbedding):
            full_weight = torch.empty(
                module.num_embeddings,
                module.embedding_dim,
                device=module.first_weight.device,
                dtype=module.first_weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedQKV):
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
=======
    gauge_parameters = [
        model.token_emb.weight,
        model.pos_emb.first_weight,
        *[block.attn.proj_bias for block in model.blocks],
>>>>>>> REPLACE