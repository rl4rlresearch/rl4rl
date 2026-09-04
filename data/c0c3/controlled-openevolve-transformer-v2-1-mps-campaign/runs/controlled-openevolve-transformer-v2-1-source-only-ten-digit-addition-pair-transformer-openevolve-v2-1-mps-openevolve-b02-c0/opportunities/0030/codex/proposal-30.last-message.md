MECHANISM: Single-position embedding common-mode gauge anchoring

HYPOTHESIS: Anchoring one coordinate of the final positional embedding row at zero will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because a position-specific all-coordinate offset propagates only through residual streams and is removed by every downstream LayerNorm.

INTENDED_EDIT: Replace the positional embedding with an initialization-preserving embedding that stores one fewer scalar and reconstructs the final coordinate as zero.

EVIDENCE: The current design reaches 99.84% at 1,632 parameters, and final-MLP common-mode bias anchoring previously retained 99.95%; this tests the same normalization gauge at a single positional row, minimizing the optimization disturbance that likely caused the attention-projection anchor failure.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with one common-mode coordinate anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )
        return F.embedding(idx, weight)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = PositionAnchoredEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            with torch.no_grad():
                module.weight.copy_(full_weight.flatten()[:-1])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE