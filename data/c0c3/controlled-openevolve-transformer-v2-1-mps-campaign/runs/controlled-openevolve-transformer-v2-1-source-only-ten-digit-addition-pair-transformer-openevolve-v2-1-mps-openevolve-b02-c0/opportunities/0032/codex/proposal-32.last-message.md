MECHANISM: Global tied-token embedding common-mode gauge anchoring

HYPOTHESIS: Anchoring one global scalar of the tied token embedding will reduce the model from 1,631 to 1,630 parameters while retaining at least 99% accuracy, because subtracting that scalar from every embedding coordinate produces only a LayerNorm-invariant input offset and a softmax-invariant common logit offset.

INTENDED_EDIT: Store one fewer token-embedding scalar, reconstruct the final scalar as zero, initialize by subtracting the sampled final scalar globally, and preserve weight tying through a parameterless output projection wrapper.

EVIDENCE: A single positional common-mode anchor achieved 99.93% at 1,631 parameters, while extending it to a second positional row fell to 72.89%; this motivates testing the same exact normalization gauge in the distinct tied-token embedding rather than further constraining positional rows.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with its global common-mode gauge anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class TiedTokenLinear(nn.Module):
    """Parameterless output projection tied to an anchored token embedding."""

    def __init__(self, embedding: TokenAnchoredEmbedding):
        super().__init__()
        object.__setattr__(self, "_embedding", embedding)

    @property
    def weight(self) -> nn.Parameter:
        return self._embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._embedding.full_weight())


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = TokenAnchoredEmbedding(cfg.vocab_size, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.lm_head = TiedTokenLinear(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, PositionAnchoredEmbedding):
=======
        if isinstance(module, TokenAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight.sub_(anchor)
            with torch.no_grad():
                module.weight.copy_(full_weight.flatten()[:-1])
        elif isinstance(module, PositionAnchoredEmbedding):
>>>>>>> REPLACE