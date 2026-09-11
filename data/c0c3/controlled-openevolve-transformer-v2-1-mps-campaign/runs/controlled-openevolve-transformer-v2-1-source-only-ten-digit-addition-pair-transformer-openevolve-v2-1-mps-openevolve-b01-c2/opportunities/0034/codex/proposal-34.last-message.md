MECHANISM: Per-position residual-stream shift gauge fixing

HYPOTHESIS: Extending the verified 1,591-parameter design by gauge-fixing the all-ones feature direction in every positional embedding row will remove `max_seq_len` additional parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Make `ln2` affine-free, gauge every `fc1` row as in Reference Design 3, and replace the positional embedding with a learned embedding that omits one exact common-shift coordinate per position.

EVIDENCE: Reference Design 3 reached 99.96% with 1,591 parameters after making all `fc1` gauges exact; unlike the failed learned-`ln1` QKV gauges, a positional row’s common feature shift is canceled throughout the zero-dropout pre-norm residual network and by the final LayerNorm.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class PositionShiftGaugedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as nn.Embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        retained = num_embeddings * (embedding_dim - 1)
        self.weight = nn.Parameter(base.weight.new_empty(retained))

    def full_weight(self) -> torch.Tensor:
        gauged = self.weight.view(
            self.num_embeddings, self.embedding_dim - 1
        )
        return torch.cat(
            (gauged, self.weight.new_zeros(self.num_embeddings, 1)),
            dim=1,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8, 9)
=======
        self.gauged_rows = tuple(range(out_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
=======
        # With affine-free normalization, every common fc1 row coefficient is
        # an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = PositionShiftGaugedEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedLinear):
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, PositionShiftGaugedEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # Adding a position-dependent scalar to every feature only
                # shifts the residual stream along LayerNorm's null direction.
                gauged = full.clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                module.weight.copy_(gauged[:, :-1].flatten())
        elif isinstance(module, LayerNormGaugedLinear):
>>>>>>> REPLACE