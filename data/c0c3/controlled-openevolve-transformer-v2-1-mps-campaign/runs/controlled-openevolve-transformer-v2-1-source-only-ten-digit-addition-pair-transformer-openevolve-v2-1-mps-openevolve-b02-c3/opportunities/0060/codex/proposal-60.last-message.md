MECHANISM: Full-lexical six-dimensional residual state

HYPOTHESIS: A 829-parameter transformer will retain at least 99% accuracy because the five-dimensional lexical representation and seven nonlinear features are load-bearing, while the two classifier-orthogonal residual dimensions are not necessary for learned attention-based addition.

INTENDED_EDIT: Reconstruct the qualified 967-parameter design, reduce `d_model` from eight to six, and expand the token basis from `d_model - 3` to the full five-dimensional zero-mean subspace; preserve two attention heads, the seven-unit MLP, three learned final scales, and all three final-bias coordinates.

EVIDENCE: The 967-parameter design achieved 99.77%; the rank-four lexical bottleneck and six-unit MLP failed, so this patch retains rank five and seven nonlinear units while testing the previously shared assumption that computation requires two additional classifier-invisible scratch dimensions.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with three learned scales and remaining scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 3))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 3),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.fixed_weight))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 3
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-model", type=int, default=6)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE