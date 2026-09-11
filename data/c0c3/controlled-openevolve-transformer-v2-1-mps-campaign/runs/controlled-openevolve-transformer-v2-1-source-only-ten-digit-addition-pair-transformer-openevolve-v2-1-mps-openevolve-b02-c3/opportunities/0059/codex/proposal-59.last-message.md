MECHANISM: Third classifier-visible terminal-scale fixing

HYPOTHESIS: Fixing a third classifier-visible terminal LayerNorm scale at one will produce a 967-parameter transformer with at least 99% accuracy, because three learned visible scales and the retained three-coordinate final latent bias preserve sufficient output calibration.

INTENDED_EDIT: Reconstruct the verified seven-unit MLP and replace the terminal LayerNorm’s eight learned scales with three learned scales followed by five fixed unit scales.

EVIDENCE: The adjacent 968-parameter design achieved 99.98% accuracy after fixing two classifier-visible scales, improving on the successful 969-parameter design’s 99.93%; extending that same compression by one scale is the narrowest informative next test.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with three visible scales and remaining scales fixed."""

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
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=9)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE