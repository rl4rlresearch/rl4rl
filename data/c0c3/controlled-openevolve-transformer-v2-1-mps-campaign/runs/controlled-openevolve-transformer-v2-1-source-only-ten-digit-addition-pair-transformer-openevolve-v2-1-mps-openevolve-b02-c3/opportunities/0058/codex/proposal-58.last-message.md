MECHANISM: Second classifier-visible terminal-scale fixing

HYPOTHESIS: Fixing a second classifier-visible terminal LayerNorm scale at one will produce a 968-parameter transformer with at least 99% accuracy, because four learned visible scales and the three-coordinate final latent bias retain substantial output calibration capacity.

INTENDED_EDIT: Replace the eight-parameter terminal LayerNorm with four learned scales and four fixed unit scales, preserving the verified seven-unit MLP and all attention parameters.

EVIDENCE: The adjacent 969-parameter design achieved 99.93% accuracy after fixing one classifier-visible scale, while removing another final-bias coordinate collapsed; extending the successful terminal-scale compression by one parameter is the narrowest informative next test.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with two visible scales and ignored scales fixed."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features - 2))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features + 2),
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