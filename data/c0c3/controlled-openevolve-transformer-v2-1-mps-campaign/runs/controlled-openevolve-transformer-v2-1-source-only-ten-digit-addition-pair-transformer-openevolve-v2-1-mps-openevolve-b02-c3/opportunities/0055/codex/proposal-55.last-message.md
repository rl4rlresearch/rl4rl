MECHANISM: Classifier-aware terminal normalization pruning

HYPOTHESIS: The qualified seven-unit MLP will retain at least 99% accuracy with 970 parameters because the tied rank-five classifier ignores the terminal LayerNorm’s final two output coordinates, so fixing their scales at one removes two functionally irrelevant learned parameters.

INTENDED_EDIT: Restore the verified `d_ff=7` architecture and replace its eight-parameter terminal LayerNorm scale with six learned classifier-visible scales plus two fixed scales.

EVIDENCE: The seven-unit design achieved 99.66% accuracy with 972 parameters, while six units collapsed to 24.19%; this motivates preserving the seven nonlinear features and instead removing two terminal scales whose coordinates are outside the five-dimensional classifier subspace.

<<<<<<< SEARCH
        return x


class TinyDecoderLM(nn.Module):
=======
        return x


class ClassifierAwareLayerNorm(nn.Module):
    """LayerNorm with fixed scales on coordinates ignored by the classifier."""

    def __init__(self, normalized_shape: int, relevant_features: int):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(relevant_features))
        self.register_buffer(
            "fixed_weight",
            torch.ones(normalized_shape - relevant_features),
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
    p.add_argument("--d-ff", type=int, default=10)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE