MECHANISM: Third final-classifier latent-bias coordinate fixing

HYPOTHESIS: The verified seven-unit, classifier-aware 970-parameter design will retain at least 99% accuracy with 969 parameters when a third zero-initialized final latent-bias coordinate is fixed at zero, because the first two such reductions retained 99.77% and 99.92% accuracy.

INTENDED_EDIT: Reconstruct the qualified `d_ff=7` architecture with six learned terminal LayerNorm scales, then shorten the final latent bias from three learned coordinates to two and reconstruct the remaining three as fixed zeros.

EVIDENCE: The seven-unit classifier-aware design achieved 99.61% accuracy with 970 parameters, while successive final-bias reductions previously retained 99.77% and 99.92%; a one-coordinate extension of that successful output-path compression is the narrowest informative change below 970 parameters.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
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
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
=======
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_bias = F.pad(self.final_bias, (0, 2))
=======
        final_bias = F.pad(self.final_bias, (0, 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE