MECHANISM: Positional-embedding common-mode gauge fixing

HYPOTHESIS: Removing one common-mode coordinate from the first positional embedding will reduce the model to 1,606 parameters while retaining at least 99% accuracy, because per-token pre-LayerNorms and the final LayerNorm make uniform hidden-coordinate shifts functionally invisible.

INTENDED_EDIT: Replace the positional embedding with an otherwise equivalent learned embedding that omits the final coordinate of its first row, reconstructs it as zero, and gauge-adjusts initialization to preserve the initial model function and RNG stream.

EVIDENCE: The three-row `fc1` gauge reached 99.97% at 1,607 parameters, while extending it to a fourth row fell to 97.77%; this tests an orthogonal exact null direction instead of further constraining the sensitive MLP.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
class PositionGaugedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Consume the same constructor RNG stream as the replaced embedding.
        base = nn.Embedding(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            base.weight.new_empty(num_embeddings * embedding_dim - 1)
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        first = torch.cat(
            (
                self.weight[: self.embedding_dim - 1],
                self.weight.new_zeros(1),
            )
        )
        weight = torch.cat((first, self.weight[self.embedding_dim - 1 :])).view(
            self.num_embeddings, self.embedding_dim
        )
        return F.embedding(indices, weight)


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionGaugedEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                rows = module.gauged_rows
                full[:rows, :-1].sub_(full[:rows, -1:].clone())
                module.weight.copy_(
                    torch.cat(
                        (full[:rows, :-1].flatten(), full[rows:].flatten())
                    )
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, PositionGaugedEmbedding):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.num_embeddings, module.embedding_dim
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # A position-specific uniform hidden shift survives residual
                # additions but is erased before every learned sublayer and
                # before the logits by per-token LayerNorm.
                full[0, :-1].sub_(full[0, -1:].clone())
                module.weight.copy_(
                    torch.cat((full[0, :-1], full[1:].flatten()))
                )
        elif isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                rows = module.gauged_rows
                full[:rows, :-1].sub_(full[:rows, -1:].clone())
                module.weight.copy_(
                    torch.cat(
                        (full[:rows, :-1].flatten(), full[rows:].flatten())
                    )
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE