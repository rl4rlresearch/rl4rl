MECHANISM: Six-dimensional learned token bottleneck with eight-dimensional internal computation

HYPOTHESIS: Restricting the tied token embeddings to six learned channels while retaining the eight-channel transformer will reduce the model from 1426 to 1199 parameters and retain at least 99% accuracy, because learned relative-lag attention already supplies positional structure separately and addition-token identity should not require every residual channel.

INTENDED_EDIT: Replace each full-width token vector with a learned six-dimensional vector padded into the eight-dimensional residual stream, use the same compact vectors for logits, and preserve the verified initialization random stream by reconstructing and truncating the former conceptual initialization.

EVIDENCE: The learned relative-lag design reached 99.96% after removing the 184-parameter position table, and the current model reaches 99.93%; this challenges the shared assumption that all eight residual channels must be independently learned for every token while leaving the successful eight-dimensional attention and MLP computation intact.

<<<<<<< SEARCH
class ShiftGaugeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        size = num_embeddings * embedding_dim
        self.coordinates = nn.Parameter(torch.empty(size - 1))

        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coordinates, mean=0.0, std=0.02)

    def materialized_weight(self) -> torch.Tensor:
        flat = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, flat) / self.reflector_norm_sq
        flat = flat - self.reflector * projection
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())
=======
class CompactTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim <= 2:
            raise ValueError("embedding_dim must exceed the two-channel bottleneck")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compact_dim = embedding_dim - 2
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, self.compact_dim)
        )

        # Retain the former full embedding's conceptual initialization so this
        # change isolates representation width and preserves the random stream.
        size = num_embeddings * embedding_dim
        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("init_reflector", reflector, persistent=False)
        self.init_reflector_norm_sq = float(reflector.dot(reflector))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            size = self.num_embeddings * self.embedding_dim
            conceptual_coordinates = self.weight.new_empty(size - 1)
            nn.init.normal_(conceptual_coordinates, mean=0.0, std=0.02)
            flat = F.pad(conceptual_coordinates, (1, 0))
            projection = (
                2.0
                * torch.dot(self.init_reflector, flat)
                / self.init_reflector_norm_sq
            )
            flat = flat - self.init_reflector * projection
            conceptual_weight = flat.view(
                self.num_embeddings, self.embedding_dim
            )
            self.weight.copy_(conceptual_weight[:, : self.compact_dim])

    def materialized_weight(self) -> torch.Tensor:
        return F.pad(
            self.weight, (0, self.embedding_dim - self.compact_dim)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = CompactTokenEmbedding(cfg.vocab_size, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ShiftGaugeEmbedding):
=======
        if isinstance(module, CompactTokenEmbedding):
>>>>>>> REPLACE