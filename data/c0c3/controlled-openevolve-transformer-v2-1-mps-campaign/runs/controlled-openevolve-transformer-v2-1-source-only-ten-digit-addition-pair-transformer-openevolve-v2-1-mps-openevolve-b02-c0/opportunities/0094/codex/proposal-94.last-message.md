MECHANISM: Function-preserving joint token–position translation gauge

HYPOTHESIS: Exactly compensating a second reference-token coordinate anchor in the positional embeddings, while preserving the verified 1,580-parameter initialization and RNG consumption, will retain at least 99% accuracy with 1,579 parameters.

INTENDED_EDIT: Remove the second-last reference-token scalar, initialize from the current 111-scalar token draw, translate every token in that coordinate, and absorb the translation into every positional row modulo its LayerNorm-invisible common mode.

EVIDENCE: A direct second token-coordinate anchor reached 96.04%, but RNG alignment alone reached only 71.34%; unlike both attempts, this patch applies the full joint token–position gauge transformation and therefore preserves the successful model’s initial computation.

<<<<<<< SEARCH
class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with one global common-mode scalar anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )
=======
class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with two reference-token coordinates anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-2].clone())
        self.register_buffer(
            "_initial_translation", torch.zeros(()), persistent=False
        )

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 2)).view(
            self.num_embeddings, self.embedding_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight = nn.Parameter(compact_weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
=======
        self.weight = nn.Parameter(compact_weight.clone())

    def add_coordinate_translation_(
        self, coordinate: int, value: torch.Tensor
    ) -> None:
        """Absorb a token-space translation into the positional parameters."""
        with torch.no_grad():
            self.weight[
                coordinate : self.earlier_gauge_index : self.embedding_dim
            ].add_(value)

            gauge_delta = self.gauge_basis[coordinate] * value
            for start in (
                self.earlier_gauge_index,
                self.preceding_gauge_index,
                self.leading_gauge_index,
                self.zeroth_gauge_index,
                self.first_gauge_index,
                self.second_gauge_index,
                self.third_gauge_index,
                self.fourth_gauge_index,
            ):
                self.weight[start : start + self.embedding_dim - 1].add_(
                    gauge_delta
                )

            self.weight[self.gauge_end_index + coordinate].add_(value)
            self.weight[self.tie_index + coordinate].add_(value)
            self.weight[self.anchor_index + coordinate].add_(value)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        with torch.no_grad():
            translation = self.token_emb._initial_translation.clone()
            coordinate = cfg.d_model - 2
            reference_start = (cfg.vocab_size - 1) * cfg.d_model
            self.token_emb.weight[
                coordinate : reference_start : cfg.d_model
            ].sub_(translation)
            self.pos_emb.add_coordinate_translation_(coordinate, translation)
            self.token_emb._initial_translation.zero_()

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
=======
        elif isinstance(module, TokenAnchoredEmbedding):
            baseline_weight = module.weight.new_empty(
                module.weight.numel() + 1
            )
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(baseline_weight[:-1])
                module._initial_translation.copy_(baseline_weight[-1])
        elif isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
>>>>>>> REPLACE