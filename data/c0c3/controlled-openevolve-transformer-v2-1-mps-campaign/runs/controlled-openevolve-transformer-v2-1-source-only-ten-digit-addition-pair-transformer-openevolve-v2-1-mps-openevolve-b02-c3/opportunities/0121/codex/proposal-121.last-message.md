MECHANISM: Single-coordinate cross-token lexical sharing

HYPOTHESIS: A 671-parameter transformer will retain at least 99% accuracy because sharing one direct input-code coordinate between two tokens preserves four-coordinate token codes, both nonlinear lifts, and the qualified 672-parameter attention/MLP architecture.

INTENDED_EDIT: Replace the token embedding table with a one-scalar-smaller parameterization in which the first coordinate of the first two token codes shares one learned coefficient.

EVIDENCE: The 672-parameter design achieved 99.52%, while collapsing every token to three coordinates fell to 39.64% and removing the remaining MLP output bias fell to 33.33%; this motivates an isolated one-scalar lexical tie that retains the full four-coordinate representation and leaves load-bearing attention and MLP paths unchanged.

<<<<<<< SEARCH
        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank + 1)
=======
        if num_embeddings < 2:
            raise ValueError("at least two token codes are required")

        self.num_embeddings = num_embeddings
        self.rank = rank
        self.code_coeff = nn.Parameter(
            torch.empty(num_embeddings * rank - 1)
        )
        nn.init.normal_(self.code_coeff, mean=0.0, std=0.02)

        basis = torch.zeros(embedding_dim, rank + 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def projection_weight(self) -> torch.Tensor:
        return self.basis

    @staticmethod
=======
    def projection_weight(self) -> torch.Tensor:
        return self.basis

    def code_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.code_coeff[: self.rank],
                self.code_coeff[:1],
                self.code_coeff[self.rank :],
            )
        ).view(self.num_embeddings, self.rank)

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        code = self._lift(self.code(tokens), classifier=False)
        return F.linear(code, self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projection_weight()
        latent = F.linear(x, weight.transpose(0, 1))
        classifier_code = self._lift(self.code.weight, classifier=True)
        return F.linear(latent, classifier_code)
=======
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_code = F.embedding(tokens, self.code_weight())
        code = self._lift(token_code, classifier=False)
        return F.linear(code, self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projection_weight()
        latent = F.linear(x, weight.transpose(0, 1))
        classifier_code = self._lift(
            self.code_weight(), classifier=True
        )
        return F.linear(latent, classifier_code)
>>>>>>> REPLACE