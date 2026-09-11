MECHANISM: Global LayerNorm-null embedding-offset gauge

HYPOTHESIS: Centering the first positional feature will reduce the model from 1621 to 1620 parameters while retaining at least 99% accuracy, because transferring the eliminated joint token-position mean as an equal offset across all eight residual features leaves every LayerNorm output unchanged and alters tied logits only by a class-independent offset.

INTENDED_EDIT: Represent positional feature one with centered learned contrasts, subtract its eliminated joint mean from the remaining free positional and token columns during initialization, and preserve the original initialization RNG sequence.

EVIDENCE: All eight token-position mean redundancies have now been removed successfully, with the fifth positional-oriented gauge reaching 99.97% at 1621 parameters; the same proven positional contrast parameterization can fix the remaining global all-features offset symmetry.

<<<<<<< SEARCH
        self.embedding_dim = embedding_dim
        self.fourth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fifth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.sixth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.seventh_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.eighth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 5))
        self.register_buffer(
            "fourth_column_shift", torch.zeros(()), persistent=False
        )
=======
        self.embedding_dim = embedding_dim
        self.first_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fifth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.sixth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.seventh_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.eighth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 6))
        self.register_buffer(
            "first_column_shift", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "fourth_column_shift", torch.zeros(()), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _set_from_full(self, weight: torch.Tensor) -> None:
        fourth_column = weight[:, 3]
        fifth_column = weight[:, 4]
        sixth_column = weight[:, 5]
        seventh_column = weight[:, 6]
        eighth_column = weight[:, 7]
        with torch.no_grad():
            self.fourth_column.copy_(
=======
    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_column = weight[:, 0]
        fourth_column = weight[:, 3]
        fifth_column = weight[:, 4]
        sixth_column = weight[:, 5]
        seventh_column = weight[:, 6]
        eighth_column = weight[:, 7]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.fourth_column.copy_(
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.rest.copy_(weight[:, :3])
            self.fourth_column_shift.copy_(fourth_column.mean())
=======
            self.rest.copy_(weight[:, 1:3])
            self.first_column_shift.copy_(first_column.mean())
            self.fourth_column_shift.copy_(fourth_column.mean())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def tied_weight(self) -> torch.Tensor:
        fourth_anchored = torch.cat(
            (self.fourth_column, self.fourth_column.new_zeros(1))
        )
=======
    def tied_weight(self) -> torch.Tensor:
        first_anchored = torch.cat(
            (self.first_column, self.first_column.new_zeros(1))
        )
        fourth_anchored = torch.cat(
            (self.fourth_column, self.fourth_column.new_zeros(1))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fourth_column = fourth_anchored - fourth_anchored.mean()
        fifth_column = fifth_anchored - fifth_anchored.mean()
=======
        first_column = first_anchored - first_anchored.mean()
        fourth_column = fourth_anchored - fourth_anchored.mean()
        fifth_column = fifth_anchored - fifth_anchored.mean()
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.rest[:, :3],
                fourth_column.unsqueeze(1),
=======
                first_column.unsqueeze(1),
                self.rest,
                fourth_column.unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_shift = full_token_weight[:, :3].mean(dim=0)
        full_token_weight[:, 3].add_(self.pos_emb.fourth_column_shift)
        full_token_weight[:, 4].add_(self.pos_emb.fifth_column_shift)
        full_token_weight[:, 5].add_(self.pos_emb.sixth_column_shift)
        full_token_weight[:, 6].add_(self.pos_emb.seventh_column_shift)
        full_token_weight[:, 7].add_(self.pos_emb.eighth_column_shift)
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.rest[:, :3].add_(token_shift)
=======
        token_shift = full_token_weight[:, :3].mean(dim=0)
        global_shift = token_shift[0] + self.pos_emb.first_column_shift
        full_token_weight[:, 3].add_(self.pos_emb.fourth_column_shift)
        full_token_weight[:, 4].add_(self.pos_emb.fifth_column_shift)
        full_token_weight[:, 5].add_(self.pos_emb.sixth_column_shift)
        full_token_weight[:, 6].add_(self.pos_emb.seventh_column_shift)
        full_token_weight[:, 7].add_(self.pos_emb.eighth_column_shift)
        full_token_weight[:, 3:].sub_(global_shift)
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.rest.add_(token_shift[1:] - global_shift)
>>>>>>> REPLACE