MECHANISM: Residual-stream common-mode gauge fixing

HYPOTHESIS: Removing the unobservable all-ones component from every positional embedding and from the MLP output bias will reduce the model from 1,600 to 1,576 parameters while preserving at least 99% accuracy, because all subsequent pre-LayerNorm computations and the final LayerNorm are invariant to per-position scalar residual shifts.

INTENDED_EDIT: Store seven learned coordinates per positional embedding and seven learned coordinates for the MLP output bias, padding each with a fixed zero eighth coordinate in the residual stream.

EVIDENCE: The seven-coordinate non-affine MLP input reached 99.99%, while reducing positional embeddings to four dimensions failed at 73.99%; this patch retains all seven observable positional contrast dimensions and removes only exact LayerNorm-invariant common-mode degrees of freedom.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x[..., :-1]))))
=======
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        pos_emb = F.pad(self.pos_emb(pos), (0, 1))
        x = self.token_emb(idx) + pos_emb
>>>>>>> REPLACE