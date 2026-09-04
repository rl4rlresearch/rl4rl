MECHANISM: Isolated MLP residual-output gauge fixing

HYPOTHESIS: Reducing only the MLP output projection from eight to seven learned coordinates will produce a 1,564-parameter model with at least 99% accuracy, because the omitted coordinate can be subtracted from every MLP output coordinate as a LayerNorm-invisible common-mode residual shift.

INTENDED_EDIT: Make `fc2` produce seven coordinates and zero-pad the eighth, while leaving the previously implicated attention output projection and shared value/output bias unchanged.

EVIDENCE: Seven-coordinate positional embeddings and MLP output bias reached 99.82%. The combined attention-and-MLP output reduction failed at 72.44%, but the attention projection bias is coupled to the value bias; isolating the independently gauge-equivalent MLP projection tests the supported invariance without disturbing that load-bearing attention pathway.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y + F.pad(self.fc2_bias, (0, 1))
=======
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        y = y + F.pad(self.fc2_bias, (0, 1))
>>>>>>> REPLACE