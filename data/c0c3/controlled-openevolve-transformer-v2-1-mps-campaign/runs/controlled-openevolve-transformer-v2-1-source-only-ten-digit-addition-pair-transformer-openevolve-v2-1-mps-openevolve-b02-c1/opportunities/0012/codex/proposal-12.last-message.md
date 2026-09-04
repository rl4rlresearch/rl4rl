MECHANISM: Residual-output common-mode gauge fixing

HYPOTHESIS: Constraining the attention and MLP output projections to seven learned residual coordinates will reduce parameters from 1,576 to 1,556 while retaining at least 99% accuracy, because their omitted eighth-coordinate contributions are per-position scalar residual shifts removed by subsequent LayerNorms.

INTENDED_EDIT: Make both residual-branch output projections produce seven coordinates and zero-pad the eighth, while retaining the attention projection bias as a full-width shared value/output bias.

EVIDENCE: Seven-coordinate positional embeddings and MLP output bias already achieved 99.82% accuracy at 1,576 parameters by removing the same LayerNorm-invariant common mode; this extends that successful exact gauge fixing to the projections generating those residual updates.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model - 1, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + self.proj.bias
=======
        q = q + self.q_bias
        v = v + self.proj_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = F.pad(self.proj(y), (0, 1)) + self.proj_bias
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y + F.pad(self.fc2_bias, (0, 1))
=======
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = F.pad(y + self.fc2_bias, (0, 1))
>>>>>>> REPLACE