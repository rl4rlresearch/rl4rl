MECHANISM: Single-column residual common-mode gauge fixing

HYPOTHESIS: Removing only one LayerNorm-invisible common-mode weight from one MLP output column will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because the other 11 columns keep their original optimization geometry.

INTENDED_EDIT: Split `fc2` into an eight-output projection for 11 hidden features and a seven-coordinate projection vector for the final feature, padding its eighth residual coordinate with zero.

EVIDENCE: Removing all 12 MLP output common modes at once reached only 93.33% or 12.75%, despite exact functional redundancy, indicating an optimization-geometry problem; the 1,576-parameter model reached 99.82%, motivating the smallest possible isolated gauge reduction.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
=======
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2_main = nn.Linear(d_ff - 1, d_model, bias=False)
        self.fc2_last = nn.Parameter(torch.empty(d_model - 1))
        nn.init.normal_(self.fc2_last, mean=0.0, std=0.02)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x[..., :-1]))
        y = self.fc2_main(h[..., :-1])
        y = y + F.pad(h[..., -1:] * self.fc2_last, (0, 1))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
>>>>>>> REPLACE