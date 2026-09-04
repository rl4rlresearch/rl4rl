MECHANISM: Second cross-branch bias-coordinate sharing

HYPOTHESIS: Reusing a second attention projection-bias coordinate as one MLP output-bias coordinate will reduce the model from 1,316 to 1,315 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Replace one dedicated MLP output-bias parameter with `attn.proj.bias[0]`, while preserving the seven-coordinate MLP output bias, its derived hidden bias, zero initialization, and optimized linear computation.

EVIDENCE: The current tied-query/key model achieved 100% accuracy with 1,316 parameters, and the earlier cross-branch bias-sharing design achieved 100% accuracy while replacing one MLP bias coordinate with the attention projection-bias mean; this tests the smallest direct extension of that proven mechanism.

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shared_bias: torch.Tensor) -> torch.Tensor:
        output_bias = torch.cat((self.output_bias, shared_bias.reshape(1)))
        hidden = self.fc1(x) + output_bias.mean()
=======
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shared_biases: torch.Tensor) -> torch.Tensor:
        output_bias = torch.cat((self.output_bias, shared_biases.reshape(-1)))
        hidden = self.fc1(x) + output_bias.mean()
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_bias = self.attn.proj.bias.mean()
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), shared_bias)
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = torch.stack(
            (self.attn.proj.bias.mean(), self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x
>>>>>>> REPLACE