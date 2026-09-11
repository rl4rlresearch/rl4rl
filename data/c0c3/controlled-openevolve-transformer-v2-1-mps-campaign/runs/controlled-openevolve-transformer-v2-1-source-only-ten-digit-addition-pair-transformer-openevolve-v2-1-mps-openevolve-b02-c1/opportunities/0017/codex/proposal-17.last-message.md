MECHANISM: Orthogonal reuse of residual common-mode weights

HYPOTHESIS: Reusing the LayerNorm-invisible common-mode component of the 12 MLP output-weight columns as the 12 hidden biases will reduce parameters from 1,576 to 1,564 while maintaining at least 99% accuracy.

INTENDED_EDIT: Remove the independent `fc1` bias, derive an equally conditioned hidden bias from the normalized row-sum of `fc2.weight`, and center that weight before producing the residual update.

EVIDENCE: Direct seven-coordinate MLP output projections failed at 93.33% and 12.75% despite common-mode redundancy, suggesting harmful optimization geometry; the successful 1,576-parameter design confirms that residual common modes are unobservable. This patch preserves the full output-weight tensor and repurposes its redundant orthogonal component instead of deleting it.

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
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_mean = self.fc2.weight.mean(dim=0, keepdim=True)
        hidden_bias = self.fc2.weight.sum(dim=0) / math.sqrt(self.fc2.out_features)
        hidden = F.gelu(F.linear(x[..., :-1], self.fc1.weight, hidden_bias))
        y = F.linear(hidden, self.fc2.weight - weight_mean)
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
>>>>>>> REPLACE