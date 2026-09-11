MECHANISM: Anchored final-normalization gain

HYPOTHESIS: Fixing one observable final LayerNorm gain at its initial value of one will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because the remaining learned gains and tied embedding/readout weights can adapt its scale without adding per-step reconstruction overhead.

INTENDED_EDIT: Learn three rather than four visible final LayerNorm gains and materialize the fourth, along with the already-fixed hidden-channel gains, as ones.

EVIDENCE: The current 845-parameter model achieved 99.84% accuracy, while every subsequent 844-parameter design timed out; this tests a minimal, initialization-preserving reduction whose forward pass retains the existing single padding operation.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tail = self.normalized_shape[0] - self.output_dim
        weight = F.pad(self.weight, (0, tail), value=1.0)
        bias = F.pad(self.bias, (0, tail))
=======
        self.weight = nn.Parameter(torch.ones(output_dim - 1))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_tail = self.normalized_shape[0] - self.weight.numel()
        bias_tail = self.normalized_shape[0] - self.bias.numel()
        weight = F.pad(self.weight, (0, weight_tail), value=1.0)
        bias = F.pad(self.bias, (0, bias_tail))
>>>>>>> REPLACE