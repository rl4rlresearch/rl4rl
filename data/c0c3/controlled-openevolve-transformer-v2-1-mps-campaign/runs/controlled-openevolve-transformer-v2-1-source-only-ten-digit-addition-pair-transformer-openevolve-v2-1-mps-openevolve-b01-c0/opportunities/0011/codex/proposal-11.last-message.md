MECHANISM: Incremental pre-MLP LayerNorm bias reduction

HYPOTHESIS: Fixing a second `ln2` bias coordinate at zero will reduce the model from 1635 to 1634 parameters while retaining at least 99% accuracy, because the 1635-parameter design achieved 99.87% and both removed offsets are locally representable by the learned `fc1` bias.

INTENDED_EDIT: Store six learned `ln2` bias coordinates instead of seven and append two fixed zeros during normalization.

EVIDENCE: Fixing one `ln2` bias coordinate at zero retained 99.87% accuracy at 1635 parameters; this is the only tested single-parameter reduction that passed and directly motivates a conservative second-coordinate ablation in the same redundant pathway.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(2)))
>>>>>>> REPLACE