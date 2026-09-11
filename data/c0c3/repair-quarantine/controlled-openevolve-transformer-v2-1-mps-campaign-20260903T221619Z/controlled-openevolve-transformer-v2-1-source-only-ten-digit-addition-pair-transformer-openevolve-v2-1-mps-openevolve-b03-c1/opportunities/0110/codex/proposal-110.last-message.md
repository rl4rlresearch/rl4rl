MECHANISM: Projectively tied associative MLP

HYPOTHESIS: The ten GELU detectors are load-bearing, but their decoder directions need not be independently learned; decoding normalized detector directions through a shared learned 7×7 transport and ten learned gains will retain at least 99% accuracy while reducing parameters from 606 to 595.

INTENDED_EDIT: Retain all ten nonlinear MLP units, but replace their independent 70-parameter output matrix with a 49-parameter shared transport and ten per-unit gains.

EVIDENCE: Reducing `d_ff` from 10 to 9 collapsed accuracy to 39.46%, showing that removing a nonlinear detector is destructive. Conversely, the tied rank-four lexical input/output geometry reached 99.74%, and the current model reaches 99.89%, motivating an associative geometry that preserves detector count while challenging the assumption that every detector requires an unrelated output vector.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Each nonlinear detector retains an independent learned output
        # amplitude, while a shared transport maps detector directions into
        # residual-update directions.
        self.fc2_gain = nn.Parameter(self.fc2.weight.new_ones(d_ff))
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.fc2.weight = nn.Parameter(
                (self.fc1_basis @ centered).clone()
            )
=======
    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            decoder = self.fc1_basis @ centered
            encoder_direction = F.normalize(
                self.fc1.weight.T, dim=0
            )
            decoder_norm = torch.linalg.vector_norm(
                decoder, dim=0
            ).clamp_min(1e-8)
            decoder_direction = decoder / decoder_norm
            transport = torch.linalg.lstsq(
                encoder_direction.T, decoder_direction.T
            ).solution.T
            transported = transport @ encoder_direction
            gain = (transported * decoder).sum(dim=0) / (
                transported.square().sum(dim=0).clamp_min(1e-8)
            )
            self.fc2.weight = nn.Parameter(transport.clone())
            self.fc2_gain.copy_(gain)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        encoder_direction = F.normalize(
            self.fc1.weight.T, dim=0
        )
        decoder = (
            self.fc2.weight @ encoder_direction
        ) * self.fc2_gain.unsqueeze(0)
        fc2_weight = self.fc1_basis.T @ decoder
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)
>>>>>>> REPLACE