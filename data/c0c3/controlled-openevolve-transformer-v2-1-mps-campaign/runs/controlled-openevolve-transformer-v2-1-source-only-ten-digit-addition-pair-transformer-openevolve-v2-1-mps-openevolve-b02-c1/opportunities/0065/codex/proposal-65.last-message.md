MECHANISM: Learned adjacent final-normalization bias sharing

HYPOTHESIS: Sharing the final LayerNorm’s last two learned bias coordinates will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because it preserves an adaptive offset where tying that coordinate to the fixed reference failed.

INTENDED_EDIT: Store six final LayerNorm bias coordinates and reconstruct the seventh learned coordinate from its adjacent predecessor; leave the eighth coordinate as the common-mode reference.

EVIDENCE: Removing all final LayerNorm biases collapsed accuracy to 15.64%, and tying one bias coordinate to the fixed reference reached only 19.67%, showing this pathway needs learned offsets; learned adjacent relative-bias sharing succeeded repeatedly, including the current 99.68% design, motivating an adaptive tie instead of deletion.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            F.pad(self.bias, (0, 1)),
            self.eps,
        )
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            [
                self.bias,
                self.bias[-1:],
                self.bias.new_zeros(1),
            ]
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            bias,
            self.eps,
        )
>>>>>>> REPLACE