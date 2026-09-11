MECHANISM: LayerNorm-hyperplane canonicalization of one MLP input weight

HYPOTHESIS: Removing one redundant `fc1` weight coordinate while preserving the initialized function will reduce the model from 1,618 to 1,617 parameters and retain at least 99% accuracy, because LayerNorm outputs lie on an affine hyperplane and therefore one weight per downstream affine row is functionally redundant.

INTENDED_EDIT: Store 95 rather than 96 `fc1` weights, reconstruct the first coordinate as zero during inference, and shift the remaining first-row weights after initialization so the initial MLP preactivation is unchanged.

EVIDENCE: The current one-key/one-value-bias design achieved 99.96% at 1,618 parameters, while removing another value-bias coordinate fell to 96.11%; this motivates testing a distinct single-coordinate redundancy while preserving the successful initialization trajectory and initial model function.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = torch.cat(
            (
                self.fc1.weight.new_zeros(1),
                self.fc1.weight,
            )
        ).view(self.fc1.out_features, self.fc1.in_features)
        return self.drop(self.fc2(F.gelu(F.linear(x, fc1_weight, self.fc1.bias))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # A LayerNorm output lies on an affine hyperplane, so one input
        # weight in an affine row is redundant. Canonicalize the first
        # fc1 row while preserving its initialized function.
        for block in self.blocks:
            weight = block.mlp.fc1.weight.detach().reshape(-1)
            block.mlp.fc1.weight = nn.Parameter(
                torch.cat(
                    (
                        weight[1 : cfg.d_model] - weight[:1],
                        weight[cfg.d_model :],
                    )
                )
            )

    @staticmethod
>>>>>>> REPLACE