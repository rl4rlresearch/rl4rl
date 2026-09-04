MECHANISM: Reference-fixed shared terminal final-LayerNorm bias

HYPOTHESIS: Fixing the shared bias of residual coordinates one through six to the reference value will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because it preserves the distinct first-coordinate bias destroyed by the failed complete-sharing change.

INTENDED_EDIT: Store only the first final-LayerNorm bias contrast and reconstruct all remaining coordinates with zero bias.

EVIDENCE: Six-way terminal bias sharing achieved 99.98% while complete nonreference sharing collapsed to 14.27%; this isolates the first-coordinate contrast as load-bearing and makes removal of the remaining shared terminal contrast the most informative complementary test.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            [
                self.bias,
                self.bias[-1:].expand(5),
                self.bias.new_zeros(1),
            ]
        )
=======
        bias = F.pad(self.bias, (0, 7))
>>>>>>> REPLACE