MECHANISM: Central value-bias anchoring

HYPOTHESIS: Fixing the mirrored central value-bias pair to zero will reduce the model from 1,612 to 1,611 parameters while retaining at least 99% accuracy, because its position-independent contribution can be absorbed by the attention output bias without forcing the third value-bias coordinate to serve four dimensions.

INTENDED_EDIT: Store three independent value-bias coordinates and reconstruct the bias as `[a, b, c, 0, 0, c, b, a]`.

EVIDENCE: The four-coordinate symmetric value bias achieved 99.98% accuracy, while tying its central coordinate to the third collapsed to 17.32%; anchoring the central pair tests whether that failure came from destructive coordinate conflation rather than removing the redundant additive degree itself.

<<<<<<< SEARCH
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-12].clone())
=======
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-13].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias[self.d_model + 3 : self.d_model + 4],
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
=======
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias.new_zeros(2),
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
>>>>>>> REPLACE