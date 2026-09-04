MECHANISM: Second MLP-output residual-shift weight gauge

HYPOTHESIS: Fixing `fc2.weight[0,4]` alongside the verified `fc2.weight[0,0]` anchor will produce a 1566-parameter model with at least 99% accuracy, because subtracting each coefficient from its full output column adds only feature-uniform residual shifts removed by the final LayerNorm.

INTENDED_EDIT: Store 94 learned MLP output weights, reconstruct `fc2.weight[0,0]` and `fc2.weight[0,4]` as zeros, and canonicalize both columns during fresh initialization.

EVIDENCE: The first MLP-output anchor achieved 99.46% at 1567 parameters; additionally, the nonadjacent column-4 projection anchor succeeded at 99.63% after the adjacent column-1 anchor failed, motivating column 4 as the best-supported second residual-gauge coordinate.

<<<<<<< SEARCH
class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with one residual-gauge weight and bias anchor."""
=======
class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with two residual-gauge weights and bias anchor."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
=======
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )
=======
    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (zero, self.weight_rest[:3], zero, self.weight_rest[3:])
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Removing this coefficient from every output row changes the
                # residual stream only by a feature-uniform, tokenwise shift.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0
                module.weight_rest.copy_(weight.flatten()[1:])
                nn.init.zeros_(module.bias_rest)
=======
                # Removing either coefficient from every output row changes
                # the residual stream only by feature-uniform tokenwise shifts.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:4], flat[5:]))
                )
                nn.init.zeros_(module.bias_rest)
>>>>>>> REPLACE