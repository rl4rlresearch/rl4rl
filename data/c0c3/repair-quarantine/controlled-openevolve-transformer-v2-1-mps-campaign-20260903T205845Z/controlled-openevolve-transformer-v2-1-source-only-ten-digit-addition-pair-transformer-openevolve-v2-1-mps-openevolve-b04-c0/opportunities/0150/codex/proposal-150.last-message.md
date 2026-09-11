MECHANISM: Extend the exact feature-uniform residual projection gauge

HYPOTHESIS: Fixing `proj.weight[0,1]` at zero through the same column-offset canonicalization already used for four projection columns will reduce the model from 1562 to 1561 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add a fifth zero anchor to `ResidualGaugeLinear`, initialize it by subtracting column 1’s first-row coefficient from every output row, and omit that scalar from learned storage.

EVIDENCE: The current 1562-parameter design reached 0.9999 accuracy with four identical residual-gauge anchors, whereas extending the query/key shear to `qkv.weight[2,1]` fell to 0.721; this motivates reducing a demonstrated exact residual redundancy instead of another attention-coordinate shear.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with four weights fixed along feature-uniform residual gauges."""
=======
class ResidualGaugeLinear(nn.Module):
    """Projection with five weights fixed along feature-uniform residual gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 4)
        )
=======
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                zero,
                self.weight_rest[2:3],
                zero,
                self.weight_rest[3:],
            )
        ).view(self.out_features, self.in_features)
=======
        return torch.cat(
            (
                zero,
                zero,
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                zero,
                self.weight_rest[2:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 2].clone()
=======
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 1].clone()
                weight[:, 1].sub_(offset)
                weight[0, 1] = 0.0

                offset = weight[0, 2].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:6], flat[7:]))
                )
=======
                module.weight_rest.copy_(
                    torch.cat((flat[3:4], flat[5:6], flat[7:]))
                )
>>>>>>> REPLACE