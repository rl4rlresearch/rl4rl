MECHANISM: Fourth query–key coordinate scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[3,0]` at 0.02 with reciprocal scaling of head-0 key row 3 will reduce the model to 1580 parameters while retaining at least 99% accuracy, because it uses the fourth independent coordinate-scale gauge in the same head whose first three nonzero anchors all passed verification.

INTENDED_EDIT: Store 186 learned QKV weights, reconstruct weights 0 and 2 as zero and weights 4, 8, 16, and 24 as 0.02, then initialize query row 3 and key row 11 with reciprocal scaling.

EVIDENCE: The successive nonzero anchors at query rows 0, 1, and 2 achieved 99.79%, 99.93%, and 99.86% respectively while reducing the model from 1583 to 1581 parameters; applying the identical function-preserving mechanism to the remaining head-0 coordinate is the most direct next reduction.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and two query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 5))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        fixed = self.weight_rest.new_full((1,), 0.02)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                fixed,
                self.weight_rest[2:5],
                fixed,
                self.weight_rest[5:12],
                fixed,
                self.weight_rest[12:],
            )
        ).view(self.out_features, self.in_features)
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and four query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 6))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        fixed = self.weight_rest.new_full((1,), 0.02)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                fixed,
                self.weight_rest[2:5],
                fixed,
                self.weight_rest[5:12],
                fixed,
                self.weight_rest[12:19],
                fixed,
                self.weight_rest[19:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Fix two nonzero query coefficients using independent Q/K
                # reciprocal scales while preserving the attention scores.
                scale = 0.02 / weight[0, 4]
                weight[0].mul_(scale)
                weight[8].div_(scale)

                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:],
                        )
                    )
                )
=======
                # Fix four nonzero query coefficients using independent Q/K
                # reciprocal scales while preserving the attention scores.
                scale = 0.02 / weight[0, 4]
                weight[0].mul_(scale)
                weight[8].div_(scale)

                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                scale = 0.02 / weight[3, 0]
                weight[3].mul_(scale)
                weight[11].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:24],
                            flat[25:],
                        )
                    )
                )
>>>>>>> REPLACE