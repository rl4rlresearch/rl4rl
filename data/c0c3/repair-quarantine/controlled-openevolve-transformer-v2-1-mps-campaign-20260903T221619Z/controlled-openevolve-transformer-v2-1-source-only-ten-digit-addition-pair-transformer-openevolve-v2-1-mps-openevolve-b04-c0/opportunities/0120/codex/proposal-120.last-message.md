MECHANISM: Mirrored cross-head query/key scale-and-shear gauge

HYPOTHESIS: Mirroring the verified head-0 row-2 shear in head 1—anchoring `qkv.weight[4,4]` and eliminating `qkv.weight[6,4]`—will yield 1566 parameters with at least 99% accuracy because both transformations exactly preserve attention scores.

INTENDED_EDIT: Move query row 4’s scale anchor from column 0 to column 4, store 178 learned QKV weights, and reconstruct `qkv.weight[6,4]` as zero using an inverse shear of key row 12 against key row 14.

EVIDENCE: The corresponding stable-pivot shear eliminating `qkv.weight[2,4]` reached 99.59%, and component-zero anchors subsequently succeeded in both attention heads; this tests the closest cross-head replication rather than another MLP residual anchor after `fc2.weight[0,4]` reached only 92.75%.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with six zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 13))

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
                self.weight_rest[12:15],
                zero,
                self.weight_rest[15:18],
                zero,
                self.weight_rest[18:25],
                fixed,
                self.weight_rest[25:32],
                fixed,
                self.weight_rest[32:39],
                fixed,
                self.weight_rest[39:46],
                fixed,
                self.weight_rest[46:56],
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:],
            )
        ).view(self.out_features, self.in_features)
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with seven zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 14))

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
                self.weight_rest[12:15],
                zero,
                self.weight_rest[15:18],
                zero,
                self.weight_rest[18:29],
                fixed,
                self.weight_rest[29:32],
                fixed,
                self.weight_rest[32:39],
                fixed,
                self.weight_rest[39:42],
                zero,
                self.weight_rest[42:45],
                fixed,
                self.weight_rest[45:55],
                zero,
                self.weight_rest[55:86],
                zero,
                self.weight_rest[86:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[4, 0]
                weight[4].mul_(scale)
                weight[12].div_(scale)

                scale = 0.02 / weight[5, 0]
                weight[5].mul_(scale)
                weight[13].div_(scale)

                scale = 0.02 / weight[6, 0]
                weight[6].mul_(scale)
                weight[14].div_(scale)

                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Canonicalize one key row along LayerNorm's null direction.
=======
                # Mirror head 0's stable column-4 pivot in head 1.
                scale = 0.02 / weight[4, 4]
                weight[4].mul_(scale)
                weight[12].div_(scale)

                scale = 0.02 / weight[5, 0]
                weight[5].mul_(scale)
                weight[13].div_(scale)

                scale = 0.02 / weight[6, 0]
                weight[6].mul_(scale)
                weight[14].div_(scale)

                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Eliminate the matching row-6 coefficient through the fixed
                # row-4 pivot, with the inverse key shear preserving scores.
                shear = weight[6, 4] / weight[4, 4]
                weight[6].sub_(shear * weight[4])
                weight[12].add_(shear * weight[14])
                weight[6, 4] = 0.0

                # Canonicalize one key row along LayerNorm's null direction.
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:67],
                            flat[68:99],
                            flat[100:],
=======
                            flat[25:36],
                            flat[37:40],
                            flat[41:48],
                            flat[49:52],
                            flat[53:56],
                            flat[57:67],
                            flat[68:99],
                            flat[100:],
>>>>>>> REPLACE