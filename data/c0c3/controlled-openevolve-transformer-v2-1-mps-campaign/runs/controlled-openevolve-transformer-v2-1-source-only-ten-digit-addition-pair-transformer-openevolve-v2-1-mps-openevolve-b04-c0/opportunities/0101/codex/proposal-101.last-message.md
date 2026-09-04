MECHANISM: Balanced query–key norm gauge with stereographic direction coordinates

HYPOTHESIS: Constraining head-0 query row 3 and its matching key row to equal norms will produce a 1576-parameter model with at least 99% accuracy, because it removes the same exact scale redundancy as the failed coefficient anchors while preserving initialization and avoiding a potentially ill-conditioned fixed coefficient.

INTENDED_EDIT: Store 182 learned QKV values, encode key row 11’s direction with seven stereographic coordinates, derive its norm from query row 3, and initialize the pair through reciprocal scaling that exactly preserves attention scores.

EVIDENCE: The seven coefficient-based Q/K scale anchors reached 99.13% at 1577 parameters, proving this gauge family is viable, but both fixed-coefficient attempts for head-0 row 3 failed at 71.61% and 39.61%. A balanced norm constraint tests whether those failures were caused by the coordinate-pivot parameterization rather than loss of necessary capacity.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 9))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with coefficient anchors and one balanced Q/K norm gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                self.weight_rest[12:27],
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:48],
                fixed,
                self.weight_rest[48:],
            )
        ).view(self.out_features, self.in_features)
=======
    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        fixed = self.weight_rest.new_full((1,), 0.02)

        # Use query row 3's norm for matching key row 11. Seven
        # stereographic coordinates describe the key direction.
        q3 = self.weight_rest[19:27]
        stereo = self.weight_rest[79:86]
        stereo_sq = stereo.square().sum().unsqueeze(0)
        key3_unit = torch.cat((2.0 * stereo, one - stereo_sq))
        key3_unit = key3_unit / (one + stereo_sq)
        key3 = key3_unit * q3.norm()

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
                self.weight_rest[12:27],
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:48],
                fixed,
                self.weight_rest[48:79],
                key3,
                self.weight_rest[86:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Fix two nonzero query coefficients using independent Q/K
                # reciprocal scales while preserving the attention scores.
=======
                # Fix seven nonzero query coefficients using independent Q/K
                # reciprocal scales while preserving the attention scores.
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
                        )
                    )
                )
=======
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Balance the remaining head-0 Q/K coordinate without choosing
                # a fragile coefficient pivot. A simultaneous sign choice puts
                # the key direction in the stable stereographic hemisphere.
                q_norm = weight[3].norm()
                k_norm = weight[11].norm()
                scale = torch.sqrt(k_norm / q_norm)
                if weight[11, -1].item() < 0.0:
                    scale = -scale
                weight[3].mul_(scale)
                weight[11].div_(scale)

                key_unit = weight[11] / weight[11].norm()
                stereo = key_unit[:7] / (1.0 + key_unit[7])

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:88],
                            stereo,
                            flat[96:],
                        )
                    )
                )
>>>>>>> REPLACE