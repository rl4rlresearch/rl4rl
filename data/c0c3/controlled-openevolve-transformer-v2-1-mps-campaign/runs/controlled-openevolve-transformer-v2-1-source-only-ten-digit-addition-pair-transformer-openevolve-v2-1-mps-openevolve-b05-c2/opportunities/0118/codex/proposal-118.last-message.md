MECHANISM: Fourth inverse-compensated first-head query–key shear

HYPOTHESIS: Extending the qualified 1490-parameter design with a shear that fixes first-head query row 3, input coordinate 3 will yield 1489 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Remove query coordinate `(row 3, input 3)` by adding a multiple of row 0, apply the exact inverse-transpose compensation to key row 0, and reconstruct the removed coordinate as zero.

EVIDENCE: The previous three successive first-head query–key shears preserved qualification, with the current 1490-parameter design reaching 99.92%; row 0 is already zero in coordinates 0–2, so this fourth shear fixes coordinate 3 without disturbing any earlier query anchors.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, three shears, and ten value gauges."""
=======
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, four shears, and ten value gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fourth_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[3, :1],
                    fixed_weight[3, 3:],
                )
            )
        )
=======
        self.fourth_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[3, :1],
                    fixed_weight[3, 4:],
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_shear = -fixed_weight[3, 2] / fixed_weight[1, 2]
        fixed_weight[3] = (
            fixed_weight[3] + third_shear * fixed_weight[1]
        )
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - third_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 2] = 0.0

        value_rotations = []
=======
        third_shear = -fixed_weight[3, 2] / fixed_weight[1, 2]
        fixed_weight[3] = (
            fixed_weight[3] + third_shear * fixed_weight[1]
        )
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - third_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 2] = 0.0

        fourth_shear = -fixed_weight[3, 3] / fixed_weight[0, 3]
        fixed_weight[3] = (
            fixed_weight[3] + fourth_shear * fixed_weight[0]
        )
        fixed_weight[key_start] = (
            fixed_weight[key_start]
            - fourth_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 3] = 0.0

        value_rotations = []
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.fourth_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[3, :1],
                        fixed_weight[3, 3:],
                    )
                )
            )
=======
            self.fourth_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[3, :1],
                        fixed_weight[3, 4:],
                    )
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fourth_row = torch.cat(
            (
                self.fourth_weight[:1],
                self.fourth_weight.new_zeros(2),
                self.fourth_weight[1:],
            )
        )
=======
        fourth_row = torch.cat(
            (
                self.fourth_weight[:1],
                self.fourth_weight.new_zeros(3),
                self.fourth_weight[1:],
            )
        )
>>>>>>> REPLACE