MECHANISM: Fourth inverse-compensated query–key shear with final-only validation

HYPOTHESIS: Extending the qualified 1490-parameter gauge with the fourth exact shear will produce 1489 parameters and at least 99% accuracy; evaluating only after training will avoid the prior timeout without changing the training trajectory.

INTENDED_EDIT: Add the third and fourth first-head query–key shears, remove query-row-3 coordinates 2 and 3, reconstruct them as zeros, and perform validation only at the final training step.

EVIDENCE: The three-shear 1490-parameter design achieved 99.92% accuracy, while the fourth-shear attempt failed only because training timed out; the added reduction is the same exact GL symmetry, and removing intermediate autoregressive validation directly addresses runtime.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, four shears, and ten value gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fourth_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[3, :1],
                    fixed_weight[3, 2:],
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
        fixed_weight[3, 1] = 0.0

        value_rotations = []
=======
        fixed_weight[3, 1] = 0.0

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
                        fixed_weight[3, 2:],
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
                self.fourth_weight.new_zeros(1),
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

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE