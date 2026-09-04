MECHANISM: Fourth inverse-compensated query–key shear with lower-overhead training

HYPOTHESIS: Completing the first-head GL gauge will produce a 1489-parameter model with at least 99% accuracy, while bulk holdout checks and final-only validation will let training finish within the verification limit.

INTENDED_EDIT: Add the three remaining first-head query–key shears, reconstruct query row 3 with coordinates 1–3 fixed to zero, optimize strict holdout rejection without changing sampled batches, and perform one larger-batch validation at the final step.

EVIDENCE: The three-shear 1490-parameter reference achieved 99.92% accuracy; both fourth-shear attempts ended only in timeouts, so preserving the exact symmetry while reducing Python sampling and validation overhead directly targets the observed blocker.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, one shear, and ten value gauges fixed."""
=======
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, four shears, and ten value gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
=======
        self.fourth_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[3, :1],
                    fixed_weight[3, 4:],
                )
            )
        )
        self.head_two_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight[2, 2] = 0.0

        value_rotations = []
=======
        fixed_weight[2, 2] = 0.0

        second_shear = -fixed_weight[3, 1] / fixed_weight[2, 1]
        fixed_weight[3] = (
            fixed_weight[3] + second_shear * fixed_weight[2]
        )
        fixed_weight[key_start + 2] = (
            fixed_weight[key_start + 2]
            - second_shear * fixed_weight[key_start + 3]
        )
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
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
=======
            self.fourth_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[3, :1],
                        fixed_weight[3, 4:],
                    )
                )
            )
            self.head_two_weight.copy_(
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_row = F.pad(self.head_two_weight, (3, 0))
=======
        fourth_row = torch.cat(
            (
                self.fourth_weight[:1],
                self.fourth_weight.new_zeros(3),
                self.fourth_weight[1:],
            )
        )
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
=======
                third_row.unsqueeze(0),
                fourth_row.unsqueeze(0),
                head_two_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
=======
        # Strictly avoid holdout leakage, using bulk conversion for the
        # overwhelmingly common collision-free path.
        a_values = a.tolist()
        b_values = b.tolist()
        for i, (ai, bi) in enumerate(zip(a_values, b_values)):
            current_hash = pair_hash(ai, bi)
            if current_hash not in self.reserved_hashes:
                continue
            while current_hash in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                current_hash = pair_hash(ai, bi)
            a[i] = ai
            b[i] = bi
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-batch-size", type=int, default=512)
=======
    p.add_argument("--eval-batch-size", type=int, default=2000)
>>>>>>> REPLACE