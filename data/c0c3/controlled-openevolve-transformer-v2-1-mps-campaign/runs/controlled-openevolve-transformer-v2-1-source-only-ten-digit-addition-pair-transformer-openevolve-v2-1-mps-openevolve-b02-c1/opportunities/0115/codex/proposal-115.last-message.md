MECHANISM: Six-distance head-0 bias sharing with lower-overhead sampling and final-only validation

HYPOTHESIS: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy; preserving the exact training samples while removing per-example tensor accesses and all pre-final validation will allow the previously timed-out configuration to complete 4,999 updates.

INTENDED_EDIT: Remove one head-0 relative-bias parameter, reconstruct six terminal distances with the zero reference, optimize holdout checking without changing sampled batches, and validate only at the final step by default.

EVIDENCE: Five-distance sharing achieved 99.89% at 981 parameters, while all six-distance attempts ended only in timeouts; the current sampler performs millions of unnecessary scalar tensor reads and writes, providing a concrete runtime confound to remove.

<<<<<<< SEARCH
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the six farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 6)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 5)),
=======
                F.pad(self.rel_bias[0], (0, 6)),
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
        return a, b
=======
        # Strictly avoid holdout leakage. Converting each vector once avoids
        # per-example tensor reads and writes while preserving the RNG stream
        # and replacing exactly the same rejected pairs.
        for i, (ai, bi) in enumerate(zip(a.tolist(), b.tolist())):
            original_ai, original_bi = ai, bi
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            if ai != original_ai:
                a[i] = ai
            if bi != original_bi:
                b[i] = bi
        return a, b
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (step == train_cfg.train_steps - 1) or (
            step > 0 and step % train_cfg.eval_interval == 0
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=5000)
>>>>>>> REPLACE