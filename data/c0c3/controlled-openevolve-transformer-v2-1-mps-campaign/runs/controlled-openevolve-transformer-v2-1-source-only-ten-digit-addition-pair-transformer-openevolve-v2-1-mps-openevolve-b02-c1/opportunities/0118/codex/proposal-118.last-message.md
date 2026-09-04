MECHANISM: Three-way terminal MLP sharing with single-thread tiny-kernel training

HYPOTHESIS: Sharing the final MLP coefficient across three adjacent residual coordinates will yield a 980-parameter model with at least 99% accuracy, while single-thread execution and final-only validation will let the previously timed-out configuration complete 4,999 updates.

INTENDED_EDIT: Remove one MLP input parameter, preserve initialization sampling, retain the proposed three-way reconstruction, reduce sampler overhead without changing sampled batches, use one CPU intra-op thread, and validate only after the final update by default.

EVIDENCE: Disjoint MLP input pairing previously achieved 99.93%; the three-way extension produced no negative accuracy evidence because verification timed out. Earlier sampling and validation optimizations were insufficient, motivating the additional tiny-kernel threading optimization.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 4, d_ff, bias=False)
        self.fc1._removed_input_features = 3

        # Preserve the constructor RNG stream of the removed input weights.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(3 * d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )
=======
        self.fc1 = nn.Linear(d_model - 5, d_ff, bias=False)
        self.fc1._removed_input_features = 4

        # Preserve the constructor RNG stream of the removed input weights.
        fc1_weight_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(4 * d_ff).uniform_(
            -fc1_weight_bound, fc1_weight_bound
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 3:],
=======
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
                self.fc1.weight[:, 2:3],
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
        # Strictly avoid holdout leakage without per-example tensor access
        # when the initially sampled pair is not reserved.
        a_values = a.tolist()
        b_values = b.tolist()
        for i, (initial_a, initial_b) in enumerate(
            zip(a_values, b_values)
        ):
            ai = initial_a
            bi = initial_b
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            if ai != initial_a or bi != initial_b:
                a[i] = ai
                b[i] = bi
>>>>>>> REPLACE

<<<<<<< SEARCH
def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict:
    device = torch.device(train_cfg.device)
=======
def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> Dict:
    # This model consists of tiny kernels for which intra-op thread
    # coordination costs more than the parallel work it exposes.
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)

    device = torch.device(train_cfg.device)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if ((step + 1) % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=5000)
>>>>>>> REPLACE