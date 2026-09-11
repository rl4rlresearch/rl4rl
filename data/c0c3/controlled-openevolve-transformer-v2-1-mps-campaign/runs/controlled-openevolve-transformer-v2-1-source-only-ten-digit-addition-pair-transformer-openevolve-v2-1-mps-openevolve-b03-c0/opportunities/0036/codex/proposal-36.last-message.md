MECHANISM: Near-threshold key-bias merge with low-rate refinement

HYPOTHESIS: Tying the second head’s remaining key-bias coordinate to the validated coordinate-1 group and extending training with 1,000 minimum-learning-rate steps will reduce the model to 1,610 parameters while raising the prior 98.92% result above 99%.

INTENDED_EDIT: Store 14 QKV bias parameters, reconstruct the second head’s final key bias from the shared coordinate-1 scalar, preserve the existing value-bias mapping, and train for 6,000 steps while retaining the original 5,000-step cosine schedule.

EVIDENCE: This exact key-bias tie reached 98.92% at 5,000 steps—only eight test examples below the threshold—whereas other 1,610-parameter ties achieved 68.48% or less; continued low-rate optimization is therefore the most targeted next test.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 9))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model + self.head_dim - 2 : d_model + self.head_dim - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 5 : 2 * d_model + self.head_dim - 8],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 5 : 2 * d_model - 3],
                self.qkv.bias[2 * d_model + self.head_dim - 8 :],
=======
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 6 : 2 * d_model + self.head_dim - 9],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 6 : 2 * d_model - 4],
                self.qkv.bias[2 * d_model + self.head_dim - 9 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
    for step in range(train_cfg.train_steps):
        model.train()
=======
    schedule_steps = min(train_cfg.train_steps, 5000)

    for step in range(train_cfg.train_steps):
        model.train()
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
=======
        lr_now = cosine_lr(step, schedule_steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (
            (step % train_cfg.eval_interval == 0)
            or (step == schedule_steps - 1)
            or (step == train_cfg.train_steps - 1)
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=6000)
>>>>>>> REPLACE