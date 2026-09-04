MECHANISM: Two-coordinate cross-head query-bias sharing with lighter batches

HYPOTHESIS: Sharing two corresponding query-bias coordinates will achieve at least 99% accuracy with 1,395 parameters after 75,000 updates, while batch size 256 and positive-step endpoint validation will finish within the verification limit.

INTENDED_EDIT: Represent eight query-bias coordinates with six learned scalars, halve the training batch size, and validate only at the final positive step by default.

EVIDENCE: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters. Two-coordinate attempts yielded no accuracy result because they timed out even at batch size 384, motivating the same incremental constraint with a lighter batch while preserving the 75,000-update schedule.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Share two corresponding query coordinates
        # across heads while keeping every query coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
=======
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:2],
                self.qkv.bias[self.head_dim :],
            )
        )
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if ((step + 1) % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--batch-size", type=int, default=512)
=======
    p.add_argument("--batch-size", type=int, default=256)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE