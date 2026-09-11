MECHANISM: Two-coordinate cross-head query-bias sharing with lighter batches

HYPOTHESIS: Sharing a second corresponding query-bias coordinate will retain at least 99% accuracy with 1,395 parameters; using batch size 384 will preserve 75,000 optimization steps while completing within the verification time limit.

INTENDED_EDIT: Represent eight query-bias coordinates with six learned scalars, reduce the default batch size from 512 to 384, and validate only at the positive final step.

EVIDENCE: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters at 74,999 steps. Both second-coordinate trials timed out at batch size 512, including a 70,000-step run, so reducing per-step computation while retaining the qualified 75,000-update schedule is the closest unresolved test.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads; independent Q/K rotations make that equality a
        # one-dimensional gauge choice while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Two corresponding query coordinates are
        # shared across the first two heads while remaining learned.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:1],
                self.qkv.bias[self.head_dim :],
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (step > 0 and step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--batch-size", type=int, default=512)
=======
    p.add_argument("--batch-size", type=int, default=384)
>>>>>>> REPLACE