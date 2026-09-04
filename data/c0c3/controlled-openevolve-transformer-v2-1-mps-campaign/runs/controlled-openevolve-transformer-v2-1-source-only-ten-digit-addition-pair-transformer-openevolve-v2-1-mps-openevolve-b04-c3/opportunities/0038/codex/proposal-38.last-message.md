MECHANISM: Two-coordinate cross-head query-bias sharing with shortened endpoint-only training

HYPOTHESIS: Sharing a second corresponding query-bias coordinate will retain at least 99% accuracy with 1,395 parameters after 45,000 steps, a completed-duration schedule supported by the 99.92%-accurate 45,000-step reference.

INTENDED_EDIT: Encode eight query-bias coordinates with six learned scalars, train for 45,000 steps, and validate only at the final positive step.

EVIDENCE: One-coordinate sharing achieved 99.97% accuracy with 1,396 parameters, while all two-coordinate runs at 60,000 or more steps timed out rather than yielding negative accuracy evidence; a separate 45,000-step run completed, and Reference Design 3 reached 99.92% at that duration.

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
        # shared across the first two heads, keeping every coordinate adaptive.
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
        if ((step + 1) % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=75000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=75000)
=======
    p.add_argument("--eval-interval", type=int, default=45000)
>>>>>>> REPLACE