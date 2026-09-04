MECHANISM: Single-coordinate query-bias gauge fixing with endpoint validation

HYPOTHESIS: Fixing one query-bias coordinate while retaining the learned sinusoidal gain will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps, and endpoint-only validation will let training finish within the time limit.

INTENDED_EDIT: Reconstruct one fixed-zero query-bias coordinate, retain all other learned parameters, and evaluate only at positive interval steps and the final step with a 75,000-step default interval.

EVIDENCE: The qualified 1,397-parameter design achieved 99.69% accuracy at 74,999 steps. Removing six query-bias coordinates collapsed accuracy, but the prior one-coordinate experiment produced no accuracy result because it timed out; testing the smallest gauge reduction with validation overhead reduced to one final evaluation is the closest unresolved experiment.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, value bias is absorbed by the output
        # projection, and one query-bias coordinate per head can be fixed by a
        # joint Q/K rotation. Remove one such coordinate while constructing the
        # original Linear first to preserve the constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
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
                self.qkv.bias,
                self.qkv.bias.new_zeros(1),
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
        if ((step > 0) and (step % train_cfg.eval_interval == 0)) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE