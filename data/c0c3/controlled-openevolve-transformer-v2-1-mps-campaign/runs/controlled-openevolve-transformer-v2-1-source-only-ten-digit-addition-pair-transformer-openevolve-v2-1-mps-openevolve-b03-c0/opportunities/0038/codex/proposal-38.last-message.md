MECHANISM: Detached query-to-key bias reuse

HYPOTHESIS: Reconstructing all remaining learned key-bias offsets from a detached query-bias scalar will reduce the model from 1,609 to 1,608 parameters while retaining at least 99% accuracy after 1,000 additional low-rate steps, because key offsets are softmax-invariant and detachment prevents the gradient coupling implicated by the failed direct query/key reuse.

INTENDED_EDIT: Remove the final dedicated key-bias scalar, derive its key offsets from a detached query-bias view, shift value-bias slices to preserve their validated mapping, and train for 8,000 steps on the existing 5,000-step cosine schedule.

EVIDENCE: Detached key-group sharing achieved 99.95% at 1,609 parameters after 6,999 steps, while non-detached query/key reuse reached only 12.53%; this directly motivates testing detached reuse of a query scalar with another 1,000 refinement steps.

<<<<<<< SEARCH
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 11))
=======
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 12))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias[d_model : d_model + 1].detach(),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 7 : 2 * d_model + self.head_dim - 10],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 7 : 2 * d_model - 5],
                self.qkv.bias[2 * d_model + self.head_dim - 10 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 2],
                self.qkv.bias[d_model + 1 :],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=7000)
=======
    p.add_argument("--train-steps", type=int, default=8000)
>>>>>>> REPLACE