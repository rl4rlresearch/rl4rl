MECHANISM: Value-bias/output-bias affine quotient

HYPOTHESIS: Fixing one value-projection bias scalar at zero will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because causal softmax rows sum to one, making every value bias a context-independent offset that the learned attention output-projection bias can exactly absorb.

INTENDED_EDIT: Remove the final value-bias coordinate from `qkv.bias` and restore it as a fixed zero during the forward pass, preserving all weights, query biases, initialization behavior, and successful training settings.

EVIDENCE: The current 1,578-parameter model achieves 99.92% accuracy. Unlike the unsuccessful second `fc1`-bias and `ln1`-scale removals, this quotient lies entirely within two consecutive affine attention operations: the mean-free output projection maps the omitted value offset into its own representable seven-dimensional bias space.

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        # Keep constructor RNG consumption, remove the softmax-null key bias,
        # and absorb one value-bias coordinate into the output-projection bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat(
                (full_bias[:d_model], full_bias[2 * d_model : -1])
            ).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
=======
        q_bias = self.qkv.bias[:d_model]
        v_bias = F.pad(self.qkv.bias[d_model:], (0, 1))
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
>>>>>>> REPLACE