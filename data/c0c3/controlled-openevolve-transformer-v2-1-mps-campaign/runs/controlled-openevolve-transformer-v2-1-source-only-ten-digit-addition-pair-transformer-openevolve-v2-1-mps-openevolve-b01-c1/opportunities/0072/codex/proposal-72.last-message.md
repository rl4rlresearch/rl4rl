MECHANISM: Attention value/output affine-bias quotient

HYPOTHESIS: Removing all eight value-projection bias parameters will reduce the model from 1,547 to 1,539 parameters while retaining at least 99% accuracy, because causal attention weights sum to one, making the value bias an input-independent offset exactly representable by the retained mean-free output-projection bias.

INTENDED_EDIT: Retain only the learned query bias in `qkv`; reconstruct zero key and value biases during the forward pass while leaving all attention weights and the output-projection bias unchanged.

EVIDENCE: The 1,547-parameter design achieved 99.88% accuracy after completely quotienting the analogous pre-MLP affine bias, while multi-query weight sharing failed at 6.64%; this patch removes only a mathematically redundant affine pathway without sharing load-bearing representation weights.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then retain only query bias. Key
        # bias is softmax-null, and value bias is absorbed by the output bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(full_bias[:d_model].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
=======
        q_bias = self.qkv.bias
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(2 * d_model))
        )
>>>>>>> REPLACE