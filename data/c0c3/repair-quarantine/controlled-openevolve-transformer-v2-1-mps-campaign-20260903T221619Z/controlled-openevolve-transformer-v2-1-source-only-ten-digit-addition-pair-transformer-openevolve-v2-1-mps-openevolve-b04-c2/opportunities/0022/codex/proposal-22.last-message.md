MECHANISM: Value-bias/output-bias reparameterization

HYPOTHESIS: Removing the eight-dimensional attention value bias will reduce the model from 1,629 to 1,621 parameters while retaining at least 99% accuracy, because a position-independent value bias passes unchanged through normalized attention and is exactly absorbable by the retained output-projection bias.

INTENDED_EDIT: Store only the learned query bias in `qkv.bias`; reconstruct zero key and value biases in the attention forward pass while leaving the full output-projection bias trainable.

EVIDENCE: The current 1,629-parameter design achieved 99.81%, whereas further positional compression failed at 96.93%; the prior output-bias quotient also failed, motivating a structurally redundant reduction that preserves every coordinate of the sensitive output bias.

<<<<<<< SEARCH
        # A shared key bias cancels from every attention-softmax row. Retain
        # the original parameter slot while storing only query/value biases.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
=======
        # A shared key bias cancels from attention softmax. A shared value
        # bias passes unchanged through attention and is absorbable by the
        # retained output-projection bias, so store only the query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
>>>>>>> REPLACE