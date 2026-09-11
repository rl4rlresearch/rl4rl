MECHANISM: Optimizer-order-preserving key-bias elimination

HYPOTHESIS: Removing the mathematically redundant eight-element key bias while preserving baseline RNG consumption, parameter registration order, and fused QKV bias addition will reduce parameters from 1,644 to 1,636 and recover at least 99% accuracy.

INTENDED_EDIT: Resize the existing QKV bias parameter to query/value biases only, reconstruct a zero key-bias slice during the forward pass, and retain the original fused linear operation.

EVIDENCE: Initialization-preserving key-bias removal improved accuracy from 64.55% to 94.58%; preserving the original parameter slot and fused QKV computation targets the remaining optimizer-order and numerical differences without restoring redundant parameters.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the original parameter slot and initialization RNG consumption,
        # but store only the query and value biases. A shared key bias cancels
        # from every row of the attention softmax.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE