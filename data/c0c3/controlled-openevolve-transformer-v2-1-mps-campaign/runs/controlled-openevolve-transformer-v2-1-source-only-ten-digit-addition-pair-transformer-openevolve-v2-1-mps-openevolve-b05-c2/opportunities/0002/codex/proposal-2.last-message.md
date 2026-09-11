MECHANISM: Softmax-invariant key-bias elimination

HYPOTHESIS: Removing only the attention key bias will reduce parameters from 1644 to 1636 while preserving at least 99% accuracy because adding the same bias to every key changes each query’s attention logits by a row-wise constant, which softmax cancels exactly.

INTENDED_EDIT: Replace the fused 24-element QKV bias with learned query and value biases while fixing the functionally redundant key bias to zero; preserve the feed-forward width of 12.

EVIDENCE: The 1644-parameter width-12 model reached 99.96%, whereas reducing d_ff to 11 collapsed accuracy to 72.21%; this motivates retaining nonlinear capacity and removing an attention parameter that cannot affect the model’s output.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
=======
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
>>>>>>> REPLACE