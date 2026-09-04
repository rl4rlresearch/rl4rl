MECHANISM: Softmax-invariant key-bias elimination

HYPOTHESIS: Removing only the mathematically redundant key-projection bias will preserve at least 99% accuracy while reducing parameters from 1,644 to 1,636.

INTENDED_EDIT: Make the packed QKV projection bias-free, then restore learned query and value biases as separate parameters while omitting the key bias.

EVIDENCE: Reducing MLP width to 11 collapsed accuracy to 74.93%, motivating preservation of effective capacity. A constant key bias adds the same value to every unmasked attention logit in each query row and therefore cancels under softmax.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
>>>>>>> REPLACE