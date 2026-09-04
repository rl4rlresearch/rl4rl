MECHANISM: Algebraically redundant value-bias removal

HYPOTHESIS: Removing only the value-projection bias will retain at least 99% accuracy while reducing parameters from 1,644 to 1,636, because softmax attention weights sum to one and the value bias can therefore be represented exactly by the retained attention output bias.

INTENDED_EDIT: Replace the combined QKV bias with learned query and key biases only; retain every other model dimension, bias, positional embedding, and training setting.

EVIDENCE: Removing all projection biases collapsed accuracy to 14.88%, so most biases should remain; unlike those broader removals, the value bias is mathematically redundant with the attention output bias and does not reduce the model’s representable function class.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qk_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, k_bias = self.qk_bias.chunk(2)
        q = q + q_bias
        k = k + k_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE