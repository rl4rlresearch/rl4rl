MECHANISM: Softmax-normalized value-bias absorption

HYPOTHESIS: Removing the shared attention value bias will reduce parameters from 1,636 to 1,628 while retaining at least 99% accuracy, because attention weights sum to one and the existing output-projection bias can represent the value bias’s constant contribution exactly.

INTENDED_EDIT: Replace the joint query/value bias with a query-only bias and remove value-bias addition from attention.

EVIDENCE: Removing the analytically inactive key bias retained 99.39% accuracy, while compressing positional or token representations failed; this motivates another function-class-preserving bias elimination instead of reducing learned representation capacity.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qv_bias.unbind(0)
        q = q + q_bias
        v = v + v_bias
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
>>>>>>> REPLACE