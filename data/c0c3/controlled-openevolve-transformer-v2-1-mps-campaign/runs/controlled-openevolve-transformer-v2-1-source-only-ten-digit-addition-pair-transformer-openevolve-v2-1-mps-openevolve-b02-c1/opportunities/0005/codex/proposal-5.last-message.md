MECHANISM: Shared value/output-projection bias

HYPOTHESIS: Reusing the learned output-projection bias as the attention value bias will reduce parameters from 1,636 to 1,628 while reaching at least 99% accuracy, because it restores the value-bias optimization path without adding an independently learned vector.

INTENDED_EDIT: Replace the independent query/value bias matrix with a query-only bias and use the projection layer’s existing bias as the value bias.

EVIDENCE: Removing the value bias entirely reached 98.99%, only one correct test example below the requirement, while retaining it reached 99.39%; sharing it with the representationally redundant projection bias preserves both bias pathways at the smaller parameter count.

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
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qv_bias.unbind(0)
        q = q + q_bias
        v = v + v_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.proj.bias
>>>>>>> REPLACE