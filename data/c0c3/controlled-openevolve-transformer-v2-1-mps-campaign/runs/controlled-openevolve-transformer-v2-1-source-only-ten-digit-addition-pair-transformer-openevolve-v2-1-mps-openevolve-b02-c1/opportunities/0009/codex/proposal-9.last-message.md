MECHANISM: Affine-hyperplane attention coordinate elimination

HYPOTHESIS: Preserving the learned first LayerNorm affine transform while omitting one dependent attention-input coordinate and adding an independent value bias will reduce the model from 1,600 to 1,584 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reduce QKV input width from eight to seven coordinates and replace the shared value/output bias role with a dedicated learned value bias, while retaining the projection’s output bias.

EVIDENCE: Seven-coordinate projection after non-affine LayerNorm failed at 77.79%, but prior results showed that losing the value-bias pathway harms accuracy and that restoring it recovers 99.99%; retaining the LayerNorm affine parameters and explicitly preserving both bias pathways isolates the analytically redundant coordinate without repeating those removals.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model - 1, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.proj.bias
=======
        qkv = self.qkv(x[..., :-1])
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
>>>>>>> REPLACE