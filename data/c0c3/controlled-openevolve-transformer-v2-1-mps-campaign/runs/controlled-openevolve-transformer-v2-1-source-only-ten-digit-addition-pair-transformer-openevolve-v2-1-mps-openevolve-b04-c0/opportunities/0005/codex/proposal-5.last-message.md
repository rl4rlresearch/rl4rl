MECHANISM: Shared value/output bias parameterization

HYPOTHESIS: Tying the attention value bias to the output-projection bias will reduce parameters from 1636 to 1628 while retaining at least 99% accuracy, because it preserves both bias pathways whose isolated removal failed while eliminating their redundant independent parameterization.

INTENDED_EDIT: Make the projection bias and value bias reference the same learned 8-element parameter, leaving all computations, widths, training, and decoding unchanged.

EVIDENCE: Removing the value bias alone reduced accuracy to 18.29%, indicating that its optimization pathway matters despite functional redundancy with the projection bias; sharing the biases preserves that pathway and the direct projection-bias pathway while deduplicating eight parameters.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = self.v_bias
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE