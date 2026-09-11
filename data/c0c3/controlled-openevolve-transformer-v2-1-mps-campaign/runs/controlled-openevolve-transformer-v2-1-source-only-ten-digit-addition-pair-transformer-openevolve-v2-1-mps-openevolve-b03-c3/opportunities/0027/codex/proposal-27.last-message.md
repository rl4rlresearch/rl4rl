MECHANISM: One-coordinate value-bias absorption

HYPOTHESIS: Fixing one value-bias coordinate at zero will produce a 1,625-parameter model with at least 99% accuracy, because attention weights sum to one and the resulting constant value contribution can be represented by the retained output-projection bias.

INTENDED_EDIT: Store seven learned value-bias coordinates, reconstruct the eighth as zero during attention, and leave the qualified dual ambient-Adam gauges unchanged.

EVIDENCE: The current 1,626-parameter design achieved up to 100% accuracy. Unlike failed 1,625-parameter gauges that perturbed sensitive positional, embedding, or residual-output optimization, this removes one coordinate from a value bias that is functionally absorbed by the preserved projection bias; the attention-output-bias ablation cautions against removing that projection bias itself.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        full_v_bias = torch.cat((self.v_bias, self.v_bias.new_zeros(1)))
        v = v + full_v_bias
>>>>>>> REPLACE