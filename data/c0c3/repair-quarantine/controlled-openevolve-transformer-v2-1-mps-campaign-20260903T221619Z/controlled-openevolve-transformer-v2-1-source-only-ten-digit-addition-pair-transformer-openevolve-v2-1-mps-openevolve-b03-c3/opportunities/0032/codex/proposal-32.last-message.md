MECHANISM: Absorb shared value bias into attention output bias

HYPOTHESIS: Removing the four-dimensional shared value bias will reduce the verified 1,554-parameter model to 1,550 parameters while retaining at least 99% accuracy, because softmax-normalized attention propagates this position-independent offset unchanged and the learned output-projection bias already represents the resulting constant residual shift.

INTENDED_EDIT: Delete the redundant shared value-bias parameter and compute values solely with the learned shared value projection, preserving both learned query heads, shared key/value attention, and the verified four-column terminal gauge.

EVIDENCE: Shared key/value attention achieved 100% at 1,555 parameters and remained at 99.98% when combined with the four-column terminal gauge at 1,554; this isolates a functional redundancy inside that qualified attention design without removing the query bias previously shown to be load-bearing.

<<<<<<< SEARCH
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x) + self.v_bias
=======
        q = self.q_proj(x) + self.q_bias
        k = self.k_proj(x)
        v = self.v_proj(x)
>>>>>>> REPLACE