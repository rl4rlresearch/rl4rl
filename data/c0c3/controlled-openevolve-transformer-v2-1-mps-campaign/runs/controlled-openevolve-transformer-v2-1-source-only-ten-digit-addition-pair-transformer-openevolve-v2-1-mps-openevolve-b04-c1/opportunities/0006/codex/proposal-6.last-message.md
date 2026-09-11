MECHANISM: Absorb value-projection bias into attention output bias

HYPOTHESIS: Removing the 8-element value bias will reduce parameters from 1620 to 1612 while retaining at least 99% accuracy, because attention weights sum to one, making the value bias a position-independent offset exactly absorbable by `proj.bias`.

INTENDED_EDIT: Delete `v_bias` and its addition in the attention forward pass while preserving the query bias and all other architecture and training settings.

EVIDENCE: The 1620-parameter design achieved 99.98% accuracy after targeted removal of mathematically absorbable biases, whereas broad linear-bias removal failed; this motivates another exact, narrowly scoped bias reparameterization.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE