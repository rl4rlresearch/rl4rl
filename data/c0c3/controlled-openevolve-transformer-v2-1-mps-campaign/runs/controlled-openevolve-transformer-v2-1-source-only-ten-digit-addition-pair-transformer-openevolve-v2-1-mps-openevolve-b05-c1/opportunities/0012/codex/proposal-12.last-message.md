MECHANISM: Single-head query-basis gauge anchoring

HYPOTHESIS: Fixing one query-bias coordinate at zero will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because an orthogonal rotation of that head’s query and key coordinates can place any query-bias vector in the resulting three-dimensional subspace without changing attention logits.

INTENDED_EDIT: Learn seven query-bias coordinates and pad the final coordinate with zero during attention; leave initialization, widths, training, and decoding unchanged.

EVIDENCE: The 1,596-parameter design achieved 99.99% accuracy. Although anchoring one MLP-output bias coordinate failed, this tests a distinct exact symmetry internal to one attention head and preserves the current zero initialization with only a one-scalar reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE