MECHANISM: LayerNorm-subspace key-weight elimination

HYPOTHESIS: Removing one redundant input weight from a second-head key row will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because bias-free LayerNorm outputs occupy a seven-dimensional subspace and that key row can remain fully expressive using seven coordinates.

INTENDED_EDIT: Replace the joint QKV projection with a full projection for all but one row and a seven-input projection for one second-head key row, while preserving all verified query, value, and output biases.

EVIDENCE: The 1608-parameter model achieved 99.88%, while reducing a load-bearing query-bias coordinate or tying the second head’s key and value maps failed; this patch preserves those capacities and removes only a LayerNorm-subspace-redundant weight.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv_except_one = nn.Linear(d_model, 3 * d_model - 1, bias=False)
        self.k2_reduced_row = nn.Linear(d_model - 1, 1, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_except_one = self.qkv_except_one(x)
        reduced_row = d_model + self.head_dim
        qkv = torch.cat(
            (
                qkv_except_one[..., :reduced_row],
                self.k2_reduced_row(x[..., :-1]),
                qkv_except_one[..., reduced_row:],
            ),
            dim=-1,
        )
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE