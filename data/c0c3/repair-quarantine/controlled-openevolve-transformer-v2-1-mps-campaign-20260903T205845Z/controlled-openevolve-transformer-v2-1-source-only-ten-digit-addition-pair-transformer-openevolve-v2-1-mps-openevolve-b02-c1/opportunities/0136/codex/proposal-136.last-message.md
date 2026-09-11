MECHANISM: Extended far-distance relative-bias sharing

HYPOTHESIS: Tying head 0’s three farthest relative distances will reduce the verified 984-parameter model to 983 parameters while retaining at least 99% accuracy, because it extends an existing far-distance tie without imposing the final-LayerNorm coupling that previously caused accuracy to collapse.

INTENDED_EDIT: Remove one head-0 relative-bias parameter and reconstruct the three farthest distances with the same fixed reference value.

EVIDENCE: The 984-parameter design reached 99.75% accuracy, while tying final-LayerNorm coordinates zero and one fell to 34.94%; this motivates testing an orthogonal reduction in the already-compressed relative-position bias instead of further final-normalization sharing.

<<<<<<< SEARCH
                nn.Parameter(torch.zeros(max_seq_len - 2)),
=======
                nn.Parameter(torch.zeros(max_seq_len - 3)),
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 2)),
=======
                F.pad(self.rel_bias[0], (0, 3)),
>>>>>>> REPLACE