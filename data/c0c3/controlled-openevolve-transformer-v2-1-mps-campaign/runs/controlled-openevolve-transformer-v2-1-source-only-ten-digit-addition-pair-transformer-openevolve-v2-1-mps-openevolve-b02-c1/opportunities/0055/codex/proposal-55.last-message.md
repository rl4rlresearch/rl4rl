MECHANISM: Per-head farthest-distance bias tying

HYPOTHESIS: Tying each head’s two farthest relative-distance biases will reduce the model from 1,046 to 1,044 parameters while retaining at least 99% accuracy, because it preserves independent, otherwise unconstrained positional tables and alters only the two sparsest causal distances.

INTENDED_EDIT: Store two fewer relative-bias coordinates and reconstruct both farthest distances from each head’s existing omitted reference; the quotient optimizer continues to train the resulting per-head virtual reference automatically.

EVIDENCE: Independent affine positional pointers failed at 0% accuracy, showing that aggressive table compression loses essential positional structure, while the current independent-table model achieved 99.95%; a single boundary tie per head is the smallest structured positional ablation that preserves head independence.

<<<<<<< SEARCH
        # One bias per head is a softmax-invisible common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
=======
        # The two farthest distances share each head's softmax-invisible
        # common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        rel_bias = F.pad(self.rel_bias, (0, 1))
=======
        rel_bias = F.pad(self.rel_bias, (0, 2))
>>>>>>> REPLACE