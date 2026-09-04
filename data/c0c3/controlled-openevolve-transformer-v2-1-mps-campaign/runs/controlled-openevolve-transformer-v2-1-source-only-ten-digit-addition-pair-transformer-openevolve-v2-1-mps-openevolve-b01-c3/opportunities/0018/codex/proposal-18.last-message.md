MECHANISM: Fixed query-bias scale gauge

HYPOTHESIS: Fixing the shared scalar query bias to one will reduce the verified 1,527-parameter model to 1,526 parameters while retaining at least 99% accuracy, because reciprocal rescaling of each head’s unrestricted query and key projections can absorb any nonzero shared bias amplitude without changing attention logits.

INTENDED_EDIT: Remove the learned shared query-bias scalar and reconstruct both heads’ query biases with a fixed unit coordinate.

EVIDENCE: Sharing the two scalar query biases achieved 99.8% accuracy with 1,527 parameters; its successful cross-head scale tying supports taking the remaining nonzero amplitude as a gauge and anchoring it while preserving full query/key width and all other verified capacities.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(1))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
=======
        q_bias = F.pad(
            q.new_ones(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
>>>>>>> REPLACE