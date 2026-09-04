MECHANISM: Symmetric aggregation of head-routed values

HYPOTHESIS: Summing the two independently routed attention contexts before one shared learned output projection will reduce parameters from 1,107 to 1,079 while maintaining at least 99% accuracy, because the heads already share their value representation and decimal addition is symmetric in its operands.

INTENDED_EDIT: Replace the separate learned output map for each attention head with one shared zero-mean map applied to their variance-preserving summed context, while preserving independent query/key routing and all relative-distance biases.

EVIDENCE: The 1,107-parameter design achieved 99.91% accuracy while using a single shared value readout across both heads, showing that operand routes need not encode digits differently. The failed direct bilinear scorer and shared-key result indicate that head-specific routing is load-bearing, so this patch preserves both heads’ query/key systems and changes only the downstream interpretation of their already-shared semantic values.

<<<<<<< SEARCH
    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )
=======
    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            # The heads retrieve values through the same learned semantic map.
            # Addition is symmetric in those retrieved operands, so initialize
            # one shared output map from the variance-preserving combination of
            # the original head-specific maps.
            centered = centered.view(
                weight.size(0), self.n_head, self.head_dim
            ).sum(dim=1) / math.sqrt(self.n_head)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_weight = self.proj_basis @ self.proj.weight
=======
        y = att @ v
        # Keep independent learned routes, but combine their shared value
        # representation commutatively before the learned output projection.
        y = y.sum(dim=1) / math.sqrt(self.n_head)
        proj_weight = self.proj_basis @ self.proj.weight
>>>>>>> REPLACE