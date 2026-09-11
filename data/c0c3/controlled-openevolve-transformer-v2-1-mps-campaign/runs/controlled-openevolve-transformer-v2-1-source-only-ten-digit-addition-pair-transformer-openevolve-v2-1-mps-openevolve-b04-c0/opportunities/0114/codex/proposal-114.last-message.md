MECHANISM: Tied value/output bias gauge canonicalization

HYPOTHESIS: Constraining the tied value/projection bias so its combined post-projection contribution is mean-zero will produce a 1569-parameter model with at least 99% accuracy, because it removes the exact uniform residual-shift gauge while preserving the original parameter’s value-bias and projection-bias coupling.

INTENDED_EDIT: Replace the eight-coordinate tied bias with seven learned coordinates, reconstruct its final coordinate from the projection matrix, and continue using the reconstructed bias both before and after attention projection.

EVIDENCE: The current tied-bias design achieved 99.46% at 1570 parameters, while directly replacing it with a mean-zero post-projection bias collapsed to 64.92%; retaining the original coupled computational path tests whether that failed reduction was caused by its altered optimization geometry rather than by loss of functional capacity.

<<<<<<< SEARCH
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = self.v_bias
        self.attn_drop = nn.Dropout(dropout)
=======
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias
        v = v + self.v_bias
=======
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias

        # The tied bias contributes (proj.weight + I) @ v_bias after
        # projection. Reconstruct its last coordinate so that contribution
        # is mean-zero; the removed uniform component is erased by subsequent
        # LayerNorms.
        output_sum_coeff = self.proj.weight.sum(dim=0) + 1.0
        last_bias = -(
            output_sum_coeff[:-1] * self.v_bias_rest
        ).sum() / output_sum_coeff[-1]
        v_bias = torch.cat((self.v_bias_rest, last_bias.view(1)))
        v = v + v_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = F.linear(y, self.proj.weight, v_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE