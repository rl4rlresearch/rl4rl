MECHANISM: Shared value/output-bias residual-shift gauge

HYPOTHESIS: Fixing shared `v_bias[0]` at zero will reduce the model to 1565 parameters while retaining at least 99% accuracy, because attention normalization makes the value bias position-independent and downstream LayerNorm removes one feature-uniform output-bias degree of freedom.

INTENDED_EDIT: Store seven learned shared value/projection-bias coordinates and reconstruct the leading coordinate as zero in both uses.

EVIDENCE: The four even-coordinate projection anchors achieved 99.87% accuracy at 1566 parameters, demonstrating tolerance to the same feature-uniform residual gauge; using component zero also follows the successful projection component-zero anchors while avoiding the failed odd-component anchors.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor(), self.bias)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None
        if self.bias is not None:
            bias = torch.cat((self.bias.new_zeros(1), self.bias))
        return F.linear(x, self.weight_tensor(), bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = AttentionWeightAnchoredLinear(d_model, 3 * d_model)
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = ResidualGaugeLinear(d_model, d_model)
        self.proj.bias = self.v_bias
=======
        self.qkv = AttentionWeightAnchoredLinear(d_model, 3 * d_model)
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = ResidualGaugeLinear(d_model, d_model)
        self.proj.bias = self.v_bias_rest
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias
        v = v + self.v_bias
=======
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        v_bias = torch.cat((self.v_bias_rest.new_zeros(1), self.v_bias_rest))
        q = q + q_bias
        v = v + v_bias
>>>>>>> REPLACE