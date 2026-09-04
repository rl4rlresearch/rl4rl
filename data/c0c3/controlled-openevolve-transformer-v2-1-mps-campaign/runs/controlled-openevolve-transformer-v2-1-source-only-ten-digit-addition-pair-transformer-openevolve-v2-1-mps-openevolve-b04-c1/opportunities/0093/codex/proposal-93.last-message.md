MECHANISM: Value-to-output bias quotient

HYPOTHESIS: Removing the three learned value-bias parameters will reduce the model from 875 to 872 parameters while retaining at least 99% accuracy, because normalized attention passes value bias through as a position-independent vector that the existing projection bias can represent modulo a common residual-stream shift annihilated by downstream LayerNorm.

INTENDED_EDIT: Remove `v_bias` and its addition to the projected values, leaving the seven-parameter attention output bias as the nonredundant branch-bias representation.

EVIDENCE: The 875-parameter design retained 99.92% accuracy after exact LayerNorm-induced gauge fixing; this applies another exact functional quotient rather than reducing the demonstrated token, value-projection, or MLP capacity.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 5))

        q = q.view(
=======
        q = q.view(
>>>>>>> REPLACE