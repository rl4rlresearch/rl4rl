MECHANISM: Pre-attention LayerNorm bias quotient

HYPOTHESIS: Reducing `ln1.bias` from seven to six orthonormal coordinates will produce a 1,564-parameter model with at least 99% accuracy because the removed constant shift is absorbable by the query and projection biases, while its key shift cancels in the attention softmax.

INTENDED_EDIT: Preserve the qualified four-coordinate `ln2.bias` design and remove one high-index `ln1.bias` direction; the existing full-coordinate AdamW machinery automatically handles the resized basis.

EVIDENCE: The current 1,565-parameter model achieved 99.83%, while reducing `ln2.bias` again fell to 98.62%; this tests an independent downstream-absorbable LayerNorm bias redundancy without repeating that failed optimization constraint.

<<<<<<< SEARCH
        # A uniform pre-attention shift is absorbable by the query and
        # output-projection biases; key shifts cancel from attention softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
=======
        # Constant pre-attention shifts are absorbable by the query and
        # output-projection biases; their key shifts cancel from the softmax.
        # Retain six coordinates while quotienting one additional direction.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
>>>>>>> REPLACE