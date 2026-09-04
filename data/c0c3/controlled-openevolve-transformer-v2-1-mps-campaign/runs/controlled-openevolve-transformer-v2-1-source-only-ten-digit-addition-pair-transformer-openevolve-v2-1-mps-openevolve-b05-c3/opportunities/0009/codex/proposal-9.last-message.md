MECHANISM: Single-coordinate query-bias ablation

HYPOTHESIS: Retaining seven learned query-bias coordinates while fixing one to zero will reduce the qualified 1628-parameter design to 1627 parameters and preserve at least 99% accuracy, because LayerNorm bias and the remaining attention biases provide compensating flexibility without altering the proven weight initialization stream.

INTENDED_EDIT: Remove key and value biases as in the qualified design, then learn only seven of the eight query-bias coordinates while fixing the final coordinate to zero.

EVIDENCE: The query-only QKV-bias design achieved 99.92% accuracy at 1628 parameters; unlike the failed broad LayerNorm and gauge reparameterizations, this tests the smallest possible localized reduction while leaving every weight tensor and seven query-bias coordinates unchanged.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct the baseline layer first to preserve the proven constructor
        # RNG stream. Key and value biases are omitted, along with one query
        # coordinate, while all weight tensors retain their baseline shapes.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        # The learned entries occupy the first seven query coordinates. The
        # final query coordinate and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 1))
        )
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE