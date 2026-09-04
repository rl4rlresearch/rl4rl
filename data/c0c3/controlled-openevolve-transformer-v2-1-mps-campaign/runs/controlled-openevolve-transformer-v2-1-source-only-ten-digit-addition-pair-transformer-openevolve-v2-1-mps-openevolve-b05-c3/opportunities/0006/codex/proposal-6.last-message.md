MECHANISM: Absorb constant value bias into the attention output bias

HYPOTHESIS: Removing the eight value-projection bias parameters in addition to the already qualified key-bias removal will produce 1628 parameters and retain at least 99% accuracy, because softmax-normalized attention makes value bias an input-independent vector that the existing output-projection bias can represent.

INTENDED_EDIT: Preserve the baseline constructor RNG stream, learn only the eight query-bias parameters in fused QKV, and insert fixed zero key and value biases during the forward pass.

EVIDENCE: The initialization-preserving key-bias design achieved 99.86% accuracy with 1636 parameters; this retains its proven architecture and initialization while eliminating another analytically redundant eight-parameter bias.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Key bias is softmax-invariant,
        # while value bias is a constant absorbed by the output bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        bias = torch.cat((self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model)))
        qkv = F.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE