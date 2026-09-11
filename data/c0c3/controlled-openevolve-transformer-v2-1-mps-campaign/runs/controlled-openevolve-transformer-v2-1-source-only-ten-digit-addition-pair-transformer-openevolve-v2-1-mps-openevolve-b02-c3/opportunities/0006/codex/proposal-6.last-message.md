MECHANISM: Head-shared learned query offset

HYPOTHESIS: Sharing the proven query bias across both attention heads will reduce parameters from 1628 to 1624 while retaining at least 99% accuracy, because both heads process the same digit-addition structure and still receive a learned four-dimensional query offset.

INTENDED_EDIT: Remove the combined QKV bias and introduce one head-dimension query bias broadcast across both attention heads.

EVIDENCE: Query-only bias achieved 100% accuracy with 1628 parameters, whereas removing it entirely fell to 77.42%; this motivates preserving its function while testing cross-head parameter sharing.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # Both heads share a learned query offset.
        self.q_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.q_bias
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE