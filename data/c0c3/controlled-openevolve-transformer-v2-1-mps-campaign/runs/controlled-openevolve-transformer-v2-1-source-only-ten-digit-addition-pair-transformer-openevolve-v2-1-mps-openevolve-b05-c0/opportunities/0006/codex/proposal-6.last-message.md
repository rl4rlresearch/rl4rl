MECHANISM: Headwise canonical query-bias gauge

HYPOTHESIS: Representing each head’s query bias with one learned scalar will reduce parameters from 1636 to 1630 while retaining at least 99% accuracy, because within-head Q/K rotations can orient any query-bias vector along one coordinate without changing attention logits.

INTENDED_EDIT: Replace the eight-element query bias with two headwise scalars expanded onto the first coordinate of each four-dimensional head.

EVIDENCE: Exact removal of the softmax-redundant key bias retained 99.84% accuracy, whereas removing the attention LayerNorm offset failed; this motivates preserving a learned query offset while removing only its rotationally redundant coordinates.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.v_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(self.q_bias.unsqueeze(-1), (0, self.head_dim - 1)).reshape(-1)
        q = q + q_bias
        v = v + self.v_bias
>>>>>>> REPLACE