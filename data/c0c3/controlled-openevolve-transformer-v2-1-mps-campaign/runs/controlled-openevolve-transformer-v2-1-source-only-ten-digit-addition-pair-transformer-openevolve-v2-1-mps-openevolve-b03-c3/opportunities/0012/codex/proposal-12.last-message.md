MECHANISM: Query-bias absorption through pre-attention normalization

HYPOTHESIS: Removing the query bias from the verified 1,628-parameter design will retain at least 99% accuracy with 1,620 parameters, because the retained `ln1` bias can supply a query offset while induced key shifts cancel under softmax and induced value shifts can be compensated by the retained value bias.

INTENDED_EDIT: Remove key and query projection biases, retain the learned value bias and width-12 MLP, and remove the second LayerNorm bias as in the verified reference design.

EVIDENCE: The 1,628-parameter design retaining query/value biases while removing key and `ln2` biases achieved 99.95%; the query-only design’s 16.49% result identifies value bias as critical but does not establish that query bias is necessary.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        v = v + self.v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE