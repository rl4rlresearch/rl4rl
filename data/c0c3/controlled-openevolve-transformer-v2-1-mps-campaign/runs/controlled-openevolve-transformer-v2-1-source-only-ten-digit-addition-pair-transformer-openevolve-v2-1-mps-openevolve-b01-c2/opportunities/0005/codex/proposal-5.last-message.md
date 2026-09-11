MECHANISM: Longer optimization for initialization-preserving key-bias removal

HYPOTHESIS: The 1,636-parameter fused-QKV model will exceed 99% accuracy when trained for 7,500 steps, because its prior 5,000-step result reached 97.79% and suggests insufficient convergence rather than a severe capacity loss.

INTENDED_EDIT: Remove the eight softmax-invariant key-bias parameters while preserving the fused QKV weight layout, and extend the default cosine-scheduled training run from 5,000 to 7,500 steps.

EVIDENCE: Initialization-preserving key-bias removal achieved 97.79% with 1,636 parameters, substantially outperforming the separate-bias version at 86.2% and the narrower 1,627-parameter MLP at 72.89%; additional optimization is therefore the most targeted next test.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qv_bias.chunk(2)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=5000)
=======
    p.add_argument("--train-steps", type=int, default=7500)
>>>>>>> REPLACE