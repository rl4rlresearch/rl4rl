MECHANISM: Attention-only residual decoder

HYPOTHESIS: Removing the final one-neuron MLP will reduce the model from 1,384 to 1,360 learned parameters while retaining at least 99% accuracy after 56,000 training steps.

INTENDED_EDIT: Remove the MLP branch from the transformer block and extend training by 4,000 steps.

EVIDENCE: The one-neuron MLP model achieved 99.99% accuracy at 52,000 steps, and every preceding one-neuron width reduction succeeded with a 4,000-step allowance; this motivates testing the structural boundary where attention alone performs the computation.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=52000)
=======
    p.add_argument("--train-steps", type=int, default=56000)
>>>>>>> REPLACE