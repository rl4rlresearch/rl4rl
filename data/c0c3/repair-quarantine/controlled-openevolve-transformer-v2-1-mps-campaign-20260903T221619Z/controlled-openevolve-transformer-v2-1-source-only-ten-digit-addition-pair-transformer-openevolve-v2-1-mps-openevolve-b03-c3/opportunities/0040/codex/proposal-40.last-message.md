MECHANISM: Two-dimensional learned routing with four-dimensional shared content

HYPOTHESIS: Decoupling query/key width from value width will reduce the verified model from 1,555 to 1,503 parameters while retaining at least 99% accuracy, because two learned routing coordinates can encode fixed positional correspondences while the full four-dimensional value stream preserves digit content.

INTENDED_EDIT: Replace each four-dimensional query/key head with a two-dimensional learned routing head, while retaining two independent query heads, the shared four-dimensional learned value stream, and all qualified residual and MLP pathways.

EVIDENCE: Shared-key/value attention achieved 100% accuracy at 1,555 parameters, showing that routing and content need not use independent per-head representations. This cleanly challenges the remaining assumption that their widths must match; unlike the failed additional gauges and gated MLP, it preserves the qualified optimizer and nonlinear computation.

<<<<<<< SEARCH
        # Keep a distinct learned query for each head, but share one learned
        # key/value representation across the query heads.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        # Use a compact learned space for positional routing while retaining
        # the full head width for the content returned by attention.
        self.qk_dim = max(1, self.head_dim // 2)
        self.q_proj = nn.Linear(
            d_model, self.n_head * self.qk_dim, bias=False
        )
        self.k_proj = nn.Linear(d_model, self.qk_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(
            torch.zeros(self.n_head * self.qk_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
=======
        q = q.view(
            bsz, seqlen, self.n_head, self.qk_dim
        ).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.qk_dim)
>>>>>>> REPLACE