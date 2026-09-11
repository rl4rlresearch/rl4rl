MECHANISM: Ten-feature nonlinear carry bottleneck

HYPOTHESIS: A 10-unit GELU MLP on the qualified two-coordinate final-bias reduction will produce a 1,017-parameter transformer with at least 99% accuracy, because the task may require fewer nonlinear features than the universally retained 11-unit width while preserving both independent attention heads and the five-dimensional token representation.

INTENDED_EDIT: Start from the qualified 1,032-parameter output-bias design and reduce `d_ff` from 11 to 10, removing 15 additional parameters by narrowing both learned MLP projections.

EVIDENCE: The 1,032-parameter final-bias design achieved 99.92%, while the 982-parameter shared-key/value model failed at 83.68% and the four-coordinate lexical bottleneck also failed; this identifies attention-head specialization and token representation as load-bearing, but leaves the shared assumption that 11 independent GELU features are necessary untested.

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        x = x + self.final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
=======
        x = self.ln_f(x)
        final_bias = F.pad(self.final_bias, (0, 2))
        x = x + final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE