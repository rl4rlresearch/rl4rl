MECHANISM: Complete value/output-bias quotient

HYPOTHESIS: Removing the final learned value-bias coordinate will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy, because position-independent value bias is functionally absorbable into the attention output bias.

INTENDED_EDIT: Remove the remaining scalar `v_bias` parameter and omit value-bias addition in attention.

EVIDENCE: Successively fixing seven of eight value-bias coordinates reached 99.92% accuracy at 1622 parameters; despite index 5 being the historically weakest candidate, the later successes of previously failing indices 3 and 4 make completing this proven quotient mechanism more informative than retrying the MLP-bias or first-head key gauges that collapsed.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.proj = nn.Linear(d_model, d_model, bias=False)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v_bias = torch.cat(
            (
                self.v_bias.new_zeros(5),
                self.v_bias,
                self.v_bias.new_zeros(2),
            )
        )
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE