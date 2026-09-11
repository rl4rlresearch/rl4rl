MECHANISM: MLP output-bias pair tying with proven positional capacity preserved

HYPOTHESIS: A 1,271-parameter model will retain at least 99% accuracy because it preserves the successful 1,272-parameter model’s fourteen shared far-distance biases while removing one orthogonal parameter by tying a single pair of MLP output-bias coordinates.

INTENDED_EDIT: Adopt the verified fourteen-bin attention-bias sharing design and replace the MLP output projection with the existing one-pair-tied mean-zero projection, reducing parameters from 1,275 to 1,271.

EVIDENCE: The fourteen-bin 1,272-parameter design achieved 99.3%, while sharing the fifteenth positional bin caused the 1,271-parameter model to fall to 94.79%; this motivates retaining all load-bearing positional distinctions and testing the next reduction in the MLP instead.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(11))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 11),
=======
                self.far_rel_bias.expand(self.n_head, 14),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = MeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = PairTiedBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE