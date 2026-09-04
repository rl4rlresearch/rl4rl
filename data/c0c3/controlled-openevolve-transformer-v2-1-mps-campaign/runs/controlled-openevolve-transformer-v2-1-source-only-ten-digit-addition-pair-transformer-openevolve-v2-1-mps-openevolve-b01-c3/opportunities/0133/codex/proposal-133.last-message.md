MECHANISM: Rank-six vocabulary bottleneck with two-pair secondary-head gain sharing

HYPOTHESIS: The resulting 861-parameter transformer will maintain at least 99% accuracy because the current rank-six, final-pair-tied model achieved 99.99%, while the same `[a,b,c,d,d,e,e]` gain pattern achieved 99.97% with the six-unit MLP before vocabulary compression.

INTENDED_EDIT: Replace six learned secondary-head gains expanded as `[a,b,c,d,e,f,f]` with five gains expanded as `[a,b,c,d,d,e,e]`, removing one parameter while preserving the successful rank-six embedding and six-unit MLP.

EVIDENCE: The 862-parameter rank-six model reached 99.99%; although jumping directly to four gains failed at 78.54%, the five-gain pattern independently retained 99.97%, making this isolated intermediate tie the most informative next compression.

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 1)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain,
                self.secondary_value_gain[..., -1:],
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE