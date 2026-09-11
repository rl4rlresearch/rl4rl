MECHANISM: Triplet-and-quartet-tied attention output bias

HYPOTHESIS: A 1,059-parameter transformer will maintain at least 99% accuracy because the verified 1,060-parameter `[a,b,b,c,c,c,c]` attention-bias design achieved 99.95%, and merging its first singleton with the adjacent pair preserves the demonstrated quartet boundary.

INTENDED_EDIT: Adopt the verified final quartet and tie the three remaining attention output-bias coordinates, producing `[a,a,a,b,b,b,b]` and reducing the current model by three parameters.

EVIDENCE: The 1,060-parameter pair-tied-singleton design reached 99.95%, while the failed quintet design showed that the coordinate adjoining the final quartet must remain distinct; this patch preserves that boundary and applies the triplet-and-quartet sharing pattern that previously achieved 100% for the MLP bias.

<<<<<<< SEARCH
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.output_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(3)))
=======
        bias = torch.cat(
            (self.output_bias[:1].expand(3), self.output_bias[1:].expand(4))
        )
>>>>>>> REPLACE