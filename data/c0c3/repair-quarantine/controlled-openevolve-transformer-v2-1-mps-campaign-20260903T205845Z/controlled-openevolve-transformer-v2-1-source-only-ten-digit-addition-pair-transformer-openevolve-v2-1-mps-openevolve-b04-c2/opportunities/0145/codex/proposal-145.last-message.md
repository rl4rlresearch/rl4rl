MECHANISM: LayerNorm scale absorption into unrestricted MLP input columns

HYPOTHESIS: Fixing a fourth terminal `ln2.weight` coordinate at one will reduce the qualified 1,561-parameter model to 1,560 parameters while retaining at least 0.99 accuracy, because its effective scale remains representable by the corresponding unrestricted `fc1` column.

INTENDED_EDIT: Retain four learned `ln2` scales and reconstruct the final four as fixed ones.

EVIDENCE: The current 1,561-parameter design achieved 0.9999 accuracy after fixing three terminal `ln2` scales, improving on the 1,562-parameter design’s 0.9993 accuracy after fixing two; this supports testing one further scale absorption with substantial margin above 0.99.

<<<<<<< SEARCH
        # The compact bias basis leaves the terminal three channels bias-free.
        # Their scales are absorbable by the corresponding unrestricted fc1
        # columns, so reconstruct those scales as fixed ones.
        self.ln2.weight = nn.Parameter(
            self.ln2.weight[:-3].detach().clone()
        )
=======
        # LayerNorm scales immediately preceding the unrestricted fc1 are
        # absorbable by its corresponding input columns. Retain four learned
        # scales and reconstruct the final four as fixed ones.
        self.ln2.weight = nn.Parameter(
            self.ln2.weight[:-4].detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(3))
        )
=======
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(4))
        )
>>>>>>> REPLACE