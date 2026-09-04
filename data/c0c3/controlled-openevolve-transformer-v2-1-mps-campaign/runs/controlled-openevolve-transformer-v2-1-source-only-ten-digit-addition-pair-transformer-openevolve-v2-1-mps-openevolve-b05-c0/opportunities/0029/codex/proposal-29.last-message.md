MECHANISM: Third incremental value/output-bias quotient

HYPOTHESIS: Fixing a third value-bias coordinate to zero will reduce the model from 1627 to 1626 parameters while retaining at least 99% accuracy, because a position-independent value bias is absorbable into the attention output bias and the first two removals both trained successfully.

INTENDED_EDIT: Store five rather than six value-bias coordinates and reconstruct the final three coordinates as zero.

EVIDENCE: Fixing the second value-bias coordinate achieved 99.92% accuracy at 1627 parameters, while further first-head key gauges repeatedly failed; continuing the successful value-bias mechanism by one coordinate is the smallest informative reduction.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 2))
=======
        v = v + F.pad(self.v_bias, (0, 3))
>>>>>>> REPLACE