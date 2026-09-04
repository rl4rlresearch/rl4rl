MECHANISM: Bias-free eight-logit readout

HYPOTHESIS: Removing the classifier’s eight bias parameters will retain at least 85% validation accuracy with unchanged MACs and recurrent steps, while reducing learned parameters from 45,360 to 45,352.

INTENDED_EDIT: Preserve the qualified 110-unit, 22-step, 219-feature architecture and disable only the final classifier bias.

EVIDENCE: The current design achieved 85.15% accuracy; removing 40 LayerNorm affine parameters missed by only 0.09 points, while further feature pruning failed substantially, motivating a smaller parameter-only reduction that leaves every recurrent and readout coordinate intact.

<<<<<<< SEARCH
        self.classifier = nn.Linear(219, 8)
=======
        self.classifier = nn.Linear(219, 8, bias=False)
>>>>>>> REPLACE