MECHANISM: Complete gated-MLP intercept fixing

HYPOTHESIS: A 804-parameter transformer will retain at least 99% accuracy because fixing the final learned value-feature intercept at zero preserves all four gated features, every input-dependent projection weight, and the qualified lexical and attention paths.

INTENDED_EDIT: Disable the gated MLP input projection bias, fixing all eight value and gate intercepts at zero and removing seven parameters from the current 811-parameter implementation.

EVIDENCE: The 805-parameter design achieved 99.93% with only the first value-feature bias learned; fixing that sole remaining intercept is the narrowest continuation of the repeatedly successful bias-removal sequence from 811 through 805 parameters.

<<<<<<< SEARCH
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
>>>>>>> REPLACE