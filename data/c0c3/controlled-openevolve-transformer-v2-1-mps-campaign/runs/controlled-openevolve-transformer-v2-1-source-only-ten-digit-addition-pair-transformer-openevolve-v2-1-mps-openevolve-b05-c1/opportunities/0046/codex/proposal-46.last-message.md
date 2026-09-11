MECHANISM: MLP output-gauge reuse

HYPOTHESIS: Reusing the LayerNorm-invisible common mode of the MLP output bias as its hidden-unit bias will reduce the model from 1,376 to 1,375 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Remove the standalone `fc1` bias and derive its zero-initialized learned value from the mean of `fc2.bias`.

EVIDENCE: The current 1,376-parameter model achieved 100% accuracy after successfully reusing the attention-output bias common mode; the MLP output is likewise followed by LayerNorm, making its bias common mode functionally redundant and available for the same reuse.

<<<<<<< SEARCH
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        hidden = self.fc1(x) + self.fc2.bias.mean()
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE