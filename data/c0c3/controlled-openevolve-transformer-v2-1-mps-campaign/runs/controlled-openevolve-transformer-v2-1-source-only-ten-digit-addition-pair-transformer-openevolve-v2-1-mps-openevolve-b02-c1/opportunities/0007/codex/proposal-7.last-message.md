MECHANISM: Zero-mean LayerNorm coordinate elimination

HYPOTHESIS: Because non-affine LayerNorm outputs sum to zero across the eight residual dimensions, the MLP input projection needs only seven coordinates; reducing `fc1` by 12 parameters to produce a 1,600-parameter model will preserve at least 99% accuracy.

INTENDED_EDIT: Replace the MLP’s eight-input projection with a seven-input projection and omit the linearly dependent final normalized coordinate.

EVIDENCE: Making the MLP-preceding LayerNorm non-affine retained 99.92% accuracy at 1,612 parameters, supporting a further exact reparameterization at the same interface rather than compressing positional or token representations, which failed.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = nn.Linear(d_model - 1, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        return self.drop(self.fc2(F.gelu(self.fc1(x[..., :-1]))))
>>>>>>> REPLACE