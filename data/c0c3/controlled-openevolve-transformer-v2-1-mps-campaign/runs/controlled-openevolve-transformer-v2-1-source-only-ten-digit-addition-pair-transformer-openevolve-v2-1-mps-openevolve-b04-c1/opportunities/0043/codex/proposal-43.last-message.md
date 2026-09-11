MECHANISM: Redundant value-bias elimination

HYPOTHESIS: Reducing the attention value bias from three learned coordinates to two will lower the model from 1602 to 1601 parameters while retaining at least 99% accuracy, because attention rows sum to one and the omitted value-bias effect can be absorbed by the seven-coordinate output-projection bias modulo residual-stream shifts.

INTENDED_EDIT: Remove the third learned value-bias coordinate and increase its fixed-zero padding accordingly, leaving initialization, architecture, and training unchanged.

EVIDENCE: The verified 1602-parameter design reached 99.99%, while further `fc2` quotienting was optimization-sensitive; this instead removes one parameter from a different branch where the projection bias already spans every residual-relevant constant output direction.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 5))
=======
        v = v + F.pad(self.v_bias, (0, 6))
>>>>>>> REPLACE