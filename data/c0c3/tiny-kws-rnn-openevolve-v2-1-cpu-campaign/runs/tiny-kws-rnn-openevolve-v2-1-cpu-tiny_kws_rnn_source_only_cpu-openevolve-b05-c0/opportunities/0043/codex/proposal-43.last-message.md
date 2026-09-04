MECHANISM: Absorb redundant normalization affine into GRU input weights

HYPOTHESIS: Disabling LayerNorm’s learned affine terms will preserve at least 85% accuracy and identical MACs and recurrent steps while reducing parameters from 14,896 to 14,856.

INTENDED_EDIT: Retain input normalization but remove its 20 learned scales and 20 learned biases.

EVIDENCE: The verified 21-step design achieved 85.15%, while three 20-step schedules and width-57 failed; preserving all recurrent computation is therefore lower risk. LayerNorm’s affine transform is representationally redundant before the GRU’s learned input weights and biases.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
>>>>>>> REPLACE