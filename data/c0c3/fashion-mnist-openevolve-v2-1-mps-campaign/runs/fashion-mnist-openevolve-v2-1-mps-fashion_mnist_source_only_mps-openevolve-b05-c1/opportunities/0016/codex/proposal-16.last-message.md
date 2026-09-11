MECHANISM: Compute-neutral classifier bottleneck widening

HYPOTHESIS: Widening the flattened classifier bottleneck from 30 to 31 features will exceed 9,240 correct predictions by using additional representational capacity without the runtime overhead of convolutional attention.

INTENDED_EDIT: Increase both classifier linear layers to a 31-feature hidden width, raising learned parameters from 245,044 to 248,191.

EVIDENCE: The strongest design leaves 4,956 parameters unused, while the 2,484-parameter squeeze-and-excitation addition timed out; widening the existing head uses 3,147 of those parameters with negligible added computation.

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
=======
            nn.Linear(64 * 7 * 7, 31),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(31, 10),
>>>>>>> REPLACE