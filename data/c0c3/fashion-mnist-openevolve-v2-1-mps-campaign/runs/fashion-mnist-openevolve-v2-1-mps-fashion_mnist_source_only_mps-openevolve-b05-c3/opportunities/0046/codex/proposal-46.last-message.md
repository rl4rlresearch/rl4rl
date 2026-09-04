MECHANISM: Late convolutional feature-capacity reallocation

HYPOTHESIS: Restoring the 48-unit positional head and widening the final feature map from 96 to 99 channels will exceed 9,167 correct predictions by spending spare parameters on spatial feature extraction rather than the head.

INTENDED_EDIT: Revert the classifier head to the best verified width and widen only the final convolution, producing 249,712 learned parameters.

EVIDENCE: The 48-unit reference achieved 9,167 correct, while widening its head to 52 units used comparable capacity but fell to 9,164; this motivates allocating the remaining budget to convolutional representation quality instead.

<<<<<<< SEARCH
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
=======
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 99))
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 52),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(52, 10),
=======
            nn.Linear(99 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
>>>>>>> REPLACE