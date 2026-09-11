MECHANISM: Single-coordinate classifier-path pruning

HYPOTHESIS: The qualified 22-step schedule will retain at least 85% validation accuracy when one final-hidden coordinate is omitted from the classifier, reducing total inference MACs from 259,218,900 to 259,213,195 and parameters from 16,020 to 16,013.

INTENDED_EDIT: Adopt the qualified frames 3–21, 24, 26, and 28 schedule and reduce the bias-free classifier input from 180 to 179 features by excluding one coordinate only from its final-hidden branch.

EVIDENCE: The 22-step design achieved 85.40% accuracy at 259,218,900 MACs, while removing seven classifier biases preserved qualification. This equally small seven-weight reduction retains both complete mean and maximum summaries and 59 of 60 final-hidden coordinates, avoiding the aggressive recurrent-input and classifier-feature reductions that failed.

<<<<<<< SEARCH
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
=======
        self.classifier = nn.Linear(
            3 * self.hidden_size - 1, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
=======
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :-1]), dim=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(3, available_frames - 9)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
=======
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE