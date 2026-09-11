MECHANISM: Single-coordinate classifier-feature pruning

HYPOTHESIS: Removing one final-state coordinate from the classifier input will retain at least 85% validation accuracy while reducing total inference MACs by 5,705 and learned parameters by 7.

INTENDED_EDIT: Keep the proven 60-unit GRU, 25-frame schedule, temporal summaries, and seven-logit head, but reduce the classifier input from 180 to 179 features by omitting one final-state coordinate.

EVIDENCE: The current design qualifies at 85.03%, while reducing recurrent width to 59 or input rank to 17 failed substantially; pruning one classifier coordinate is a smaller structural reduction, and that hidden coordinate remains represented through both mean and maximum summaries.

<<<<<<< SEARCH
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
=======
        self.classifier = nn.Linear(3 * self.hidden_size - 1, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
=======
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :-1]), dim=1
        )
        relative_logits = self.classifier(features)
>>>>>>> REPLACE