MECHANISM: Single-coordinate complementary-readout pruning

HYPOTHESIS: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when one of its 110 final-state readout coordinates is removed, while reducing dense classifier MACs by 6,520 and learned parameters by 8.

INTENDED_EDIT: Adopt the qualified 22-frame schedule and classify from the complete 110-coordinate mean output plus 109 coordinates of the final hidden state.

EVIDENCE: The 110-unit, 22-step design achieved 85.03% accuracy, while reducing recurrent width to 109 units and reducing outputs to seven logits both failed; pruning one of 220 readout coordinates preserves the qualified recurrent path and all eight logits.

<<<<<<< SEARCH
        self.classifier = nn.Linear(220, 8)
=======
        self.classifier = nn.Linear(219, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
=======
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames >= 5:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    1,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
=======
        if available_frames >= 6:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    0,
                    1,
                    2,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
>>>>>>> REPLACE