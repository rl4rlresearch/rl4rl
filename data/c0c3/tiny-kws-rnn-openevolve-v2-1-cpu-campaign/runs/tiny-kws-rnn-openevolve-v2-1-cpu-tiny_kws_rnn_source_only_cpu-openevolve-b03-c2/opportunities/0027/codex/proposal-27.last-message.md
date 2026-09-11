MECHANISM: Second-coordinate complementary-readout pruning

HYPOTHESIS: The qualified 110-unit, 22-step model will retain at least 85% validation accuracy when a second final-state coordinate is removed, reducing dense validation MACs by another 6,520.

INTENDED_EDIT: Restore the qualified 22-frame schedule and 110-unit GRU, then classify from all 110 mean coordinates and 108 final-state coordinates.

EVIDENCE: The first final-state coordinate removal improved accuracy from 85.03% to 85.15% while reducing MACs; larger reductions to 109 recurrent units or seven learned logits failed, motivating another minimal readout-only reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(218, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
=======
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-2]), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames >= 4:
            return [
                frame
                for frame in schedule
                if frame not in (1, available_frames - 2)
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