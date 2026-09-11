MECHANISM: Conservative recurrent-width reduction on the qualified sparse schedule

HYPOTHESIS: A 109-unit dual-readout GRU on the qualified 23-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 1.67%.

INTENDED_EDIT: Adopt the qualified 23-step schedule and reduce the GRU, summary, and dual-view classifier widths from 110 to 109 units.

EVIDENCE: The 110-unit 23-step design achieved 85.52% accuracy, while the prior 111-to-110 reduction on the qualified 26-step schedule retained accuracy and improved it from 86.13% to 86.63%, supporting another one-unit reduction without changing temporal coverage.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
=======
        self.gru = nn.GRU(20, 109, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(218, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 109, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 109, device=device, dtype=dtype)
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
                    1,
                    2,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
>>>>>>> REPLACE