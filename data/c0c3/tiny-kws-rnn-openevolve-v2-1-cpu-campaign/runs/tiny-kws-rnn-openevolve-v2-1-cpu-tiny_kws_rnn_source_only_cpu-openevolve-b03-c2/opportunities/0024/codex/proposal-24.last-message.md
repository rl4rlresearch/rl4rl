MECHANISM: Seven-coordinate centered logit basis

HYPOTHESIS: The qualified 110-unit, 23-step model will retain at least 85% validation accuracy while eliminating 179,300 dense validation MACs, because eight-way softmax needs only seven independent logits after removing its common-offset degree of freedom.

INTENDED_EDIT: Restore the qualified 23-frame dual-readout GRU, but learn seven classifier outputs and derive the eighth as their negative sum, reducing total inference MACs and parameters without restricting softmax decision expressivity.

EVIDENCE: The qualified 110-unit, 23-step design reached 85.52% accuracy, while reducing it to 109 units or 22 steps missed the threshold; this motivates preserving its recurrent path and removing redundancy only from the eight-way output parameterization.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(220, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        basis_logits = self.classifier(features)
        final_logit = -basis_logits.sum(dim=1, keepdim=True)
        return torch.cat((basis_logits, final_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
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
        return schedule
>>>>>>> REPLACE