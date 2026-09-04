MECHANISM: Adjacent dual-readout recurrent-width reduction

HYPOTHESIS: A 124-unit GRU retaining the qualified 27-frame early-drop schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing total inference MACs from 1.198B to approximately 1.180B.

INTENDED_EDIT: Reduce recurrent width to 124, classify from concatenated mean and terminal states, and use the qualified schedule that removes frame 0.

EVIDENCE: The otherwise identical 125-unit design achieved 85.52% accuracy at 1.198B MACs; testing the adjacent width is the most direct structural boundary search below the best qualified design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
=======
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(248, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=-1
        )
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE