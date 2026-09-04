MECHANISM: Dual-readout recurrent-width boundary search

HYPOTHESIS: A 125-unit GRU retaining the qualified 27-frame early-drop schedule and mean-plus-terminal readout will achieve at least 85% validation accuracy while reducing exact inference MACs below the qualified 126-unit design.

INTENDED_EDIT: Reduce recurrent width from 128 to 125, classify from concatenated mean and terminal states, and switch to the qualified schedule that removes frame 0.

EVIDENCE: The 126-unit version of this exact dual-readout, 27-step design achieved 86.50% accuracy at 1.216B MACs, leaving 1.50 percentage points of qualification margin and motivating the adjacent lower-width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(250, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 125, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 125, device=device, dtype=dtype)
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
        if len(schedule) == 28:
            schedule.pop(-2)
=======
        if len(schedule) == 28:
            schedule.pop(0)
>>>>>>> REPLACE