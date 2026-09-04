MECHANISM: Learned mean-and-terminal dual recurrent readout

HYPOTHESIS: A 127-unit GRU using the best-performing 27-frame schedule will exceed 85% validation accuracy when its classifier jointly uses the mean recurrent output and terminal hidden state, while remaining below the qualified 128-unit, 27-step design in exact MACs.

INTENDED_EDIT: Reduce recurrent width to 127, remove the earliest frame from the 28-frame grid, and concatenate mean-pooled and terminal recurrent features for a 254-to-8 classifier.

EVIDENCE: The 127-unit model dropping frame 0 reached 84.91%, only one validation example below qualification; the added terminal-state readout restores sequence-order information discarded by mean pooling at a classifier cost much smaller than the recurrent MAC savings versus 128 units.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(254, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
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
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE