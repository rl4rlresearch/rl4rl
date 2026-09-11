MECHANISM: Dual-view temporal readout on a nested 28-step GRU

HYPOTHESIS: A learned readout combining the final hidden state with the mean recurrent output will recover the nested 28-step model’s 0.215-point accuracy deficit and reach at least 85%, while remaining below 997 million total validation MACs.

INTENDED_EDIT: Reduce the GRU to 111 units, use the qualified 29-frame schedule minus frame 1, and classify concatenated final-state and mean-output features.

EVIDENCE: The 111-unit nested 28-step model reached 84.785% with cross-entropy nearly identical to the qualified 29-step model; adding the complementary final hidden state costs only 888 parameters and approximately 0.724 million validation MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(30, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
        steps = min(29, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        return [frame for frame in schedule if frame != 1]
>>>>>>> REPLACE