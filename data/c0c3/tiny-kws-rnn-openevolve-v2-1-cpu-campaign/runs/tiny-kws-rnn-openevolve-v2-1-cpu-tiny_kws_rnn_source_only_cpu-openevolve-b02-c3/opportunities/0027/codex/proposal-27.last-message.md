MECHANISM: Parameter-free cross-timescale interaction readout

HYPOTHESIS: An 86-unit GRU over 28 frames with mean–final multiplicative features will achieve at least 85% validation accuracy while keeping total inference MACs below the qualified 88-unit, 27-frame model.

INTENDED_EDIT: Use the near-qualified 86-unit, 28-frame architecture and augment its linear classifier with the elementwise product of mean and final recurrent outputs.

EVIDENCE: The 86-unit, 28-frame model missed qualification by one validation example while achieving lower cross-entropy than the qualified 88-unit, 27-frame model; adding an inexpensive interaction feature targets this narrow accuracy gap with a predicted 625,763,520 total MACs, still below 628,554,080.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(258, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
=======
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        features = torch.cat(
            (mean_output, final_output, mean_output * final_output), dim=1
        )
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = max(available_frames - 30, 0)
=======
        start = max(available_frames - 28, 0)
>>>>>>> REPLACE