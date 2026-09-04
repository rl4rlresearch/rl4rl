MECHANISM: Recurrent capacity-for-time exchange

HYPOTHESIS: A 119-unit GRU using the best-performing 23-frame schedule will recover the 0.092-point accuracy shortfall of the 117-unit version, achieving at least 85% accuracy while reducing total inference MACs by approximately 1.1% versus the passing 117-unit, 24-step design.

INTENDED_EDIT: Increase recurrent and readout width from 117 to 119, then remove the second processed frame to execute 23 causal steps.

EVIDENCE: Removing the second processed frame produced the strongest 117-unit, 23-step result at 84.908%, narrowly missing the threshold, while the 119-unit model previously achieved 86.135% at 25 steps; this tests whether a small capacity increase can enable the lower-step schedule.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)
=======
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return schedule[1:-4] + schedule[-1:] if len(schedule) > 5 else schedule
=======
        return (
            schedule[1:2] + schedule[3:-4] + schedule[-1:]
            if len(schedule) > 6
            else schedule
        )
>>>>>>> REPLACE