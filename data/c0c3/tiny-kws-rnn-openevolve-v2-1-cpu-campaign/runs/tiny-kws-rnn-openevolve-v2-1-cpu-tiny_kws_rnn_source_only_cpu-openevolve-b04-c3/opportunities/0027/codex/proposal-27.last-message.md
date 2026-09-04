MECHANISM: Late-biased 26-step temporal thinning

HYPOTHESIS: The qualified 120-unit GRU will retain at least 85% validation accuracy after dropping the next-earliest scheduled frame, while reducing total inference MACs from 1.111B to approximately 1.070B.

INTENDED_EDIT: Reconstruct the qualified 120-unit deep-supervised dual-readout model and reduce its schedule from 27 to 26 steps by removing the two earliest frames from the original 28-frame grid.

EVIDENCE: The 120-unit 27-step design achieved 86.13% accuracy with 1.13 percentage points of margin, and prior schedule testing found that removing the earliest frame was the best-performing 27-frame thinning strategy, motivating one further early-frame removal.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(244, 8)
        self.mean_aux = nn.Linear(122, 8)
        self.terminal_aux = nn.Linear(122, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
        self.mean_aux = nn.Linear(120, 8)
        self.terminal_aux = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
=======
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE