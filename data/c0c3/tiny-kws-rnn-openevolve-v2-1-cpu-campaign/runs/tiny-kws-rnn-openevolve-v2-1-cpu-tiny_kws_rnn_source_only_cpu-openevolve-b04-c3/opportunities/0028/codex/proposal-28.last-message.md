MECHANISM: Deep-supervised adjacent-width reduction at 26 recurrent steps

HYPOTHESIS: A 119-unit GRU using the qualified 26-frame schedule and unchanged dual-view supervision will retain at least 85% validation accuracy while reducing total inference MACs from 1,069,540,800 to approximately 1,053,063,130.

INTENDED_EDIT: Reconstruct the best 26-step design while reducing the GRU state and all associated readout dimensions from 120 to 119 units.

EVIDENCE: The 120-unit 26-step design qualified at 85.28%; at 27 steps, every adjacent reduction from 123 through 120 units qualified, with accuracy improving from 85.77% to 86.13%, supporting one adjacent width test on the lower-step design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(246, 8)
        self.mean_aux = nn.Linear(123, 8)
        self.terminal_aux = nn.Linear(123, 8)
=======
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(238, 8)
        self.mean_aux = nn.Linear(119, 8)
        self.terminal_aux = nn.Linear(119, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
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