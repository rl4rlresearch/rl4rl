MECHANISM: Deep-supervised recurrent-width boundary search

HYPOTHESIS: A 123-unit GRU with the qualified 27-frame schedule and training-only mean/terminal supervision will retain at least 85% validation accuracy while reducing inference MACs below the qualified 124-unit design.

INTENDED_EDIT: Reduce the recurrent state and both readout widths from 124 to 123 units, preserving the schedule, dual readout, auxiliary losses, and all training settings.

EVIDENCE: Training-only dual-view supervision raised the 124-unit design from 84.79% to 86.38% without adding inference MACs, leaving 1.38 percentage points of margin and motivating the adjacent structural width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(248, 8)
        self.mean_aux = nn.Linear(124, 8)
        self.terminal_aux = nn.Linear(124, 8)
=======
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(246, 8)
        self.mean_aux = nn.Linear(123, 8)
        self.terminal_aux = nn.Linear(123, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
>>>>>>> REPLACE