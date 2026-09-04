MECHANISM: Deep-supervised adjacent-width reduction

HYPOTHESIS: A 120-unit GRU preserving the qualified 27-frame schedule, dual readout, and training-only auxiliary supervision will retain at least 85% validation accuracy while reducing exact inference MACs below the qualified 121-unit design.

INTENDED_EDIT: Reduce the recurrent state and all associated readout dimensions from 121 to 120 units without changing training or temporal coverage.

EVIDENCE: The otherwise identical 121-unit design achieved 85.89% accuracy at 1.128B MACs, slightly improving upon both the qualified 122- and 123-unit designs; its 0.89-point qualification margin motivates testing the adjacent structural width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 121, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(242, 8)
        self.mean_aux = nn.Linear(121, 8)
        self.terminal_aux = nn.Linear(121, 8)
=======
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
        self.mean_aux = nn.Linear(120, 8)
        self.terminal_aux = nn.Linear(120, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 121, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 121, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
>>>>>>> REPLACE