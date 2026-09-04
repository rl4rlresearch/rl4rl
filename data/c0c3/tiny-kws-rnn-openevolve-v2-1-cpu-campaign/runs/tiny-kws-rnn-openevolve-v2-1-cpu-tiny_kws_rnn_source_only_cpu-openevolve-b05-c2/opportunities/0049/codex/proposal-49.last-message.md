MECHANISM: One-unit recurrent-width contraction at the qualified temporal frontier

HYPOTHESIS: A 62-unit signed-extrema GRU on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 262,654,940.

INTENDED_EDIT: Reduce the GRU width from 63 to 62 units and resize its recurrent states and classifier input accordingly.

EVIDENCE: The 63-unit, 21-step model qualified at 85.03%, while several 20-step schedules failed; the adjacent width reduction is therefore the most informative orthogonal test of the structural-compute boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 62, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(248, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 63, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 62, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 62, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 62, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 62, device=device, dtype=dtype)
>>>>>>> REPLACE