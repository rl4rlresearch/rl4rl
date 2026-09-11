MECHANISM: Boundary-width GRU pruning with triple temporal readout

HYPOTHESIS: A 77-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 78-unit design.

INTENDED_EDIT: Reduce the GRU width from 78 to 77 and resize its recurrent state and triple-readout classifier accordingly, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 78-, 79-, and 80-unit triple-readout designs all achieved 86.87% validation accuracy, indicating a stable accuracy plateau with 1.87 points of margin above the requirement and motivating another one-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 78, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(234, 8)
=======
        self.gru = nn.GRU(20, 77, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(231, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 78, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 78, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 77, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 77, device=device, dtype=dtype)
>>>>>>> REPLACE