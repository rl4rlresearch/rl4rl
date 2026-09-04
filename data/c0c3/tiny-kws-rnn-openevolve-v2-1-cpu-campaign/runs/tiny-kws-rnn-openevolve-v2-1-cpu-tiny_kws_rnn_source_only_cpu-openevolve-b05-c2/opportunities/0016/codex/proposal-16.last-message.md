MECHANISM: Boundary-width GRU pruning with triple temporal readout

HYPOTHESIS: A 76-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 77-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent summary width from 79 to 76 and resize the triple-readout classifier from 237 to 228 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 77-unit triple-readout design achieved 86.26% validation accuracy with a 1.26-point margin, after 78–80 units all achieved 86.87%; this supports probing one unit below the smallest qualified width while retaining the successful matrix-free maximum summary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 79, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(237, 8)
=======
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 79, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
>>>>>>> REPLACE