MECHANISM: Boundary-width GRU pruning with triple temporal readout

HYPOTHESIS: A 74-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.37% versus the qualified 75-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent-summary width from 75 to 74 and resize the triple-readout classifier from 225 to 222 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 75-unit design achieved 86.75% validation accuracy, while every tested triple-readout width from 75 through 80 qualified; its 1.75-point margin motivates the next one-unit structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(225, 8)
=======
        self.gru = nn.GRU(20, 74, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 75, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 74, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 74, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 74, device=device, dtype=dtype)
>>>>>>> REPLACE