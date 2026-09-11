MECHANISM: Adjacent-width GRU pruning with triple temporal readout

HYPOTHESIS: A 66-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.62% versus the qualified 67-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent-summary width from 69 to 66 units and resize the triple-readout classifier from 207 to 198 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 67-unit design achieved 86.63% validation accuracy at 457,371,480 total MACs, while every tested triple-readout width from 67 through 80 qualified; its 1.63-point margin motivates testing the adjacent structural compute boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(207, 8)
=======
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 69, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 66, device=device, dtype=dtype)
>>>>>>> REPLACE