MECHANISM: Boundary-width GRU pruning with triple temporal readout

HYPOTHESIS: A 75-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.3% versus the qualified 76-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent summaries from 80 to 75 units and resize the triple-readout classifier from 240 to 225 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 76-unit triple-readout design achieved 85.89% validation accuracy, and every tested triple-readout width from 76 through 80 qualified; probing 75 units directly identifies whether the stable pruning trend extends below the current lowest-MAC design.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
=======
        self.gru = nn.GRU(20, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(225, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 75, device=device, dtype=dtype)
>>>>>>> REPLACE