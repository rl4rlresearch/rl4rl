MECHANISM: Adjacent-width GRU pruning with triple temporal readout

HYPOTHESIS: A 65-unit full-resolution GRU with mean, final, and temporal-maximum readouts will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.66% versus the qualified 66-unit design.

INTENDED_EDIT: Reduce the GRU and recurrent-summary width from 70 to 65 units and resize the triple-readout classifier from 210 to 195 inputs, preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 66-unit design qualified at 85.28% accuracy and 445,381,200 total MACs, while every tested triple-readout width from 66 through 80 qualified; testing the adjacent 65-unit width directly identifies the next structural compute boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)
=======
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(195, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 70, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 65, device=device, dtype=dtype)
>>>>>>> REPLACE