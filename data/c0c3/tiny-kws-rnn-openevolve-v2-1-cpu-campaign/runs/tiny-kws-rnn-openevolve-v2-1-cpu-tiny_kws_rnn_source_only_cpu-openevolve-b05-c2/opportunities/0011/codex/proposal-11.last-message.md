MECHANISM: Boundary-seeking GRU width pruning with dual-timescale readout

HYPOTHESIS: An 81-unit full-resolution GRU with concatenated mean/final states will achieve at least 85% validation accuracy while reducing total inference MACs by approximately 2.2% versus the qualified 82-unit design.

INTENDED_EDIT: Reduce recurrent and summary width from 112 to 81, resize the classifier for a 162-feature mean/final readout, and retain all 32 causal frames and the established training procedure.

EVIDENCE: The 82-unit dual-readout model achieved 86.01% accuracy, while 80 units achieved 84.29%; testing the intervening 81-unit width directly resolves the remaining feasibility boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 81, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(162, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 81, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 81, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(torch.cat((mean_output, final_output), dim=-1))
>>>>>>> REPLACE