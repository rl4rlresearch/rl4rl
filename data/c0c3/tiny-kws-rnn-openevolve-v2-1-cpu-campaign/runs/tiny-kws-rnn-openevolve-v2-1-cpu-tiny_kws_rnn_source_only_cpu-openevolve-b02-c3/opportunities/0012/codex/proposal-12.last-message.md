MECHANISM: Conservative dual-timescale width reduction

HYPOTHESIS: A 96-unit GRU classifying concatenated mean and final-state features will retain at least 85% validation accuracy while reducing total inference MACs below the qualified 98-unit model’s 906,045,280.

INTENDED_EDIT: Reduce GRU width from 100 to 96 and use the qualified mean-plus-final recurrent readout with a 192-dimensional classifier input.

EVIDENCE: The 98-unit dual-timescale model achieved 87.12%, outperforming the 98-unit mean-only model by 2.58 points; a two-unit reduction is a conservative test of how much of that accuracy margin can be converted into structural MAC savings.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
>>>>>>> REPLACE