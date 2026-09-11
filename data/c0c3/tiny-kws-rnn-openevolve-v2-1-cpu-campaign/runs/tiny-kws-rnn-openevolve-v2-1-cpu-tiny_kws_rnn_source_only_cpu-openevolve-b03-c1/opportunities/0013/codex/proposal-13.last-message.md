MECHANISM: Narrow gated recurrence with endpoint-and-trajectory readout

HYPOTHESIS: A 96-unit full-rate GRU classifying from both its final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by 7.2% versus the successful 100-unit model.

INTENDED_EDIT: Reduce the GRU from 100 to 96 units, preserve all 32 causal frames, and concatenate the final hidden state with the mean recurrent output for classification.

EVIDENCE: The 100-unit full-rate GRU achieved 85.03%, while every tested temporal reduction failed; retaining full temporal coverage and enriching the readout is the most direct way to compensate for another structural width reduction.

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
        return self.classifier(torch.cat((hidden[:, 0, :], mean_output), dim=-1))
>>>>>>> REPLACE