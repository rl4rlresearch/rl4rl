MECHANISM: Dual-timescale recurrent readout

HYPOTHESIS: A 98-unit GRU classifying the concatenation of its final hidden state and mean temporal output will reach at least 85% accuracy while reducing total inference MACs below the qualified 99-unit mean-only model.

INTENDED_EDIT: Reduce recurrent width to 98 and expand the classifier input to combine final-state and sequence-average representations.

EVIDENCE: The 98-unit mean-only model narrowly missed at 84.54%, while 99 units passed at 85.40%; a richer readout may recover the small accuracy gap while retaining lower recurrent MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))
>>>>>>> REPLACE