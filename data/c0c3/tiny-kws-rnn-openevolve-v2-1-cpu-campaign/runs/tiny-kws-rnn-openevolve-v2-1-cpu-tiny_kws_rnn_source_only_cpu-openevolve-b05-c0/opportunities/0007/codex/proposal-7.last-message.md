MECHANISM: Optimization-supported width reduction with dual temporal readout

HYPOTHESIS: A 104-unit GRU using both mean and final recurrent outputs, trained with batch size 64, will retain at least 85% accuracy while reducing total inference MACs by approximately 12.7% versus the verified 112-unit model.

INTENDED_EDIT: Reduce GRU width to 104, concatenate mean-pooled and final hidden features for classification, and double the number of optimizer updates through a smaller training batch.

EVIDENCE: Full 32-frame coverage with 112 units achieved 85.15%, whereas reducing temporal coverage repeatedly failed or approached the threshold; retaining all frames while trimming width is therefore the strongest cost-reduction direction, with the richer readout and additional optimizer steps intended to recover capacity.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(208, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))
>>>>>>> REPLACE