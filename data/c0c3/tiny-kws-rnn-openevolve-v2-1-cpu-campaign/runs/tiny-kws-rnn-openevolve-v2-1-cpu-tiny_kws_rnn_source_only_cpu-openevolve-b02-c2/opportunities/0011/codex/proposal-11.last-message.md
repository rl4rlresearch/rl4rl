MECHANISM: Dual-view recurrent pooling

HYPOTHESIS: A 98-unit GRU classifying the concatenated mean and final recurrent outputs will reach at least 85% accuracy while using fewer total MACs than the qualified 99-unit mean-pooled GRU.

INTENDED_EDIT: Reduce the GRU width from 120 to 98 and give the classifier separate learned weights for the sequence-wide mean and terminal hidden state, retaining all 32 frames.

EVIDENCE: The mean-pooled 98-unit model narrowly missed at 84.54%, while 99 units qualified at 85.40%; adding the terminal recurrent representation costs only 784 classifier MACs per example and directly tests whether pooling—not recurrent capacity—caused the boundary failure.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
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
        pooled = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        return self.classifier(pooled)
>>>>>>> REPLACE