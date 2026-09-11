MECHANISM: Hard-target training-compensated recurrent-width compression

HYPOTHESIS: Removing label smoothing will sharpen class boundaries enough for the 97-unit GRU to recover the 0.58-point accuracy shortfall and reach at least 85%, while retaining its lower measured inference cost and avoiding the batch-64 timeout.

INTENDED_EDIT: Reduce the GRU and classifier width from 98 to 97, keep batch size 128, and replace label-smoothed cross-entropy with standard cross-entropy.

EVIDENCE: The 97-unit model achieved 84.42% accuracy but lower validation cross-entropy than the passing 98-unit model (0.4598 versus 0.4703), indicating viable representations; the attempted smaller-batch remedy timed out, motivating a zero-runtime-cost objective adjustment.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    return F.cross_entropy(logits, labels)
>>>>>>> REPLACE