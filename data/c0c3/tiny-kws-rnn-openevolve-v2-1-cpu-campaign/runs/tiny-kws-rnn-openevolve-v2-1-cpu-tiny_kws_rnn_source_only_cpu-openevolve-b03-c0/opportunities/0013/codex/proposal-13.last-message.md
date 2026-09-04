MECHANISM: Smaller-batch training–compensated recurrent-width compression

HYPOTHESIS: Training the 97-unit GRU with batch size 64 will approximately double optimizer updates and recover validation accuracy to at least 85%, while retaining the previously measured 97-unit inference cost.

INTENDED_EDIT: Reduce the GRU, state, summary, and classifier width from 98 to 97, and reduce training batch size from 128 to 64.

EVIDENCE: The 97-unit model narrowly missed at 84.42% while achieving lower cross-entropy (0.4598) than the passing 98-unit model (0.4703), suggesting its representation remains viable and may benefit from more optimization updates per fixed 50,000-example exposure.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

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