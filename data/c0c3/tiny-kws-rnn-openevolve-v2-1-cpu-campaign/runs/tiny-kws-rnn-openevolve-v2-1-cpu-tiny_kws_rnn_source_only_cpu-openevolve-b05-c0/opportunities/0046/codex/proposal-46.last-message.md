MECHANISM: Progressive temporal-summary coordinate pruning

HYPOTHESIS: Omitting a third mean-summary coordinate will retain at least 85% validation accuracy while preserving all 21 recurrent steps and reducing total inference by another 6,520 MACs and parameters by eight.

INTENDED_EDIT: Reduce the classifier input width from 114 to 113 by using 55 mean-summary features alongside all 58 final-hidden features.

EVIDENCE: Omitting one and then two mean-summary coordinates each retained 85.03% accuracy while reducing MACs and parameters, making one additional coordinate the smallest evidence-backed structural reduction.

<<<<<<< SEARCH
        self.classifier = nn.Linear(114, 8)
=======
        self.classifier = nn.Linear(113, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
=======
            torch.cat((mean_output[:, :-3], hidden[:, 0, :]), dim=-1)
>>>>>>> REPLACE