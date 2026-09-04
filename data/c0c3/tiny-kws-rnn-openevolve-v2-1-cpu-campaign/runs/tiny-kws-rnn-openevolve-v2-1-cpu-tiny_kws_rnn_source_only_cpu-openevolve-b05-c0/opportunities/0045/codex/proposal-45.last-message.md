MECHANISM: Progressive temporal-summary coordinate pruning

HYPOTHESIS: Omitting a second mean-summary coordinate will retain at least 85% validation accuracy while reducing total inference by another 6,520 MACs and parameters by eight, with all 21 recurrent steps unchanged.

INTENDED_EDIT: Classify from 56 mean-summary features and all 58 final-hidden features, reducing the classifier input width from 115 to 114.

EVIDENCE: Omitting one mean-summary coordinate retained 85.03% accuracy and reduced both MACs and parameters; pruning one adjacent coordinate is the smallest evidence-backed continuation that can improve the primary MAC objective without altering the verified recurrent path or schedule.

<<<<<<< SEARCH
        self.classifier = nn.Linear(115, 8)
=======
        self.classifier = nn.Linear(114, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
            torch.cat((mean_output[:, :-1], hidden[:, 0, :]), dim=-1)
=======
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
>>>>>>> REPLACE