MECHANISM: Nonadjacent temporal-summary coordinate pruning

HYPOTHESIS: Omitting the first mean-summary coordinate instead of the third-from-last coordinate will retain at least 85% validation accuracy while reducing classifier width from 114 to 113, saving 6,520 validation MACs and eight parameters.

INTENDED_EDIT: Keep the verified GRU, 21-step schedule, final-hidden features, and two previously pruned summary coordinates unchanged; additionally discard mean-summary coordinate 0 and shrink the classifier input to 113.

EVIDENCE: Dropping the final two summary coordinates achieved 85.03%, but dropping the adjacent third coordinate fell to 83.93%, suggesting coordinate-specific importance rather than a demonstrated minimum classifier width; testing a distant coordinate isolates that distinction.

<<<<<<< SEARCH
        self.classifier = nn.Linear(114, 8)
=======
        self.classifier = nn.Linear(113, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
=======
            torch.cat((mean_output[:, 1:-2], hidden[:, 0, :]), dim=-1)
>>>>>>> REPLACE