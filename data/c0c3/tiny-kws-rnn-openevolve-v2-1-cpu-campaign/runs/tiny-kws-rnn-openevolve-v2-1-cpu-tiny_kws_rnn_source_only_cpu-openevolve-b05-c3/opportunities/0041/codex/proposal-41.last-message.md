MECHANISM: Five-frame clocked slow recurrence

HYPOTHESIS: Updating the qualified 80-unit slow GRU from four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 549,244,800 to approximately 517,948,800.

INTENDED_EDIT: Change the slow recurrent clock from every fourth fast step to every fifth fast step and average each corresponding five-frame block.

EVIDENCE: The current dual-timescale model achieved 86.75% accuracy with five slow transitions, leaving a 1.75-point margin; reducing to four slow transitions directly tests how much slow temporal resolution is necessary while preserving both proven 80-unit GRUs and the 20-frame acoustic path.

<<<<<<< SEARCH
        if int(count[0, 0].detach().item()) % 4 == 0:
            slow_input = self.slow_norm(block_sum * 0.25)
=======
        if int(count[0, 0].detach().item()) % 5 == 0:
            slow_input = self.slow_norm(block_sum * 0.2)
>>>>>>> REPLACE