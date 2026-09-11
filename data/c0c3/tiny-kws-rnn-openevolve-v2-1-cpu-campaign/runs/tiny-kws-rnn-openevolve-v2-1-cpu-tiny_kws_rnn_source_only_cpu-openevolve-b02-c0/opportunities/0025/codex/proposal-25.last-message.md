MECHANISM: Two-step certified early exit

HYPOTHESIS: Extending exact classifier-bound checks from step 29 to step 28 will preserve validation accuracy at or above 85% while reducing total inference MACs below 825,918,687 by allowing confident examples to skip both remaining recurrent steps.

INTENDED_EDIT: Run the learned classifier beginning at recurrent step 28 and apply the existing mathematically conservative bounded-output certificate at both steps 28 and 29.

EVIDENCE: The step-29 certificate preserved 85.28% accuracy while 86.3% of examples exited early; the broader step-20 attempt provided no contrary accuracy evidence because training timed out. Adding only the immediately preceding check is a lower-overhead probe whose certified exits cannot change the final predicted class.

<<<<<<< SEARCH
        if bool(torch.all(count < 29.0)):
            return averaged[:, :8]
=======
        if bool(torch.all(count < 28.0)):
            return averaged[:, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        eligible = completed == float(total_steps - 1)
=======
        eligible = (completed >= float(total_steps - 2)) & (
            completed < float(total_steps)
        )
>>>>>>> REPLACE