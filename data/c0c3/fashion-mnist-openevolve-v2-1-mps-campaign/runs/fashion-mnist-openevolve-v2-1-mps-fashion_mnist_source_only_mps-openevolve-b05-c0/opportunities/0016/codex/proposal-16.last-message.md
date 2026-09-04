MECHANISM: Exposure-normalized strided tail EMA

HYPOTHESIS: Updating the tail EMA once every four batch-32 steps will exceed 9,250 correct predictions by matching the successful batch-128 EMA’s examples-based averaging horizon while reducing averaging overhead.

INTENDED_EDIT: Keep batch size, architecture, loss, optimizer, and schedule unchanged, but perform EMA interpolation every fourth step after its midpoint initialization.

EVIDENCE: Tail EMA improved the batch-128 design from 9,168 to 9,170 correct, while the unchanged per-step EMA at batch 32 averages over one quarter as many examples; the attempted lower interpolation rate targeted this mismatch but timed out, motivating an equivalent lower-overhead strided update.

<<<<<<< SEARCH
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
            else:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

            if completed_steps >= total_steps:
=======
            ema_start = total_steps // 2
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

            if completed_steps >= total_steps:
>>>>>>> REPLACE