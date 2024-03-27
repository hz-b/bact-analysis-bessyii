import numpy as np

from bact_analysis_bessyii.steerer_response.model import AcceleratorDescription, Position, SurveyPositions


def twiss_from_at() -> AcceleratorDescription:
    from dt4acc.resources.bessy2_sr_reflat import bessy2Lattice
    from dt4acc.calculator.pyat_calculator import PyAtTwissCalculator

    acc = bessy2Lattice()
    twiss_calculator = PyAtTwissCalculator(acc)
    twiss = twiss_calculator.calculate()
    s = np.add.accumulate([0] + [elem.Length for elem in acc])
    # plot_twiss_functions(twiss, )
    return AcceleratorDescription(
        twiss=twiss,
        survey=SurveyPositions(
            positions=[
                Position(value=t_s, name=name) for t_s, name in zip(s, twiss.names)
            ]
        ),
    )