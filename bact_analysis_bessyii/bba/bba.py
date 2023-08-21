
import numpy as np
from bluesky import RunEngine
from bluesky.callbacks import LiveTable, LivePlot
from bluesky.plan_stubs import wait
from ophyd import EpicsMotor


class Quadrupole:
    def __init__(self, pv, name):
        self.motor = EpicsMotor(pv, name=name)

    def set_strength(self, strength):
        self.motor.set(strength)

    @property
    def position(self):
        return self.motor.position


class BeamPositionMonitor:
    def __init__(self, pv):
        self.pv = pv

    def measure_position(self):
        # Simulated beam position measurement
        return np.random.uniform(0.9, 1.1)


class BeamBasedAlignment:
    def __init__(self, quad_x_pv, quad_y_pv, bpm_pv):
        self.quad_x = Quadrupole(quad_x_pv, "quad_x")
        self.quad_y = Quadrupole(quad_y_pv, "quad_y")
        self.bpm = BeamPositionMonitor(bpm_pv)
        self.desired_position = 1.0
        self.learning_rate = 0.1
        self.num_iterations = 10

    def apply_perturbation(self, quad_x_value, quad_y_value):
        self.quad_x.set_strength(quad_x_value)
        self.quad_y.set_strength(quad_y_value)

    def perform_bba(self):
        current_position = 0.0

        for iteration in range(self.num_iterations):
            perturbation_x = np.random.uniform(-0.1, 0.1)
            perturbation_y = np.random.uniform(-0.1, 0.1)
            self.apply_perturbation(perturbation_x, perturbation_y)

            current_position = self.bpm.measure_position()
            deviation = current_position - self.desired_position

            self.quad_x.set_strength(self.quad_x.position - self.learning_rate * deviation)
            self.quad_y.set_strength(self.quad_y.position - self.learning_rate * deviation)

            yield from wait(1)

            print("Iteration:", iteration + 1, "Beam Position:", current_position)


# Create instances of the classes
quad_x_pv = "Pierre:DT:q4m2d5r:x:set"  # change to a quad
quad_y_pv = "Pierre:DT:q4m2d5r:y:set"  # change to a quad
bpm_pv = "Pierre:DT:MDIZ2T5G"

bba = BeamBasedAlignment(quad_x_pv, quad_y_pv, bpm_pv)

# Create RunEngine and set up callbacks for live feedback
RE = RunEngine()
RE.subscribe(LiveTable(["bba.quad_x.motor", "bba.quad_y.motor"]))
RE.subscribe(LivePlot(["bba.bpm.pv", "bba.quad_x.motor", "bba.quad_y.motor"]))

# Run the BBA optimization as a Bluesky plan
RE(bba.perform_bba())
