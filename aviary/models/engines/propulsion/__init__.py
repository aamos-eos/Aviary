from .motor import EmpiricalMotor, RubberMotor
from .nacelle_splitter import PowerSplitNacelle
from .prop import EmpiricalPropellerCoeffMM, EmpiricalPropellerCoeffMMMtip
from .turbine import TurboMission
from .battery import EmpiricalBatteryPower
# Pre-made propulsion systems
from .systems import (
    ParallelHybridElectricPropulsionSystem,
)
from .q400 import Q400Ptrain
