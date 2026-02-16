import pandas as pd
import openmdao.api as om

class PowerCoefficient(om.JaxExplicitComponent):
    """
    Computes power coefficient: Cp = P / (rho * (rpm/60)^3 * D^5)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')

    
    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('power', shape=(nn,), units='W')
        self.add_input('rpm', shape=(nn,), units='rpm')
        self.add_input('diameter', shape=(nn,), units='m')
        self.add_input('rho', shape=(nn,), units='kg/m**3')

        self.add_output('Cp', shape=(nn,))
        self.declare_partials('*', '*', method='exact')
    
    def compute_primal(self, power, rpm, diameter, rho):

        # Convert rpm to rev/s and compute Cp
        n = rpm / 60.0  # revolutions per second
        # Take absolute value of power to allow generating power to pass through

        Cp = power / (rho * n**3 * diameter**5)


        return Cp
    
    def self_get_statics(self):
        return (self.options['min_power_W'], self.options['max_power_W'], self.options['mu'])
