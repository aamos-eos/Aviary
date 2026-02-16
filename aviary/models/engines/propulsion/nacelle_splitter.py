import numpy as np
import openmdao.api as om

from aviary.utils.matrix_vector_converter import MatrixToVectorConverter, VectorToMatrixConverter
from aviary.models.engines.propulsion.turbine.turbo_empirical_mm_structured import ComputeMaxPower
from aviary.models.engines.propulsion.turbine.import_sorted_turbo_data import SortedTurboData
from aviary.utils.smooth_minmax import SmoothMaxComp
from aviary.utils.broadcast_scalars import ScalarToMatrixBroadcast
from aviary.models.engines.propulsion.motor.h3x_motor_data_web import MotorDataPowerVoltCurve # Power to voltage lookup table
from scipy.interpolate import griddata



class MotorPowerToThrottleMatrix(om.ExplicitComponent):
    """
    Matrix-based component to compute throttle for multiple motors
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=1, desc='Number of propellers/motors')

    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        # Inputs
        self.add_input('mech_power', shape=(npp, nn), units='kW', desc='Mechanical power for all motors')
        self.add_input('rated_power', shape=(npp, nn), units='kW', desc='Rated power for all motors')
        
        # Output - matrix throttle
        self.add_output('throttle', shape=(npp, nn), desc='Throttle for all motors')
    
        # Declare partials - diagonal structure for element-wise operations
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('throttle', 'mech_power', rows=rows, cols=cols, method='exact')
        self.declare_partials('throttle', 'rated_power', rows=rows, cols=cols, method='exact')
    
    def compute(self, inputs, outputs):
        # Calculate throttle for all motors: throttle = mech_power / rated_power
        mech_power = inputs['mech_power']
        rated_power = inputs['rated_power']
        outputs['throttle'] = mech_power / rated_power
    
    def compute_partials(self, inputs, partials):
        mech_power = inputs['mech_power']
        rated_power = inputs['rated_power']
        
        # d(throttle)/d(mech_power) = 1 / rated_power
        partials['throttle', 'mech_power'] = (1.0 / rated_power).flatten()
        
        # d(throttle)/d(rated_power) = -mech_power / rated_power^2
        partials['throttle', 'rated_power'] = (-mech_power / rated_power**2).flatten()


class MotorThrottleToPowerMatrix(om.ExplicitComponent):
    """
    Matrix-based component to compute motor power from throttle for multiple motors.
    Inverse of MotorPowerToThrottleMatrix: mech_power = throttle * rated_power
    
    Used when motor_command = 'throttle' and power_spec = 'independent'
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=1, desc='Number of propellers/motors')

    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        
        # Inputs
        self.add_input('motor_throttle', shape=(npp, nn), desc='Throttle for all motors (0-1)')
        self.add_input('max_power_em', shape=(npp, nn), units='kW', desc='Max power for all motors')
        
        # Output - matrix mechanical power
        self.add_output('unit_mech_power_calc_em', shape=(npp, nn), units='kW', desc='Mechanical power for all motors')
    
        # Declare partials - diagonal structure for element-wise operations
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('unit_mech_power_calc_em', 'motor_throttle', rows=rows, cols=cols, method='exact')
        self.declare_partials('unit_mech_power_calc_em', 'max_power_em', rows=rows, cols=cols, method='exact')
    
    def compute(self, inputs, outputs):
        # Calculate power for all motors: mech_power = throttle * max_power
        motor_throttle = inputs['motor_throttle']
        max_power_em = inputs['max_power_em']
        outputs['unit_mech_power_calc_em'] = motor_throttle * max_power_em
    
    def compute_partials(self, inputs, partials):
        motor_throttle = inputs['motor_throttle']
        max_power_em = inputs['max_power_em']
        
        # d(mech_power)/d(throttle) = max_power
        partials['unit_mech_power_calc_em', 'motor_throttle'] = max_power_em.flatten()
        
        # d(mech_power)/d(max_power) = throttle
        partials['unit_mech_power_calc_em', 'max_power_em'] = motor_throttle.flatten()


class RubberMotorMaxPowerScaler(om.ExplicitComponent):
    """
    Scales baseline motor max power for a rubber-sized motor.
    
    max_power_em = max_power_em_baseline * (rated_power_em / reference_power_em)
    
    This preserves the voltage-dependent power derating behavior from the baseline motor
    while scaling to the new motor rating.
    """
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of analysis points')
        self.options.declare('num_props', default=4, desc='Number of propellers/motors')
        self.options.declare('reference_power_em', default=2276.0, desc='Reference motor power (kW)')
    
    def setup(self):
        nn = self.options['num_nodes']
        npp = self.options['num_props']
        ref_power = self.options['reference_power_em']
        
        # Inputs
        self.add_input('max_power_em_baseline', shape=(npp, nn), units='kW', 
                      desc='Baseline motor max power from empirical map')
        self.add_input('rated_power_em', shape=(npp, nn), units='kW',
                      desc='Rated power of rubber-sized motor')
        
        # Output - same name as downstream expects
        self.add_output('max_power_em', shape=(npp, nn), units='kW',
                       desc='Scaled max power for rubber motor')
        
        # Partials
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('max_power_em', 'max_power_em_baseline', rows=rows, cols=cols)
        self.declare_partials('max_power_em', 'rated_power_em', rows=rows, cols=cols)
    
    def compute(self, inputs, outputs):
        ref_power = self.options['reference_power_em']
        max_power_baseline = inputs['max_power_em_baseline']
        rated_power = inputs['rated_power_em']
        
        # Scale baseline max power by ratio of rated to reference
        outputs['max_power_em'] = max_power_baseline * (rated_power / ref_power)
    
    def compute_partials(self, inputs, partials):
        ref_power = self.options['reference_power_em']
        max_power_baseline = inputs['max_power_em_baseline'].flatten()
        rated_power = inputs['rated_power_em'].flatten()
        
        # d(max_power_em)/d(max_power_em_baseline) = rated_power / ref_power
        partials['max_power_em', 'max_power_em_baseline'] = rated_power / ref_power
        
        # d(max_power_em)/d(rated_power_em) = max_power_baseline / ref_power
        partials['max_power_em', 'rated_power_em'] = max_power_baseline / ref_power


class PowerSplitNacelle(om.Group):
    """
    A power split mechanism between the turbo and electric motor within the nacelle.
    This is now a group that contains the appropriate JAX component based on the rule and bias options.
    """
    _data_loaded = False

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_props", default=1, desc="Number of turbines in the nacelle")
        self.options.declare("num_em_per_nac", default=1, desc="Number of electric motors in the nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines in the nacelle")
        self.options.declare("rule", default="fraction", desc="Control strategy - fraction or fixed power")
        self.options.declare("bias", default="gt", desc="Determines which component the power split rule is with respect to")
        self.options.declare("prop_gt_gb_efficiency", default=1.0, desc="Transmission efficiency from gas turbine to propeller (dimensionless)")
        self.options.declare("prop_em_gb_efficiency", default=1.0, desc="Transmission efficiency from electric motor to propeller (dimensionless)")
        self.options.declare("power_spec", default="relative", desc="Power specification rule: relative or independent")
        self.options.declare("motor_command", default="power", desc="Motor control: 'power', 'throttle', or 'torque_speed'")
        self.options.declare("size_motor", default=False, desc="Whether to use a rubber motor for sizing or not")
        self.options.declare("reference_power_em", default=2276.0, desc="Reference max total motor power (kW) for rubber motor scaling")
        self.options.declare("turb_type", default='PT6', desc="Turbine type: PT6 or ACCE")

        self._load_data()
        
    @classmethod
    def _load_data(cls):
        if cls._data_loaded:
            return
        

        MotorDataPowerVoltCurve.load_data(motor_filename='aviary/models/engines/propulsion/empirical_data/H3X_HPDM_2300_volts.xlsx')

        rpm_data = MotorDataPowerVoltCurve.rpm_data
        voltage_data = MotorDataPowerVoltCurve.voltage_data
        power_kW_data = MotorDataPowerVoltCurve.power_data

        min_rpm = np.min(rpm_data)
        max_rpm = np.max(rpm_data)
        min_voltage = np.min(voltage_data)
        max_voltage = np.max(voltage_data)
        min_power = np.min(power_kW_data)
        max_power = np.max(power_kW_data)

        n_rpm = 50
        n_voltage = 50
        n_power = 50
        
        vect_rpm = np.linspace(min_rpm, max_rpm, n_rpm)
        vect_voltage = np.linspace(min_voltage, max_voltage, n_voltage)
        vect_power = np.linspace(min_power, max_power, n_power)
        vect_power = vect_power * 1000

        map_power_from_rpm_voltage = np.zeros((n_rpm, n_voltage))
        rpm_grid, voltage_grid = np.meshgrid(vect_rpm, vect_voltage, indexing='ij')
        grid_points_rpm_voltage = np.column_stack([rpm_grid.flatten(), voltage_grid.flatten()])
        map_power_from_rpm_voltage_flat = griddata(np.column_stack([rpm_data, voltage_data]), power_kW_data, grid_points_rpm_voltage, method='cubic', fill_value=0.2)
        map_power_from_rpm_voltage = map_power_from_rpm_voltage_flat.reshape(len(vect_rpm), len(vect_voltage))

        cls._motor_map_power_from_rpm_voltage = map_power_from_rpm_voltage
        cls._vect_motor_voltage = vect_voltage
        cls._vect_motor_rpm = vect_rpm

        cls._data_loaded = True


    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        rule = self.options["rule"]
        bias = self.options["bias"]
        num_em = self.options["num_em_per_nac"]
        num_turb = self.options["num_turb_per_nac"]
        power_spec = self.options["power_spec"]
        size_motor = self.options["size_motor"]
        turb_type = self.options["turb_type"]

        # Load turbo data based on turb_type (motor data already loaded in initialize)
        SortedTurboData.load_data(csv_filename=f'aviary/models/engines/propulsion/empirical_data/sorted_turbo_dataset_{turb_type}.csv')

        #self.connect('altitude', 'lim_min_altitude.input_array')
        self.add_subsystem('lim_min_alt', SmoothMaxComp(num_nodes=nn, mode='limit', units='m', limit_val=1e-6, n_comps=1), promotes_inputs=[('input_array', 'altitude')], promotes_outputs=[])

        #self.add_subsystem('lim_min_alt', SmoothMaxComp(num_nodes=nn, mode='limit', units='m', limit_val=1e-6, n_comps=1, expected_min=1e-6, expected_max=23000*0.3048), promotes_inputs=[('input_array', 'altitude')], promotes_outputs=[])
        #self.add_subsystem('alt_clip', AltitudeClipper(num_nodes=nn), promotes_inputs=["altitude"])


        self.connect('lim_min_alt.output', f'turb_compute_max_power.altitude')
        self.add_subsystem(f'turb_compute_max_power', ComputeMaxPower(num_nodes=nn,
                                                                                turb_type=turb_type,
                                                                                disa_unique=SortedTurboData.disa_unique,
                                                                                alt_unique=SortedTurboData.alt_unique,
                                                                                mach_unique=SortedTurboData.mach_unique,
                                                                                max_power_matrix=SortedTurboData.max_power_matrix),
                                                                                promotes_inputs=["mach", "disa"], promotes_outputs=[])
        
        if power_spec == 'relative':
            self.connect(f'turb_compute_max_power.max_power', f'max_power_gt')
            #self.connect('vector_to_matrix_motor.max_power_em_mat', 'max_power_em')


        if size_motor:
            # =====================================================================
            # Rubber Motor Sizing: Scale baseline motor max power by rating ratio
            # =====================================================================
            reference_power_em = self.options['reference_power_em']
            
            # 1. Convert matrix inputs to vectors for MetaModel (same as fixed motor)
            self.add_subsystem('matrix_to_vector_motor', 
                                MatrixToVectorConverter(
                                    num_nodes=nn, 
                                    num_comps=npp * num_em,
                                    input_names=['rpm', 'voltage'],
                                    units={'rpm': 'rpm', 'voltage': 'V'},
                                    output_default=[1790, 800]
                                ), 
                                promotes_inputs=['*'], promotes_outputs=[])
            
            # 2. Compute baseline max power from empirical rpm/voltage map
            motor_rpm_volt_pow_interp = om.MetaModelStructuredComp(vec_size=nn * npp)
            motor_rpm_volt_pow_interp.add_input('rpm', 1790, training_data=self._vect_motor_rpm, units="rpm", shape=(nn * npp,))
            motor_rpm_volt_pow_interp.add_input('voltage', 850, training_data=self._vect_motor_voltage, units="V", shape=(nn * npp,))
            motor_rpm_volt_pow_interp.add_output('max_power_em_baseline_vect', 2300, training_data=self._motor_map_power_from_rpm_voltage, units="kW", shape=(nn * npp,))
            motor_rpm_volt_pow_interp.options['extrapolate'] = True
            self.add_subsystem('motor_rpm_volt_pow_interp', motor_rpm_volt_pow_interp)
            
            # 3. Convert baseline power vector back to matrix
            self.add_subsystem('vector_to_matrix_baseline', VectorToMatrixConverter(
                num_nodes=nn,
                num_comps=npp * num_em,
                input_names=['max_power_em_baseline_vect'],
                output_names=['max_power_em_baseline'],
                units={'max_power_em_baseline_vect': 'kW'}
            ), promotes_outputs=['max_power_em_baseline'])
            
            # Connect the MetaModel
            self.connect('matrix_to_vector_motor.rpm_vect', 'motor_rpm_volt_pow_interp.rpm')
            self.connect('matrix_to_vector_motor.voltage_vect', 'motor_rpm_volt_pow_interp.voltage')
            self.connect('motor_rpm_volt_pow_interp.max_power_em_baseline_vect', 'vector_to_matrix_baseline.max_power_em_baseline_vect')
            
            """
            # 4. Tile rated_power_em to matrix shape
            self.add_subsystem('tiler_rated_power_em', Tiler(
                num_nodes=nn,
                n_comps=npp,
                tile_option='matrix',
                input_names=['rated_power_em'],
                output_names=['rated_power_em_mat'],
                input_units={'rated_power_em': 'kW'},
                output_units={'rated_power_em_mat': 'kW'}
            ), promotes_inputs=['rated_power_em'], promotes_outputs=['rated_power_em_mat'])
            """


            self.add_subsystem('bcast_rated_power_em',
                ScalarToMatrixBroadcast(
                    num_nodes=nn,
                    num_comps=npp,
                    units='kW'),
                promotes_inputs=[('scalar', 'rated_power_em')],
                promotes_outputs=[("matrix", "rated_power_em_mat")])
            
            # 5. Scale baseline max power by (rated_power / reference_power)
            self.add_subsystem('rubber_motor_scaler', RubberMotorMaxPowerScaler(
                num_nodes=nn,
                num_props=npp,
                reference_power_em=reference_power_em
            ), promotes_inputs=['max_power_em_baseline'], 
               promotes_outputs=['max_power_em'])
            
            # Connect rated power matrix to scaler and motor throttle calculation
            self.connect('rated_power_em_mat', 'rubber_motor_scaler.rated_power_em')
            if power_spec == 'relative':
                self.connect("rated_power_em_mat", "motor_power_to_throttle.rated_power")
        
        else:
            # Convert matrix inputs to vectors for MetaModel
            self.add_subsystem('matrix_to_vector_motor', 
                                MatrixToVectorConverter(
                                    num_nodes=nn, 
                                    num_comps=npp * num_em,
                                    input_names=['rpm', 'voltage'],
                                    units={'rpm': 'rpm', 'voltage': 'V'},
                                    output_default=[1790, 800]
                                ), 
                                promotes_inputs=['*'], promotes_outputs=[])

        

            #max_pow_out = om.IndepVarComp()
            #max_pow_out.add_output('max_power_em', 2300 * np.ones((npp,nn)), units='kW')
            #self.add_subsystem('max_pow_out', max_pow_out, promotes_outputs=['*'])

            # Compute Max Power reduction from RPM and Voltage 
            motor_rpm_volt_pow_interp = om.MetaModelStructuredComp(vec_size=nn * npp)
            motor_rpm_volt_pow_interp.add_input('rpm', 1790, training_data=self._vect_motor_rpm, units="rpm", shape=(nn * npp,))
            motor_rpm_volt_pow_interp.add_input('voltage', 800, training_data=self._vect_motor_voltage, units="V", shape=(nn * npp,))
            motor_rpm_volt_pow_interp.add_output('max_power_em', 2300, training_data=self._motor_map_power_from_rpm_voltage, units="kW", shape=(nn * npp,))
            motor_rpm_volt_pow_interp.options['extrapolate'] = True
            self.add_subsystem('motor_rpm_volt_pow_interp', motor_rpm_volt_pow_interp)
            
            #Convert vector output back to matrix
            self.add_subsystem('vector_to_matrix_motor', VectorToMatrixConverter(
                num_nodes=nn,
                num_comps=npp * num_em,
                input_names=['max_power_em'],
                units={'max_power_em': 'kW'}
            ), promotes_outputs=[])
            
            # Connect the converters
            self.connect('matrix_to_vector_motor.rpm_vect', 'motor_rpm_volt_pow_interp.rpm')
            self.connect('matrix_to_vector_motor.voltage_vect', 'motor_rpm_volt_pow_interp.voltage')
            self.connect('motor_rpm_volt_pow_interp.max_power_em', 'vector_to_matrix_motor.max_power_em')
            # Motor rating is now provided externally - removed connection to max_power_em


        if power_spec == 'relative':
            # Add the appropriate JAX component based on rule and bias
            if rule == "fraction" and bias == "gt":
                self.add_subsystem("power_split", FractionGTBias(
                    num_nodes=nn,
                    num_props=npp,
                    num_em_per_nac=self.options["num_em_per_nac"],
                    num_turb_per_nac=self.options["num_turb_per_nac"],
                    prop_gt_gb_efficiency=self.options["prop_gt_gb_efficiency"],
                    prop_em_gb_efficiency=self.options["prop_em_gb_efficiency"]
                ), promotes=['*'])
            elif rule == "fraction" and bias == "em":
                self.add_subsystem("power_split", FractionEMBias(
                    num_nodes=nn,
                    num_props=npp,
                    num_em_per_nac=self.options["num_em_per_nac"],
                    num_turb_per_nac=self.options["num_turb_per_nac"],
                    prop_gt_gb_efficiency=self.options["prop_gt_gb_efficiency"],
                    prop_em_gb_efficiency=self.options["prop_em_gb_efficiency"]
                ), promotes=['*'])
            elif rule == "fixed" and bias == "gt":
                self.add_subsystem("power_split", FixedGTBias(
                    num_nodes=nn,
                    num_props=npp,
                    num_em_per_nac=self.options["num_em_per_nac"],
                    num_turb_per_nac=self.options["num_turb_per_nac"],
                    prop_gt_gb_efficiency=self.options["prop_gt_gb_efficiency"],
                    prop_em_gb_efficiency=self.options["prop_em_gb_efficiency"]
                ), promotes=['*'])
            elif rule == "fixed" and bias == "em":
                self.add_subsystem("power_split", FixedEMBias(
                    num_nodes=nn,
                    num_props=npp,
                    num_em_per_nac=self.options["num_em_per_nac"],
                    num_turb_per_nac=self.options["num_turb_per_nac"],
                    prop_gt_gb_efficiency=self.options["prop_gt_gb_efficiency"],
                    prop_em_gb_efficiency=self.options["prop_em_gb_efficiency"]
                ), promotes=['*'])

            self.connect("unit_mech_power_in_em", "motor_power_to_throttle.mech_power")
            #self.connect("max_power_em", "motor_power_to_throttle.rated_power")
            if not size_motor:
                self.connect("vector_to_matrix_motor.max_power_em_mat", "max_power_em")

        elif power_spec == 'independent':
            # Independent power specification - motor and GT controlled independently
            motor_command = self.options["motor_command"]
            
            if motor_command == 'throttle':
                # Motor throttle is input, compute power from throttle * max_power
                self.add_subsystem("motor_throttle_to_power", MotorThrottleToPowerMatrix(
                    num_nodes=nn,
                    num_props=npp
                ), promotes_inputs=['motor_throttle'], promotes_outputs=['unit_mech_power_calc_em'])
                
                # Connect max_power_em to the throttle-to-power component
                if size_motor:
                    self.connect("max_power_em", "motor_throttle_to_power.max_power_em")
                else:
                    self.connect("vector_to_matrix_motor.max_power_em_mat", "motor_throttle_to_power.max_power_em")
                    self.connect("vector_to_matrix_motor.max_power_em_mat", "max_power_em")

        #self.add_subsystem("component_sizing_margin", ComponentSizingMargin(
        #    num_nodes=nn,
        #    num_props=npp
        #), promotes_inputs=['*'], promotes_outputs=['*'])  

        # Matrix-based motor throttle calculation for all motors (for relative power_spec)
        if power_spec == 'relative':
            self.add_subsystem("motor_power_to_throttle", MotorPowerToThrottleMatrix(
                num_nodes=nn,
                num_props=npp
            ), promotes_inputs=[], promotes_outputs=[])  

            # Motor rating is now provided externally via motor_power_to_throttle.rated_power
            if not size_motor:
                self.connect("vector_to_matrix_motor.max_power_em_mat", "motor_power_to_throttle.rated_power")

            


class FractionGTBias(om.ExplicitComponent):
    """
    Component for fraction rule with GT bias.
    """
    
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_em_per_nac", default=1, desc="Number of electric motors in the nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines in the nacelle")
        self.options.declare("prop_gt_gb_efficiency", default=1.0, desc="Transmission efficiency from gas turbine to propeller")
        self.options.declare("prop_em_gb_efficiency", default=1.0, desc="Transmission efficiency from electric motor to propeller")
        self.options.declare("num_props", default=1, desc="Number of turbines in the nacelle")
    
    def setup(self):
        nn = self.options["num_nodes"]
        nt = self.options["num_turb_per_nac"]
        ne = self.options["num_em_per_nac"]
        npp = self.options["num_props"]
        
        # Inputs
        self.add_input("total_mech_power_out", units="kW", shape=(npp, nn))
        self.add_input("max_power_gt", units="kW", shape=(npp, nn))
        self.add_input("max_power_em", units="kW", shape=(npp, nn), val = 2300)
        self.add_input("power_split_fraction_gt", shape=(npp, nn))
        
        # Outputs
        self.add_output("unit_mech_power_in_gt", units="kW", shape=(npp, nn), lower = 1e-6, upper = 872)
        self.add_output("unit_mech_power_in_em", units="kW", shape=(npp, nn), lower = 1e-6, upper = 2500)
        self.add_output("power_shortage_post_corr", units="kW", shape=(npp, nn), desc="Power shortage after correction")
    
        # Declare partials with sparsity (element-wise dependence)
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('unit_mech_power_in_gt', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_fraction_gt'], rows=rows, cols=cols, method='exact')
        self.declare_partials('unit_mech_power_in_em', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_fraction_gt'], rows=rows, cols=cols, method='exact')
        self.declare_partials('power_shortage_post_corr', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_fraction_gt'], rows=rows, cols=cols, method='exact')
    
    
    def compute(self, inputs, outputs):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        
        total_mech_power_out = inputs['total_mech_power_out']
        max_power_gt = inputs['max_power_gt']
        max_power_em = inputs['max_power_em']
        power_split_fraction_gt = inputs['power_split_fraction_gt']
        
        # Original equations from the compute method, with fixes
        unit_mech_power_in_gt_unlim = total_mech_power_out * power_split_fraction_gt / nt / eta_gt
        unit_mech_power_in_gt_lim = np.where(unit_mech_power_in_gt_unlim > max_power_gt, max_power_gt, unit_mech_power_in_gt_unlim)
        unit_mech_power_in_em_unlim = (total_mech_power_out - unit_mech_power_in_gt_lim * eta_gt * nt) / nm / eta_em
        unit_mech_power_in_em_lim = np.where(unit_mech_power_in_em_unlim > max_power_em, max_power_em, unit_mech_power_in_em_unlim)

        # Correct for power delta if power rating is exceeded (only for GT bias)
        power_delta = unit_mech_power_in_gt_lim * eta_gt * nt + unit_mech_power_in_em_lim * eta_em * nm - total_mech_power_out
        unit_mech_power_in_gt_corr = unit_mech_power_in_gt_lim - power_delta / nt / eta_gt
        unit_mech_power_in_gt = np.where(unit_mech_power_in_gt_corr > max_power_gt, max_power_gt, unit_mech_power_in_gt_corr)

        power_shortage_post_corr = unit_mech_power_in_gt * eta_gt * nt + unit_mech_power_in_em_lim * eta_em * nm - total_mech_power_out
        
        outputs['unit_mech_power_in_gt'] = unit_mech_power_in_gt
        outputs['unit_mech_power_in_em'] = unit_mech_power_in_em_lim
        outputs['power_shortage_post_corr'] = power_shortage_post_corr        

    def compute_partials(self, inputs, partials):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        n = npp * nn
        
        # Flatten inputs for elementwise ops
        P = inputs['total_mech_power_out'].flatten()
        M_gt = inputs['max_power_gt'].flatten()
        M_em = inputs['max_power_em'].flatten()
        f_gt = inputs['power_split_fraction_gt'].flatten()
        
        # Recompute conditions to determine regimes (elementwise)
        gt_unlim = P * f_gt / nt / eta_gt
        condition1 = gt_unlim > M_gt
        gt_lim = np.where(condition1, M_gt, gt_unlim)
        em_unlim = (P - gt_lim * eta_gt * nt) / nm / eta_em
        condition2 = em_unlim > M_em
        em_lim = np.where(condition2, M_em, em_unlim)
        
        # Get the power delta for this operating condition
        power_delta = gt_lim * eta_gt * nt + em_lim * eta_em * nm - P
        gt_corr = gt_lim - power_delta / nt / eta_gt
        condition3 = gt_corr > M_gt
        
        # Regime masks
        r1 = ~condition1 & ~condition2  # No capping
        r2 = condition1 & ~condition2   # GT capping only
        r3 = ~condition1 & condition2 & ~condition3  # EM capping, correction <= GT max
        r4 = condition2 & condition3    # Both capping
        
        # Partials for unit_mech_power_in_gt
        gt_wrt_P = np.zeros(n)
        gt_wrt_P[r1] = f_gt[r1] / nt / eta_gt
        gt_wrt_P[r2] = 0.0
        gt_wrt_P[r3] = 1 / nt / eta_gt
        gt_wrt_P[r4] = 0.0
        
        gt_wrt_fgt = np.zeros(n)
        gt_wrt_fgt[r1] = P[r1] / nt / eta_gt
        gt_wrt_fgt[r2] = 0.0
        gt_wrt_fgt[r3] = 0.0
        gt_wrt_fgt[r4] = 0.0
        
        gt_wrt_Mgt = np.zeros(n)
        gt_wrt_Mgt[r1] = 0.0
        gt_wrt_Mgt[r2] = 1.0
        gt_wrt_Mgt[r3] = 0.0
        gt_wrt_Mgt[r4] = 1.0
        
        gt_wrt_Mem = np.zeros(n)
        gt_wrt_Mem[r1] = 0.0
        gt_wrt_Mem[r2] = 0.0
        gt_wrt_Mem[r3] = - eta_em * nm / nt / eta_gt
        gt_wrt_Mem[r4] = 0.0
        
        # Partials for unit_mech_power_in_em
        em_wrt_P = np.zeros(n)
        em_wrt_P[r1] = (1 - f_gt[r1]) / nm / eta_em
        em_wrt_P[r2] = 1 / nm / eta_em
        em_wrt_P[r3] = 0.0
        em_wrt_P[r4] = 0.0
        
        em_wrt_fgt = np.zeros(n)
        em_wrt_fgt[r1] = -P[r1] / nm / eta_em
        em_wrt_fgt[r2] = 0.0
        em_wrt_fgt[r3] = 0.0
        em_wrt_fgt[r4] = 0.0
        
        em_wrt_Mgt = np.zeros(n)
        em_wrt_Mgt[r1] = 0.0
        em_wrt_Mgt[r2] = -eta_gt * nt / nm / eta_em
        em_wrt_Mgt[r3] = 0.0
        em_wrt_Mgt[r4] = 0.0
        
        em_wrt_Mem = np.zeros(n)
        em_wrt_Mem[r1] = 0.0
        em_wrt_Mem[r2] = 0.0
        em_wrt_Mem[r3] = 1.0
        em_wrt_Mem[r4] = 1.0
        
        # Partials for power_shortage_post_corr
        short_wrt_P = np.zeros(n)
        short_wrt_P[r1 | r2 | r3] = 0.0
        short_wrt_P[r4] = -1.0
        
        short_wrt_fgt = np.zeros(n)
        short_wrt_fgt[r1 | r2 | r3 | r4] = 0.0  # Always 0
        
        short_wrt_Mgt = np.zeros(n)
        short_wrt_Mgt[r1 | r2 | r3] = 0.0
        short_wrt_Mgt[r4] = eta_gt * nt
        
        short_wrt_Mem = np.zeros(n)
        short_wrt_Mem[r1 | r2 | r3] = 0.0
        short_wrt_Mem[r4] = eta_em * nm
        
        # Assign to partials dict (diagonal elements since elementwise)
        partials['unit_mech_power_in_gt', 'total_mech_power_out'] = gt_wrt_P
        partials['unit_mech_power_in_gt', 'max_power_gt'] = gt_wrt_Mgt
        partials['unit_mech_power_in_gt', 'max_power_em'] = gt_wrt_Mem
        partials['unit_mech_power_in_gt', 'power_split_fraction_gt'] = gt_wrt_fgt
        
        partials['unit_mech_power_in_em', 'total_mech_power_out'] = em_wrt_P
        partials['unit_mech_power_in_em', 'max_power_gt'] = em_wrt_Mgt
        partials['unit_mech_power_in_em', 'max_power_em'] = em_wrt_Mem
        partials['unit_mech_power_in_em', 'power_split_fraction_gt'] = em_wrt_fgt
        
        partials['power_shortage_post_corr', 'total_mech_power_out'] = short_wrt_P
        partials['power_shortage_post_corr', 'max_power_gt'] = short_wrt_Mgt
        partials['power_shortage_post_corr', 'max_power_em'] = short_wrt_Mem
        partials['power_shortage_post_corr', 'power_split_fraction_gt'] = short_wrt_fgt
    

class FractionEMBias(om.ExplicitComponent):
    """
    Component for fraction rule with EM bias.
    """
    
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_em_per_nac", default=1, desc="Number of electric motors in the nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines in the nacelle")
        self.options.declare("prop_gt_gb_efficiency", default=1.0, desc="Transmission efficiency from gas turbine to propeller")
        self.options.declare("prop_em_gb_efficiency", default=1.0, desc="Transmission efficiency from electric motor to propeller")
        self.options.declare("num_props", default=1, desc="Number of turbines in the nacelle")
    
    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        
        # Inputs
        self.add_input("total_mech_power_out", units="kW", shape=(npp, nn))
        self.add_input("max_power_gt", units="kW", shape=(npp, nn))
        self.add_input("max_power_em", units="kW", shape=(npp, nn), val = 2500)
        self.add_input("power_split_fraction_em", shape=(npp, nn))
        
        # Outputs
        self.add_output("unit_mech_power_in_gt", units="kW", shape=(npp, nn))
        self.add_output("unit_mech_power_in_em", units="kW", shape=(npp, nn))
        self.add_output("power_shortage_post_corr", units="kW", shape=(npp, nn), desc="Power shortage after correction")
    
        # Declare partials with sparsity (element-wise dependence)
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('unit_mech_power_in_gt', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_fraction_em'], rows=rows, cols=cols, method='exact')
        self.declare_partials('unit_mech_power_in_em', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_fraction_em'], rows=rows, cols=cols, method='exact')
        self.declare_partials('power_shortage_post_corr', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_fraction_em'], rows=rows, cols=cols, method='exact')
    
    def compute(self, inputs, outputs):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        
        total_mech_power_out = inputs['total_mech_power_out']
        max_power_gt = inputs['max_power_gt']
        max_power_em = inputs['max_power_em']
        power_split_fraction_em = inputs['power_split_fraction_em']
        
        # Original equations from the compute method, with fixes
        unit_mech_power_in_em_unlim = total_mech_power_out * power_split_fraction_em / nm / eta_em
        unit_mech_power_in_em_lim = np.where(unit_mech_power_in_em_unlim > max_power_em, max_power_em, unit_mech_power_in_em_unlim)
        unit_mech_power_in_gt_unlim = (total_mech_power_out - unit_mech_power_in_em_lim * eta_em * nm) / nt / eta_gt
        unit_mech_power_in_gt = np.where(unit_mech_power_in_gt_unlim > max_power_gt, max_power_gt, unit_mech_power_in_gt_unlim)

        # Correct for power delta if power rating is exceeded (only for EM bias)
        power_delta = unit_mech_power_in_em_lim * eta_em * nm + unit_mech_power_in_gt * eta_gt * nt - total_mech_power_out
        unit_mech_power_in_em_corr = unit_mech_power_in_em_lim - power_delta / nm / eta_em
        unit_mech_power_in_em = np.where(unit_mech_power_in_em_corr > max_power_em, max_power_em, unit_mech_power_in_em_corr)

        power_shortage_post_corr = unit_mech_power_in_em * eta_em * nm + unit_mech_power_in_gt * eta_gt * nt - total_mech_power_out
        
        outputs['unit_mech_power_in_gt'] = unit_mech_power_in_gt
        outputs['unit_mech_power_in_em'] = unit_mech_power_in_em
        outputs['power_shortage_post_corr'] = power_shortage_post_corr
        
    def compute_partials(self, inputs, partials):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        n = npp * nn
        
        # Flatten inputs for elementwise ops
        P = inputs['total_mech_power_out'].flatten()
        M_gt = inputs['max_power_gt'].flatten()
        M_em = inputs['max_power_em'].flatten()
        f_em = inputs['power_split_fraction_em'].flatten()
        
        # Recompute conditions to determine regimes (elementwise)
        em_unlim = P * f_em / nm / eta_em
        condition1 = em_unlim > M_em  # EM uncapped exceeds max
        em_lim = np.where(condition1, M_em, em_unlim)
        gt_unlim = (P - em_lim * eta_em * nm) / nt / eta_gt
        condition2 = gt_unlim > M_gt  # GT uncapped exceeds max
        delta = em_lim * eta_em * nm + np.where(condition2, M_gt * eta_gt * nt, gt_unlim * eta_gt * nt) - P
        em_corr = em_lim - delta / nm / eta_em
        condition3 = em_corr > M_em  # Corrected EM exceeds max
        
        # Regime masks
        r1 = ~condition1 & ~condition2  # No capping
        r2 = condition1 & ~condition2   # EM capping only
        r3 = condition2 & ~condition3   # GT capping, correction <= EM max
        r4 = condition2 & condition3    # GT capping, correction > EM max (both capped)
        
        # Partials for unit_mech_power_in_gt
        gt_wrt_P = np.zeros(n)
        gt_wrt_P[r1] = (1 - f_em[r1]) / nt / eta_gt
        gt_wrt_P[r2] = 1 / nt / eta_gt
        gt_wrt_P[r3 | r4] = 0.0
        
        gt_wrt_fem = np.zeros(n)
        gt_wrt_fem[r1] = -P[r1] / nt / eta_gt
        gt_wrt_fem[r2] = 0.0
        gt_wrt_fem[r3 | r4] = 0.0
        
        gt_wrt_Mgt = np.zeros(n)
        gt_wrt_Mgt[r1] = 0.0
        gt_wrt_Mgt[r2] = 0.0
        gt_wrt_Mgt[r3 | r4] = 1.0
        
        gt_wrt_Mem = np.zeros(n)
        gt_wrt_Mem[r1] = 0.0
        gt_wrt_Mem[r2] = -eta_em * nm / nt / eta_gt
        gt_wrt_Mem[r3 | r4] = 0.0
        
        # Partials for unit_mech_power_in_em
        em_wrt_P = np.zeros(n)
        em_wrt_P[r1] = f_em[r1] / nm / eta_em
        em_wrt_P[r2] = 0.0
        em_wrt_P[r3] = 1 / nm / eta_em
        em_wrt_P[r4] = 0.0
        
        em_wrt_fem = np.zeros(n)
        em_wrt_fem[r1] = P[r1] / nm / eta_em
        em_wrt_fem[r2] = 0.0
        em_wrt_fem[r3] = 0.0
        em_wrt_fem[r4] = 0.0
        
        em_wrt_Mgt = np.zeros(n)
        em_wrt_Mgt[r1] = 0.0
        em_wrt_Mgt[r2] = 0.0
        em_wrt_Mgt[r3] = -eta_gt * nt / nm / eta_em
        em_wrt_Mgt[r4] = 0.0
        
        em_wrt_Mem = np.zeros(n)
        em_wrt_Mem[r1] = 0.0
        em_wrt_Mem[r2] = 1.0
        em_wrt_Mem[r3] = 0.0
        em_wrt_Mem[r4] = 1.0
        
        # Partials for power_shortage_post_corr
        short_wrt_P = np.zeros(n)
        short_wrt_P[r1 | r2 | r3] = 0.0
        short_wrt_P[r4] = -1.0
        
        short_wrt_fem = np.zeros(n)
        short_wrt_fem[r1 | r2 | r3 | r4] = 0.0  # Always 0
        
        short_wrt_Mgt = np.zeros(n)
        short_wrt_Mgt[r1 | r2 | r3] = 0.0
        short_wrt_Mgt[r4] = eta_gt * nt
        
        short_wrt_Mem = np.zeros(n)
        short_wrt_Mem[r1 | r2 | r3] = 0.0
        short_wrt_Mem[r4] = eta_em * nm
        
        # Assign to partials dict (diagonal elements since elementwise)
        partials['unit_mech_power_in_gt', 'total_mech_power_out'] = gt_wrt_P
        partials['unit_mech_power_in_gt', 'max_power_gt'] = gt_wrt_Mgt
        partials['unit_mech_power_in_gt', 'max_power_em'] = gt_wrt_Mem
        partials['unit_mech_power_in_gt', 'power_split_fraction_em'] = gt_wrt_fem  # Change 'power_split_fraction_gt' to this in setup if needed
        
        partials['unit_mech_power_in_em', 'total_mech_power_out'] = em_wrt_P
        partials['unit_mech_power_in_em', 'max_power_gt'] = em_wrt_Mgt
        partials['unit_mech_power_in_em', 'max_power_em'] = em_wrt_Mem
        partials['unit_mech_power_in_em', 'power_split_fraction_em'] = em_wrt_fem
        
        partials['power_shortage_post_corr', 'total_mech_power_out'] = short_wrt_P
        partials['power_shortage_post_corr', 'max_power_gt'] = short_wrt_Mgt
        partials['power_shortage_post_corr', 'max_power_em'] = short_wrt_Mem
        partials['power_shortage_post_corr', 'power_split_fraction_em'] = short_wrt_fem

class FixedGTBias(om.ExplicitComponent):
    """
    Component for fixed rule with GT bias.
    """
    
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_em_per_nac", default=1, desc="Number of electric motors in the nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines in the nacelle")
        self.options.declare("prop_gt_gb_efficiency", default=1.0, desc="Transmission efficiency from gas turbine to propeller")
        self.options.declare("prop_em_gb_efficiency", default=1.0, desc="Transmission efficiency from electric motor to propeller")
        self.options.declare("num_props", default=1, desc="Number of turbines in the nacelle")
    
    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        
        # Inputs
        self.add_input("total_mech_power_out", units="kW", shape=(npp, nn))
        self.add_input("max_power_gt", units="kW", shape=(npp, nn))
        self.add_input("max_power_em", units="kW", shape=(npp, nn), val = 2500)
        self.add_input("power_split_amount_gt", units="kW", shape=(npp, nn))
        # GT derate factor: 1.0 = full MCP, 0.5 = 50% of MCP
        # Scalar input that broadcasts to all nacelles/nodes
        self.add_input("gt_derate", val=1.0, shape=(1,), desc="GT power derate factor (0.3-1.0)")
        
        # Outputs
        self.add_output("unit_mech_power_in_gt", units="kW", shape=(npp, nn))
        self.add_output("unit_mech_power_in_em", units="kW", shape=(npp, nn))
        self.add_output("power_shortage_post_corr", units="kW", shape=(npp, nn), desc="Power shortage after correction")

        # Declare partials with sparsity (element-wise dependence)
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('unit_mech_power_in_gt', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_amount_gt'], rows=rows, cols=cols, method='exact')
        self.declare_partials('unit_mech_power_in_em', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_amount_gt'], rows=rows, cols=cols, method='exact')
        self.declare_partials('power_shortage_post_corr', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_amount_gt'], rows=rows, cols=cols, method='exact')
        # Partials for gt_derate - scalar input affects all outputs
        self.declare_partials('unit_mech_power_in_gt', 'gt_derate', rows=rows, cols=np.zeros(n, dtype=int), method='exact')
        self.declare_partials('unit_mech_power_in_em', 'gt_derate', rows=rows, cols=np.zeros(n, dtype=int), method='exact')
        self.declare_partials('power_shortage_post_corr', 'gt_derate', rows=rows, cols=np.zeros(n, dtype=int), method='exact')

    def compute(self, inputs, outputs):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        
        total_mech_power_out = inputs['total_mech_power_out']
        max_power_gt = inputs['max_power_gt']
        max_power_em = inputs['max_power_em']
        power_split_amount_gt = inputs['power_split_amount_gt']
        gt_derate = inputs['gt_derate'][0]  # Scalar input
        
        # Apply derate factor to GT power command
        effective_gt_power = power_split_amount_gt * gt_derate
        
        # Original equations from the compute method, with fixes
        unit_mech_power_in_gt_unlim = effective_gt_power / eta_gt / nt
        unit_mech_power_in_gt_lim = np.where(unit_mech_power_in_gt_unlim > max_power_gt, max_power_gt, unit_mech_power_in_gt_unlim)
        unit_mech_power_in_em_unlim = (total_mech_power_out - unit_mech_power_in_gt_lim * eta_gt * nt) / nm / eta_em
        unit_mech_power_in_em_lim = np.where(unit_mech_power_in_em_unlim > max_power_em, max_power_em, unit_mech_power_in_em_unlim)

        # Correct for power delta if power rating is exceeded (only for GT bias)
        power_delta = unit_mech_power_in_gt_lim * eta_gt * nt + unit_mech_power_in_em_lim * eta_em * nm - total_mech_power_out
        unit_mech_power_in_gt_corr = unit_mech_power_in_gt_lim - power_delta / nt / eta_gt
        unit_mech_power_in_gt = np.where(unit_mech_power_in_gt_corr > max_power_gt, max_power_gt, unit_mech_power_in_gt_corr)

        power_shortage_post_corr = unit_mech_power_in_gt * eta_gt * nt + unit_mech_power_in_em_lim * eta_em * nm - total_mech_power_out
        
        outputs['unit_mech_power_in_gt'] = unit_mech_power_in_gt
        outputs['unit_mech_power_in_em'] = unit_mech_power_in_em_lim
        outputs['power_shortage_post_corr'] = power_shortage_post_corr


    def compute_partials(self, inputs, partials):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        n = npp * nn
        
        # Flatten inputs for elementwise ops
        P = inputs['total_mech_power_out'].flatten()
        M_gt = inputs['max_power_gt'].flatten()
        M_em = inputs['max_power_em'].flatten()
        S_gt = inputs['power_split_amount_gt'].flatten()
        gt_derate = inputs['gt_derate'][0]  # Scalar
        
        # Effective GT power after derate
        S_gt_eff = S_gt * gt_derate
        
        # Recompute conditions to determine regimes (elementwise)
        gt_unlim = S_gt_eff / eta_gt / nt
        condition1 = gt_unlim > M_gt
        gt_lim = np.where(condition1, M_gt, gt_unlim)
        em_unlim = (P - gt_lim * eta_gt * nt) / nm / eta_em
        condition2 = em_unlim > M_em
        em_lim = np.where(condition2, M_em, em_unlim)
        
        # Get the power delta for this operating condition
        power_delta = gt_lim * eta_gt * nt + em_lim * eta_em * nm - P
        gt_corr = gt_lim - power_delta / nt / eta_gt
        condition3 = gt_corr > M_gt
        
        # Regime masks
        r1 = ~condition1 & ~condition2  # No capping
        r2 = condition1 & ~condition2   # GT capping only
        r3 = condition2 & ~condition3   # EM capping, correction <= GT max
        r4 = condition2 & condition3    # Both capping
        
        # Partials for unit_mech_power_in_gt
        gt_wrt_P = np.zeros(n)
        gt_wrt_P[r1] = 0.0
        gt_wrt_P[r2] = 0.0
        gt_wrt_P[r3] = 1 / nt / eta_gt
        gt_wrt_P[r4] = 0.0
        
        # d(gt)/d(S_gt) = d(gt)/d(S_gt_eff) * d(S_gt_eff)/d(S_gt) = d(gt)/d(S_gt_eff) * gt_derate
        gt_wrt_Sgt = np.zeros(n)
        gt_wrt_Sgt[r1] = gt_derate / nt / eta_gt
        gt_wrt_Sgt[r2] = 0.0
        gt_wrt_Sgt[r3] = 0.0
        gt_wrt_Sgt[r4] = 0.0
        
        gt_wrt_Mgt = np.zeros(n)
        gt_wrt_Mgt[r1] = 0.0
        gt_wrt_Mgt[r2] = 1.0
        gt_wrt_Mgt[r3] = 0.0
        gt_wrt_Mgt[r4] = 1.0
        
        gt_wrt_Mem = np.zeros(n)
        gt_wrt_Mem[r1] = 0.0
        gt_wrt_Mem[r2] = 0.0
        gt_wrt_Mem[r3] = - eta_em * nm / nt / eta_gt
        gt_wrt_Mem[r4] = 0.0
        
        # d(gt)/d(gt_derate) = d(gt)/d(S_gt_eff) * d(S_gt_eff)/d(gt_derate) = d(gt)/d(S_gt_eff) * S_gt
        gt_wrt_derate = np.zeros(n)
        gt_wrt_derate[r1] = S_gt[r1] / nt / eta_gt
        gt_wrt_derate[r2] = 0.0
        gt_wrt_derate[r3] = 0.0
        gt_wrt_derate[r4] = 0.0
        
        # Partials for unit_mech_power_in_em
        em_wrt_P = np.zeros(n)
        em_wrt_P[r1] = 1 / nm / eta_em
        em_wrt_P[r2] = 1 / nm / eta_em
        em_wrt_P[r3] = 0.0
        em_wrt_P[r4] = 0.0
        
        # d(em)/d(S_gt) - chain rule through gt_derate
        em_wrt_Sgt = np.zeros(n)
        em_wrt_Sgt[r1] = -gt_derate / nm / eta_em
        em_wrt_Sgt[r2] = 0.0
        em_wrt_Sgt[r3] = 0.0
        em_wrt_Sgt[r4] = 0.0
        
        em_wrt_Mgt = np.zeros(n)
        em_wrt_Mgt[r1] = 0.0
        em_wrt_Mgt[r2] = - eta_gt * nt / nm / eta_em
        em_wrt_Mgt[r3] = 0.0
        em_wrt_Mgt[r4] = 0.0
        
        em_wrt_Mem = np.zeros(n)
        em_wrt_Mem[r1] = 0.0
        em_wrt_Mem[r2] = 0.0
        em_wrt_Mem[r3] = 1.0
        em_wrt_Mem[r4] = 1.0
        
        # d(em)/d(gt_derate) = d(em)/d(S_gt_eff) * S_gt
        em_wrt_derate = np.zeros(n)
        em_wrt_derate[r1] = -S_gt[r1] / nm / eta_em
        em_wrt_derate[r2] = 0.0
        em_wrt_derate[r3] = 0.0
        em_wrt_derate[r4] = 0.0
        
        # Partials for power_shortage_post_corr
        short_wrt_P = np.zeros(n)
        short_wrt_P[r1 | r2 | r3] = 0.0
        short_wrt_P[r4] = -1.0
        
        short_wrt_Sgt = np.zeros(n)
        short_wrt_Sgt[r1 | r2 | r3 | r4] = 0.0  # Always 0
        
        short_wrt_Mgt = np.zeros(n)
        short_wrt_Mgt[r1 | r2 | r3] = 0.0
        short_wrt_Mgt[r4] = eta_gt * nt
        
        short_wrt_Mem = np.zeros(n)
        short_wrt_Mem[r1 | r2 | r3] = 0.0
        short_wrt_Mem[r4] = eta_em * nm
        
        short_wrt_derate = np.zeros(n)
        short_wrt_derate[r1 | r2 | r3 | r4] = 0.0  # Always 0 (no effect on shortage)
        
        # Assign to partials dict (diagonal elements since elementwise)
        partials['unit_mech_power_in_gt', 'total_mech_power_out'] = gt_wrt_P
        partials['unit_mech_power_in_gt', 'max_power_gt'] = gt_wrt_Mgt
        partials['unit_mech_power_in_gt', 'max_power_em'] = gt_wrt_Mem
        partials['unit_mech_power_in_gt', 'power_split_amount_gt'] = gt_wrt_Sgt
        partials['unit_mech_power_in_gt', 'gt_derate'] = gt_wrt_derate
        
        partials['unit_mech_power_in_em', 'total_mech_power_out'] = em_wrt_P
        partials['unit_mech_power_in_em', 'max_power_gt'] = em_wrt_Mgt
        partials['unit_mech_power_in_em', 'max_power_em'] = em_wrt_Mem
        partials['unit_mech_power_in_em', 'power_split_amount_gt'] = em_wrt_Sgt
        partials['unit_mech_power_in_em', 'gt_derate'] = em_wrt_derate
        
        partials['power_shortage_post_corr', 'total_mech_power_out'] = short_wrt_P
        partials['power_shortage_post_corr', 'max_power_gt'] = short_wrt_Mgt
        partials['power_shortage_post_corr', 'max_power_em'] = short_wrt_Mem
        partials['power_shortage_post_corr', 'power_split_amount_gt'] = short_wrt_Sgt
        partials['power_shortage_post_corr', 'gt_derate'] = short_wrt_derate
        
class FixedEMBias(om.ExplicitComponent):
    """
    Component for fixed rule with EM bias.
    """
    
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_em_per_nac", default=1, desc="Number of electric motors in the nacelle")
        self.options.declare("num_turb_per_nac", default=1, desc="Number of turbines in the nacelle")
        self.options.declare("prop_gt_gb_efficiency", default=1.0, desc="Transmission efficiency from gas turbine to propeller")
        self.options.declare("prop_em_gb_efficiency", default=1.0, desc="Transmission efficiency from electric motor to propeller")
        self.options.declare("num_props", default=1, desc="Number of turbines in the nacelle")
    
    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        
        # Inputs
        self.add_input("total_mech_power_out", units="kW", shape=(npp, nn))
        self.add_input("max_power_gt", units="kW", shape=(npp, nn))
        self.add_input("max_power_em", units="kW", shape=(npp, nn), val = 2500)
        self.add_input("power_split_amount_em", units="kW", shape=(npp, nn))
        
        # Outputs
        self.add_output("unit_mech_power_in_gt", units="kW", shape=(npp, nn))
        self.add_output("unit_mech_power_in_em", units="kW", shape=(npp, nn))
        self.add_output("power_shortage_post_corr", units="kW", shape=(npp, nn), desc="Power shortage after correction")

        # Declare partials with sparsity (element-wise dependence)
        n = npp * nn
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partials('unit_mech_power_in_gt', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_amount_em'], rows=rows, cols=cols, method='exact')
        self.declare_partials('unit_mech_power_in_em', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_amount_em'], rows=rows, cols=cols, method='exact')
        self.declare_partials('power_shortage_post_corr', ['total_mech_power_out', 'max_power_gt', 'max_power_em', 'power_split_amount_em'], rows=rows, cols=cols, method='exact')

    def compute(self, inputs, outputs):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        
        total_mech_power_out = inputs['total_mech_power_out']
        max_power_gt = inputs['max_power_gt']
        max_power_em = inputs['max_power_em']
        power_split_amount_em = inputs['power_split_amount_em']
        
        # Original equations from the compute method, with fixes
        # Electric Motor can't charge gas turbine....
        # Case where power requested from electric motor is greater than total power needed from propeller
        over_supply_idx = total_mech_power_out < power_split_amount_em
        unit_mech_power_in_gt = np.zeros_like(total_mech_power_out)
        unit_mech_power_in_em = np.zeros_like(total_mech_power_out)

        unit_mech_power_in_em_unlim = total_mech_power_out / nm / eta_em
        unit_mech_power_in_em_lim = np.where(unit_mech_power_in_em_unlim > max_power_em, max_power_em, unit_mech_power_in_em_unlim)
        unit_mech_power_in_em = np.where(over_supply_idx, unit_mech_power_in_em_lim, unit_mech_power_in_em)
        unit_mech_power_in_gt = np.where(over_supply_idx, 0.0, unit_mech_power_in_gt)

        # Case where power requested from electric motor is less than total power needed from propeller
        partial_supply_idx = total_mech_power_out >= power_split_amount_em
        unit_mech_power_in_em_unlim = power_split_amount_em / nm / eta_em
        unit_mech_power_in_em_lim = np.where(unit_mech_power_in_em_unlim > max_power_em, max_power_em, unit_mech_power_in_em_unlim)
        unit_mech_power_in_gt_unlim = (total_mech_power_out - unit_mech_power_in_em_lim * eta_em * nm) / nt / eta_gt
        unit_mech_power_in_gt = np.where(partial_supply_idx, np.where(unit_mech_power_in_gt_unlim > max_power_gt, max_power_gt, unit_mech_power_in_gt_unlim), unit_mech_power_in_gt)

        # Correct for power delta if power rating is exceeded (only for EM bias)
        power_delta = unit_mech_power_in_em_lim * eta_em * nm + unit_mech_power_in_gt * eta_gt * nt - total_mech_power_out
        unit_mech_power_in_em_corr = (unit_mech_power_in_em_lim * nm - power_delta / eta_em) / nm
        unit_mech_power_in_em = np.where(partial_supply_idx, np.where(unit_mech_power_in_em_corr > max_power_em, max_power_em, unit_mech_power_in_em_corr), unit_mech_power_in_em)

        power_shortage_post_corr = unit_mech_power_in_em * eta_em * nm + unit_mech_power_in_gt * eta_gt * nt - total_mech_power_out
        
        outputs['unit_mech_power_in_gt'] = unit_mech_power_in_gt
        outputs['unit_mech_power_in_em'] = unit_mech_power_in_em
        outputs['power_shortage_post_corr'] = power_shortage_post_corr

    def compute_partials(self, inputs, partials):
        nm = self.options["num_em_per_nac"]
        nt = self.options["num_turb_per_nac"]
        eta_gt = self.options["prop_gt_gb_efficiency"]
        eta_em = self.options["prop_em_gb_efficiency"]
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        n = npp * nn
        
        # Flatten inputs for elementwise ops
        P = inputs['total_mech_power_out'].flatten()
        M_gt = inputs['max_power_gt'].flatten()
        M_em = inputs['max_power_em'].flatten()
        S_em = inputs['power_split_amount_em'].flatten()
        
        # Recompute conditions to determine regimes (elementwise)
        over = P < S_em
        partial = ~over
        
        # For over_supply
        em_unlim_over = P / nm / eta_em
        condition1 = em_unlim_over > M_em
        em_lim_over = np.where(condition1, M_em, em_unlim_over)
        
        # For partial_supply
        em_unlim_partial = S_em / nm / eta_em
        condition2 = em_unlim_partial > M_em
        em_lim_partial = np.where(condition2, M_em, em_unlim_partial)
        gt_unlim_partial = (P - em_lim_partial * eta_em * nm) / nt / eta_gt
        condition3 = gt_unlim_partial > M_gt
        gt_lim_partial = np.where(condition3, M_gt, gt_unlim_partial)
        
        power_delta_partial = em_lim_partial * eta_em * nm + gt_lim_partial * eta_gt * nt - P
        em_corr_partial = em_lim_partial - power_delta_partial / eta_em / nm
        condition4 = em_corr_partial > M_em
        
        # Regime masks
        r1_over = over & ~condition1  # No capping in over
        r2_over = over & condition1   # EM capping in over
        r1_partial = partial & ~condition2 & ~condition3  # No capping in partial
        r2_partial = partial & condition2 & ~condition3   # EM capping only in partial
        r3_partial = partial & ~condition2 & condition3 & ~condition4  # GT capping, correction <= EM max in partial
        r4_partial = partial & condition3 & condition4    # Both capping in partial
        
        # Partials for unit_mech_power_in_em
        em_wrt_P = np.zeros(n)
        em_wrt_P[r1_over] = 1 / nm / eta_em
        em_wrt_P[r2_over] = 0.0
        em_wrt_P[r1_partial] = 0.0
        em_wrt_P[r2_partial] = 0.0
        em_wrt_P[r3_partial] = 1 / nm / eta_em
        em_wrt_P[r4_partial] = 0.0
        
        em_wrt_Sem = np.zeros(n)
        em_wrt_Sem[r1_over] = 0.0
        em_wrt_Sem[r2_over] = 0.0
        em_wrt_Sem[r1_partial] = 1 / nm / eta_em
        em_wrt_Sem[r2_partial] = 0.0
        em_wrt_Sem[r3_partial] = 0.0
        em_wrt_Sem[r4_partial] = 0.0
        
        em_wrt_Mgt = np.zeros(n)
        em_wrt_Mgt[r1_over] = 0.0
        em_wrt_Mgt[r2_over] = 0.0
        em_wrt_Mgt[r1_partial] = 0.0
        em_wrt_Mgt[r2_partial] = 0.0
        em_wrt_Mgt[r3_partial] = - eta_gt * nt / nm / eta_em
        em_wrt_Mgt[r4_partial] = 0.0
        
        em_wrt_Mem = np.zeros(n)
        em_wrt_Mem[r1_over] = 0.0
        em_wrt_Mem[r2_over] = 1.0
        em_wrt_Mem[r1_partial] = 0.0
        em_wrt_Mem[r2_partial] = 1.0
        em_wrt_Mem[r3_partial] = 0.0
        em_wrt_Mem[r4_partial] = 1.0
        
        # Partials for unit_mech_power_in_gt
        gt_wrt_P = np.zeros(n)
        gt_wrt_P[r1_over] = 0.0
        gt_wrt_P[r2_over] = 0.0
        gt_wrt_P[r1_partial] = 1 / nt / eta_gt
        gt_wrt_P[r2_partial] = 1 / nt / eta_gt
        gt_wrt_P[r3_partial] = 0.0
        gt_wrt_P[r4_partial] = 0.0
        
        gt_wrt_Sem = np.zeros(n)
        gt_wrt_Sem[r1_over] = 0.0
        gt_wrt_Sem[r2_over] = 0.0
        gt_wrt_Sem[r1_partial] = -1 / nt / eta_gt
        gt_wrt_Sem[r2_partial] = 0.0
        gt_wrt_Sem[r3_partial] = 0.0
        gt_wrt_Sem[r4_partial] = 0.0
        
        gt_wrt_Mgt = np.zeros(n)
        gt_wrt_Mgt[r1_over] = 0.0
        gt_wrt_Mgt[r2_over] = 0.0
        gt_wrt_Mgt[r1_partial] = 0.0
        gt_wrt_Mgt[r2_partial] = 0.0
        gt_wrt_Mgt[r3_partial] = 1.0
        gt_wrt_Mgt[r4_partial] = 1.0
        
        gt_wrt_Mem = np.zeros(n)
        gt_wrt_Mem[r1_over] = 0.0
        gt_wrt_Mem[r2_over] = 0.0
        gt_wrt_Mem[r1_partial] = 0.0
        gt_wrt_Mem[r2_partial] = - eta_em * nm / nt / eta_gt
        gt_wrt_Mem[r3_partial] = 0.0
        gt_wrt_Mem[r4_partial] = 0.0
        
        # Partials for power_shortage_post_corr
        short_wrt_P = np.zeros(n)
        short_wrt_P[r1_over] = 0.0
        short_wrt_P[r2_over] = -1.0
        short_wrt_P[r1_partial] = 0.0
        short_wrt_P[r2_partial] = 0.0
        short_wrt_P[r3_partial] = 0.0
        short_wrt_P[r4_partial] = -1.0
        
        short_wrt_Sem = np.zeros(n)
        # Always 0
        
        short_wrt_Mgt = np.zeros(n)
        short_wrt_Mgt[r1_over] = 0.0
        short_wrt_Mgt[r2_over] = 0.0
        short_wrt_Mgt[r1_partial] = 0.0
        short_wrt_Mgt[r2_partial] = 0.0
        short_wrt_Mgt[r3_partial] = 0.0
        short_wrt_Mgt[r4_partial] = eta_gt * nt
        
        short_wrt_Mem = np.zeros(n)
        short_wrt_Mem[r1_over] = 0.0
        short_wrt_Mem[r2_over] = eta_em * nm
        short_wrt_Mem[r1_partial] = 0.0
        short_wrt_Mem[r2_partial] = 0.0
        short_wrt_Mem[r3_partial] = 0.0
        short_wrt_Mem[r4_partial] = eta_em * nm
        
        # Assign to partials dict (diagonal elements since elementwise)
        partials['unit_mech_power_in_gt', 'total_mech_power_out'] = gt_wrt_P
        partials['unit_mech_power_in_gt', 'max_power_gt'] = gt_wrt_Mgt
        partials['unit_mech_power_in_gt', 'max_power_em'] = gt_wrt_Mem
        partials['unit_mech_power_in_gt', 'power_split_amount_em'] = gt_wrt_Sem
        
        partials['unit_mech_power_in_em', 'total_mech_power_out'] = em_wrt_P
        partials['unit_mech_power_in_em', 'max_power_gt'] = em_wrt_Mgt
        partials['unit_mech_power_in_em', 'max_power_em'] = em_wrt_Mem
        partials['unit_mech_power_in_em', 'power_split_amount_em'] = em_wrt_Sem
        
        partials['power_shortage_post_corr', 'total_mech_power_out'] = short_wrt_P
        partials['power_shortage_post_corr', 'max_power_gt'] = short_wrt_Mgt
        partials['power_shortage_post_corr', 'max_power_em'] = short_wrt_Mem
        partials['power_shortage_post_corr', 'power_split_amount_em'] = short_wrt_Sem

class ComponentSizingMargin(om.ExplicitComponent):
    """
    A component that computes the sizing margin for gas turbine and electric motor components.
    
    The sizing margin is calculated as the ratio of unit mechanical power to the rated power.
    A margin > 1 indicates the component is being used beyond its rated capacity.
    
    Inputs
    ------
    unit_mech_power_in_gt : float
        Unit mechanical power from gas turbine(s). (shape=(num_props, num_nodes), W)
    unit_mech_power_in_em : float
        Unit mechanical power from electric motor(s). (shape=(num_props, num_nodes), W)
    rated_power_gt : float
        Rated power of the gas turbine. (scalar, W)
    rated_power_em : float
        Rated power of the electric motor. (scalar, W)
    
    Outputs
    -------
    sizing_margin_gt : float
        Sizing margin for gas turbine (unit_power / rated_power). (shape=(num_props, num_nodes), dimensionless)
    sizing_margin_em : float
        Sizing margin for electric motor (unit_power / rated_power). (shape=(num_props, num_nodes), dimensionless)
    
    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    num_props : int
        Number of propulsors (default 1)
    """
    
    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of flight/control conditions")
        self.options.declare("num_props", default=1, desc="Number of props")
    
    def setup(self):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        
        # Inputs
        self.add_input("unit_mech_power_in_gt", units="W", desc="Unit mechanical power from gas turbine(s)", shape=(npp, nn))
        self.add_input("unit_mech_power_in_em", units="W", desc="Unit mechanical power from electric motor(s)", shape=(npp, nn))
        #self.add_input("rated_power_gt", units="W", desc="Gas turbine power rating", val=872.0)
        #self.add_input("rated_power_em", units="W", desc="Electric motor power rating", val=2300.0)
        
        # Outputs
        self.add_output("sizing_margin_gt", desc="Sizing margin for gas turbine", shape=(npp, nn))
        self.add_output("sizing_margin_em", desc="Sizing margin for electric motor", shape=(npp, nn))
    
        # Declare partials
        # For unit_mech_power -> sizing_margin: diagonal structure (element-wise)
        rows_diag = np.arange(npp * nn)
        cols_diag = np.arange(npp * nn)
        self.declare_partials('sizing_margin_gt', 'unit_mech_power_in_gt', rows=rows_diag, cols=cols_diag)
        self.declare_partials('sizing_margin_em', 'unit_mech_power_in_em', rows=rows_diag, cols=cols_diag)
        
        # For rated_power -> sizing_margin: scalar input affects all outputs
        # Output: (npp, nn) flattened = npp*nn elements
        # Input: scalar = 1 element
        # All output rows depend on column 0
        self.declare_partials('sizing_margin_gt', 'rated_power_gt', rows=np.arange(npp * nn), cols=np.zeros(npp * nn, dtype=int))
        self.declare_partials('sizing_margin_em', 'rated_power_em', rows=np.arange(npp * nn), cols=np.zeros(npp * nn, dtype=int))
    
    def compute(self, inputs, outputs):
        """
        Compute the sizing margins for both components.
        
        sizing_margin = unit_mech_power / rated_power
        """
        # Calculate sizing margins
        # unit_mech_power: (npp, nn), rated_power: scalar
        # Scalar automatically broadcasts to all elements
        outputs['sizing_margin_gt'] = inputs['unit_mech_power_in_gt'] / inputs['rated_power_gt']
        outputs['sizing_margin_em'] = inputs['unit_mech_power_in_em'] / inputs['rated_power_em']
    
    def compute_partials(self, inputs, partials):
        nn = self.options["num_nodes"]
        npp = self.options["num_props"]
        
        # d(sizing_margin_gt)/d(unit_mech_power_in_gt) = 1 / rated_power_gt
        # This is element-wise (diagonal). Scalar rated_power broadcasts to all elements.
        partials['sizing_margin_gt', 'unit_mech_power_in_gt'] = np.full(npp * nn, 1.0 / inputs['rated_power_gt'])
        
        # d(sizing_margin_gt)/d(rated_power_gt) = -unit_mech_power_in_gt / rated_power_gt^2
        # Scalar input affects all outputs. Flatten the (npp, nn) shaped derivative.
        partials['sizing_margin_gt', 'rated_power_gt'] = (-inputs['unit_mech_power_in_gt'] / inputs['rated_power_gt']**2).flatten()
        
        # d(sizing_margin_em)/d(unit_mech_power_in_em) = 1 / rated_power_em
        # This is element-wise (diagonal). Scalar rated_power broadcasts to all elements.
        partials['sizing_margin_em', 'unit_mech_power_in_em'] = np.full(npp * nn, 1.0 / inputs['rated_power_em'])
        
        # d(sizing_margin_em)/d(rated_power_em) = -unit_mech_power_in_em / rated_power_em^2
        # Scalar input affects all outputs. Flatten the (npp, nn) shaped derivative.
        partials['sizing_margin_em', 'rated_power_em'] = (-inputs['unit_mech_power_in_em'] / inputs['rated_power_em']**2).flatten()


def run_power_split_nacelle(rule='fixed', bias='em', plot_results=True):
    """
    Test function for PowerSplitNacelle component
    """
    import matplotlib.pyplot as plt
    
    print(f"Testing PowerSplitNacelle component with rule='{rule}', bias='{bias}'...")
    
    # Set up the test problem
    nn = 5
    npp = 4
    num_em_per_nac = 1
    num_turb_per_nac = 1
    model = om.Group()
    ivc = om.IndepVarComp()
    
    # Add independent variables
    ivc.add_output('total_mech_power_out', val=np.tile(np.array([1200, 1200, 1200, 1200, 1200]), (npp, 1)), units='kW', desc='Total power')
    #ivc.add_output('max_power_gt', val=2050.0 * np.ones(nn), units='kW', desc='Gas turbine rated power')
    ivc.add_output('rated_power_em', val=2300.0 * np.ones(nn), units='kW', desc='Electric motor rated power')
    ivc.add_output('rated_power_gt', val=872 * np.ones(nn), units='kW', desc='Gas turbine rated power')
    #ivc.add_output('power_rating_gt', val=2050.0, units='kW', desc='Gas turbine rated power')
    #ivc.add_output('max_power_em', val=1300.0 * np.ones(nn), units='kW', desc='Electric motor rated power')
    ivc.add_output('voltage', val=np.tile(800, (npp, nn)), units='V', desc='Motor voltage')
    ivc.add_output('rpm', val=np.tile(1790, (npp, nn)), units='rpm', desc='Motor speed')

    ivc.add_output('mach', val=np.array([0.2, 0.2, 0.3, 0.3, 0.3]), units=None, desc='Mach number')
    ivc.add_output('disa', val=np.array([0.0, 0.0, 0.0, 0.0, 0.0]), units='degC', desc='Disa')
    ivc.add_output('altitude', val=np.array([1000, 1000, 1000, 1000, 1000]), units='ft', desc='Altitude')

    if rule == 'fraction':
        if bias == 'gt':
            ivc.add_output('power_split_fraction_gt', val=np.tile(np.array([0.4, 0.3, 0.7, 0.5, 0.6]), (npp, 1)), desc='Gas Turbine power fraction')
        elif bias == 'em':
            ivc.add_output('power_split_fraction_em', val=np.tile(np.array([0.8, 0.5, 0.7, 0.9, 0.65]), (npp, 1)), desc='Electric Motor power fraction')
    elif rule == 'fixed':
        if bias == 'gt':
            ivc.add_output('power_split_amount_gt', val=np.tile(np.array([700, 600, 800, 300, 500]), (npp, 1)), units='kW', desc='Total Gas Turbine power contribution')
            ivc.add_output('gt_derate', val=0.9, units=None)
        elif bias == 'em':
            ivc.add_output('power_split_amount_em', val=np.tile(np.array([1000, 1100, 900, 800, 1000]), (npp, 1)), units='kW', desc='Total electric Motor power contribution')

    model.add_subsystem('ivc', ivc, promotes=['*'])
    
    # Test fraction rule with GT bias
    model.add_subsystem('power_split', PowerSplitNacelle(
        num_nodes=nn, 
        num_props=npp,
        num_em_per_nac=num_em_per_nac, 
        num_turb_per_nac=num_turb_per_nac,
        rule=rule, 
        bias=bias,
        prop_gt_gb_efficiency=1.0,
        prop_em_gb_efficiency=1.0
    ), promotes=['*'])
    
    prob = om.Problem(model, reports = False)

    #prob.model.connect("rated_power_em", ["motor_power_to_throttle.rated_power","max_power_em"])
    #prob.connect('altitude', 'lim_min_alt.input_array')
    #prob.model.connect("rated_power_em", "bcast_rated_power_em.rated_power_em")

    prob.setup(force_alloc_complex=True)
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)
    om.n2(prob)

    prob.set_val('max_power_em', 1800.0 * np.ones((npp, nn)), units='kW')
    
    # Run the model
    prob.run_model()
    
    # Get results
    gt_power = prob.get_val('unit_mech_power_in_gt', units='kW')
    em_power = prob.get_val('unit_mech_power_in_em', units='kW')
    power_shortage_post_corr = prob.get_val('power_shortage_post_corr', units='kW')
    
    print(f"\nResults (rule: {rule}, bias: {bias}):")
    print(f"Net Mechanical Power (kW): {prob.get_val('total_mech_power_out', units='kW')}")

    if rule == 'fraction':
        if bias == 'gt':
            print(f"GT Fraction: {prob.get_val('power_split_fraction_gt')}")
        elif bias == 'em':
            print(f"EM Fraction: {prob.get_val('power_split_fraction_em')}")
    elif rule == 'fixed':
        if bias == 'gt':
            print(f"GT Fixed: {prob.get_val('power_split_amount_gt')}")
        elif bias == 'em':
            print(f"EM Fixed: {prob.get_val('power_split_amount_em')}")
    print(f"GT Unit Power (kW): {gt_power}")
    print(f"EM Unit Power (kW): {em_power}")

    total_power = prob.get_val('total_mech_power_out', units='kW')

    if plot_results:

        # Create stacked area plot
        fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
        
        # Data points
        x = np.arange(nn)
        num_em = prob.model.power_split.power_split.options['num_em_per_nac']
        
        # Calculate individual motor powers
        em_power_per_motor = em_power 
        
        # Separate motor power into positive and negative arrays
        em_positive = np.where(em_power_per_motor >= 0, em_power_per_motor, 0)
        em_negative = np.where(em_power_per_motor < 0, em_power_per_motor, 0)
        
        # Use stackplot for positive powers (GT and positive EM)
        positive_powers = [gt_power]
        positive_labels = ['Gas Turbine Power Out']
        positive_colors = ['moccasin']
        
        # Add positive motor power to stack if any exists
        if np.any(em_positive > 0):
            positive_powers.append(em_positive)
            positive_labels.append('Electric Motor Power')
            positive_colors.append('lightblue')
        
        # Stack plot for positive powers
        ax1.stackplot(x, positive_powers, labels=positive_labels, colors=positive_colors, alpha=0.7)
        
        # Plot negative motor power as "Generated power" below zero
        if np.any(em_negative < 0):
            ax1.fill_between(x, 0, em_negative, alpha=0.7, label='Generated Power', color='lightgreen')
        
        # Add zero line for reference
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.2, alpha=0.5)
        
        # Plot total power line
        ax1.plot(x, total_power, 'k--', linewidth=1, label='Total Nacelle Power Out', alpha=0.8)
        
        ax1.set_xlabel('Analysis Point')
        ax1.set_ylabel('Power (kW)')
        ax1.set_title(f'Mechanical Power Distribution - {num_em} Electric Motors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        fig, (ax2) = plt.subplots(1, 1, figsize=(12, 10))
        ax2.fill_between(x, 0, power_shortage_post_corr, color='red', alpha=0.5, label='Power Shortage Post Correction')
        ax2.set_xlabel('Analysis Point')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title(f'Power Shortage Post Correction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x)

        plt.show()
        # end 
    
    #prob.check_partials(compact_print=True)
    
    # Calculate power split errors by comparing actual vs expected
    # Get efficiency parameters
    eta_gt = prob.model.power_split.power_split.options['prop_gt_gb_efficiency']
    eta_em = prob.model.power_split.power_split.options['prop_em_gb_efficiency']
    nt = prob.model.power_split.power_split.options['num_turb_per_nac']
    nm = prob.model.power_split.power_split.options['num_em_per_nac']
    
    # Calculate actual power outputs
    actual_gt_power = gt_power * eta_gt * nt
    actual_em_power = em_power * eta_em * nm
    
    # Calculate errors based on rule and bias
    errors = {}
    
    if rule == 'fraction':
        if bias == 'gt':
            fraction_gt = prob.get_val('power_split_fraction_gt')
            expected_gt_power = total_power * fraction_gt
            expected_em_power = total_power * (1 - fraction_gt)
            
            gt_error_abs = np.abs(actual_gt_power - expected_gt_power)
            em_error_abs = np.abs(actual_em_power - expected_em_power)
            
            gt_error_rel = gt_error_abs / expected_gt_power
            em_error_rel = em_error_abs / expected_em_power
            
            errors = {
                'gt_absolute_error': gt_error_abs,
                'em_absolute_error': em_error_abs,
                'gt_relative_error': gt_error_rel,
                'em_relative_error': em_error_rel,
                'max_gt_abs_error': gt_error_abs.max(),
                'max_em_abs_error': em_error_abs.max(),
                'max_gt_rel_error': gt_error_rel.max(),
                'max_em_rel_error': em_error_rel.max(),
                'power_shortage': prob.get_val('power_shortage_post_corr', units='kW'),
                'max_power_shortage': prob.get_val('power_shortage_post_corr', units='kW').max()
            }
            
        elif bias == 'em':
            fraction_em = prob.get_val('power_split_fraction_em')
            expected_em_power = total_power * fraction_em
            expected_gt_power = total_power * (1 - fraction_em)
            
            gt_error_abs = np.abs(actual_gt_power - expected_gt_power)
            em_error_abs = np.abs(actual_em_power - expected_em_power)
            
            gt_error_rel = gt_error_abs / expected_gt_power
            em_error_rel = em_error_abs / expected_em_power
            
            errors = {
                'gt_absolute_error': gt_error_abs,
                'em_absolute_error': em_error_abs,
                'gt_relative_error': gt_error_rel,
                'em_relative_error': em_error_rel,
                'max_gt_abs_error': gt_error_abs.max(),
                'max_em_abs_error': em_error_abs.max(),
                'max_gt_rel_error': gt_error_rel.max(),
                'max_em_rel_error': em_error_rel.max(),
                'power_shortage': prob.get_val('power_shortage_post_corr', units='kW'),
                'max_power_shortage': prob.get_val('power_shortage_post_corr', units='kW').max()
            }
            
    elif rule == 'fixed':
        if bias == 'gt':
            amount_gt = prob.get_val('power_split_amount_gt', units='kW')
            gt_error_abs = np.abs(actual_gt_power - amount_gt)
            gt_error_rel = gt_error_abs / amount_gt
            
            errors = {
                'gt_absolute_error': gt_error_abs,
                'gt_relative_error': gt_error_rel,
                'max_gt_abs_error': gt_error_abs.max(),
                'max_gt_rel_error': gt_error_rel.max(),
                'power_shortage': prob.get_val('power_shortage_post_corr', units='kW'),
                'max_power_shortage': prob.get_val('power_shortage_post_corr', units='kW').max()
            }
            
        elif bias == 'em':
            amount_em = prob.get_val('power_split_amount_em', units='kW')
            em_error_abs = np.abs(actual_em_power - amount_em)
            em_error_rel = em_error_abs / amount_em
            
            errors = {
                'em_absolute_error': em_error_abs,
                'em_relative_error': em_error_rel,
                'max_em_abs_error': em_error_abs.max(),
                'max_em_rel_error': em_error_rel.max(),
                'power_shortage': prob.get_val('power_shortage_post_corr', units='kW'),
                'max_power_shortage': prob.get_val('power_shortage_post_corr', units='kW').max()
            }
    
    # Return the error values
    return errors


def test_fraction_gt_bias_exact():
    """Test that FractionGTBias exactly follows the fraction when no power shortage"""
    
    print("Testing FractionGTBias - Exact Fraction Compliance")
    print("-" * 50)
    
    # Run the test with fraction GT bias
    errors = run_power_split_nacelle(rule='fraction', bias='gt', plot_results=False)
    
    print(f"Power shortage: {errors['power_shortage']}")
    print(f"Max power shortage: {errors['max_power_shortage']:.8f} kW")
    print(f"GT max absolute error: {errors['max_gt_abs_error']:.8f} kW")
    print(f"EM max absolute error: {errors['max_em_abs_error']:.8f} kW")
    print(f"GT max relative error: {errors['max_gt_rel_error']:.8f}")
    print(f"EM max relative error: {errors['max_em_rel_error']:.8f}")
    
    # Check exact compliance - either power shortage is zero OR errors are zero
    power_shortage_ok = np.all(np.abs(errors['power_shortage']) < 1e-6)
    gt_error_ok = errors['max_gt_abs_error'] < 1e-6
    em_error_ok = errors['max_em_abs_error'] < 1e-6
    
    assert power_shortage_ok or (gt_error_ok and em_error_ok), \
        f"Neither power shortage zero (max: {errors['max_power_shortage']:.8f} kW) nor errors zero (GT: {errors['max_gt_abs_error']:.8f} kW, EM: {errors['max_em_abs_error']:.8f} kW)"
    
    print("✓ FractionGTBias exact compliance verified")
    return True


def test_fraction_em_bias_exact():
    """Test that FractionEMBias exactly follows the fraction when no power shortage"""
    
    print("\nTesting FractionEMBias - Exact Fraction Compliance")
    print("-" * 50)
    
    # Run the test with fraction EM bias
    errors = run_power_split_nacelle(rule='fraction', bias='em', plot_results=False)
    
    print(f"Power shortage: {errors['power_shortage']}")
    print(f"Max power shortage: {errors['max_power_shortage']:.8f} kW")
    print(f"GT max absolute error: {errors['max_gt_abs_error']:.8f} kW")
    print(f"EM max absolute error: {errors['max_em_abs_error']:.8f} kW")
    print(f"GT max relative error: {errors['max_gt_rel_error']:.8f}")
    print(f"EM max relative error: {errors['max_em_rel_error']:.8f}")
    
    # Check exact compliance - either power shortage is zero OR errors are zero
    power_shortage_ok = np.all(np.abs(errors['power_shortage']) < 1e-6)
    gt_error_ok = errors['max_gt_abs_error'] < 1e-6
    em_error_ok = errors['max_em_abs_error'] < 1e-6
    
    assert power_shortage_ok or (gt_error_ok and em_error_ok), \
        f"Neither power shortage zero (max: {errors['max_power_shortage']:.8f} kW) nor errors zero (GT: {errors['max_gt_abs_error']:.8f} kW, EM: {errors['max_em_abs_error']:.8f} kW)"
    
    print("✓ FractionEMBias exact compliance verified")
    return True


def test_fixed_gt_bias_exact():
    """Test that FixedGTBias exactly follows the fixed amount when no power shortage"""
    
    print("\nTesting FixedGTBias - Exact Fixed Amount Compliance")
    print("-" * 50)
    
    # Run the test with fixed GT bias
    errors = run_power_split_nacelle(rule='fixed', bias='gt', plot_results=False)
    
    print(f"Power shortage: {errors['power_shortage']}")
    print(f"Max power shortage: {errors['max_power_shortage']:.8f} kW")
    print(f"GT max absolute error: {errors['max_gt_abs_error']:.8f} kW")
    print(f"GT max relative error: {errors['max_gt_rel_error']:.8f}")
    
    # Check exact compliance - either power shortage is zero OR errors are zero
    power_shortage_ok = np.all(np.abs(errors['power_shortage']) < 1e-6)
    gt_error_ok = errors['max_gt_abs_error'] < 1e-6
    
    assert power_shortage_ok or gt_error_ok, \
        f"Neither power shortage zero (max: {errors['max_power_shortage']:.8f} kW) nor GT error zero (max: {errors['max_gt_abs_error']:.8f} kW)"
    
    print("✓ FixedGTBias exact compliance verified")
    return True


def test_fixed_em_bias_exact():
    """Test that FixedEMBias exactly follows the fixed amount when no power shortage"""
    
    print("\nTesting FixedEMBias - Exact Fixed Amount Compliance")
    print("-" * 50)
    
    # Run the test with fixed EM bias
    errors = run_power_split_nacelle(rule='fixed', bias='em', plot_results=False)
    
    print(f"Power shortage: {errors['power_shortage']}")
    print(f"Max power shortage: {errors['max_power_shortage']:.8f} kW")
    print(f"EM max absolute error: {errors['max_em_abs_error']:.8f} kW")
    print(f"EM max relative error: {errors['max_em_rel_error']:.8f}")
    
    # Check exact compliance - either power shortage is zero OR errors are zero
    power_shortage_ok = np.all(np.abs(errors['power_shortage']) < 1e-6)
    em_error_ok = errors['max_em_abs_error'] < 1e-6
    
    assert power_shortage_ok or em_error_ok, \
        f"Neither power shortage zero (max: {errors['max_power_shortage']:.8f} kW) nor EM error zero (max: {errors['max_em_abs_error']:.8f} kW)"
    
    print("✓ FixedEMBias exact compliance verified")
    return True


def run_all_tests():
    """Test all nacelle splitter configurations for exact compliance when no power shortage"""
    
    print("NACELLE SPLITTER EXACT COMPLIANCE TEST SUITE")
    print("=" * 60)
    print("Testing that power split rules are exactly followed when power_shortage_post_corr = 0")
    print("=" * 60)
    
    test_functions = [
        test_fraction_gt_bias_exact,
        test_fraction_em_bias_exact,
        test_fixed_gt_bias_exact,
        test_fixed_em_bias_exact
    ]
    
    results = {}
    
    for test_func in test_functions:
        try:
            result = test_func()
            results[test_func.__name__] = result
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")
            results[test_func.__name__] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("EXACT COMPLIANCE TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30} : {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All exact compliance tests passed!")
        print("✓ Power split rules are exactly followed when no power shortage")
    else:
        print("❌ Some exact compliance tests failed!")
        print("✗ Power split rules are not exactly followed")
    
    return results


if __name__ == "__main__":
    # Run exact compliance tests
    run_all_tests()
