import openmdao.api as om
import numpy as np


class PayloadWeight(om.ExplicitComponent):
    """
    Calculates payload weight including passengers and cargo.
    
    Payload consists of:
    1. Passenger weight: passenger body weight + carry-on baggage
       - Each passenger: 195 lbs body weight
       - Each passenger: 30 lbs baggage allowance
       - Carry-on baggage: (1 - cargo_fraction) * num_passengers * 30
       - Total passenger weight = 195 * num_passengers + (1 - cargo_fraction) * num_passengers * 30
    
    2. Cargo weight: checked baggage in cargo hold
       - Checked baggage: cargo_fraction * num_passengers * 30
    
    The cargo_fraction represents the portion of baggage that goes into the cargo hold
    (the rest goes as carry-on in overhead bins).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to evaluate')
        self.options.declare('passenger_weight', default=195.0, types=float,
                           desc='Weight per passenger in lbm (body weight)')
        self.options.declare('baggage_allowance', default=30.0, types=float,
                           desc='Baggage allowance per passenger in lbm')
        self.options.declare('cargo_fraction', default=0.25, types=float,
                           desc='Fraction of baggage that goes to cargo hold (rest is carry-on)')
        self.options.declare('fs_start', default=382.3, types=float,
                           desc='Fuselage Station at start of fuselage (in)')
        self.options.declare('cg_cargo_percentage', default=77.3, types=float,
                           desc='Cargo C.G location as percentage of fuselage length')
        self.options.declare('oweed_row_offset', default=25.0, types=float,
                           desc='OWEED row C.G offset from front spar location (in)')
        self.options.declare('first_row_offset', default=35.0, types=float,
                           desc='First row forward/aft offset from OWEED row (in)')
        self.options.declare('passengers_per_row', default=4.0, types=float,
                           desc='Number of passengers per row')

    def setup(self):
        self.add_input('num_passengers', val=76, units=None, desc='Number of passengers')
        self.add_input('fuselage_length', val=97.89, units='ft', desc='Fuselage length (ft)')
        self.add_input('front_spar_location', val=0.0, units='inch', desc='Front spar location (Fuselage Station)')
        self.add_input('num_rows_fwd', val=0, units=None, desc='Number of rows forward of OWEED')
        self.add_input('pitch', val=32.0, units='inch', desc='Seat pitch (spacing between rows)')
        
        self.add_output('passenger_weight', val=0.0, units='lbm', 
                       desc='Total passenger weight (body + carry-on baggage)')
        self.add_output('cargo_weight', val=0.0, units='lbm', 
                       desc='Total cargo weight (checked baggage)')
        self.add_output('payload_weight', val=0.0, units='lbm', 
                       desc='Total payload weight (passenger + cargo)')
        self.add_output('cg_cargo', val=0.0, units='inch', 
                       desc='Cargo C.G location (Fuselage Station)')
        self.add_output('cg_passenger', val=0.0, units='inch', 
                       desc='Passenger + carry-on C.G location (Fuselage Station)')
        
        # Declare partials
        self.declare_partials('passenger_weight', 'num_passengers')
        self.declare_partials('cargo_weight', 'num_passengers')
        self.declare_partials('payload_weight', 'num_passengers')
        self.declare_partials('cg_cargo', 'fuselage_length')
        self.declare_partials('cg_passenger', ['front_spar_location', 'num_rows_fwd', 'pitch', 'num_passengers'])

    def compute(self, inputs, outputs):
        # Input validation
        num_pax = inputs['num_passengers']
        if num_pax <= 0:
            raise ValueError(f"num_passengers must be positive, got {num_pax}")
        
        fuselage_length_ft = inputs['fuselage_length']
        if fuselage_length_ft <= 0:
            raise ValueError(f"fuselage_length must be positive, got {fuselage_length_ft}")
        
        front_spar = inputs['front_spar_location']
        num_rows_fwd = inputs['num_rows_fwd']
        pitch = inputs['pitch']
        if pitch <= 0:
            raise ValueError(f"pitch must be positive, got {pitch}")
        
        passenger_body_weight = self.options['passenger_weight']
        baggage_allowance = self.options['baggage_allowance']
        cargo_fraction = self.options['cargo_fraction']
        passengers_per_row = self.options['passengers_per_row']
        oweed_row_offset = self.options['oweed_row_offset']
        first_row_offset = self.options['first_row_offset']
        
        # Passenger weight = body weight + carry-on baggage
        # Carry-on = (1 - cargo_fraction) of total baggage allowance
        carry_on_per_pax = (1.0 - cargo_fraction) * baggage_allowance
        outputs['passenger_weight'] = num_pax * (passenger_body_weight + carry_on_per_pax)
        
        # Cargo weight = checked baggage in cargo hold
        # Checked baggage = cargo_fraction of total baggage allowance
        checked_baggage_per_pax = cargo_fraction * baggage_allowance
        outputs['cargo_weight'] = num_pax * checked_baggage_per_pax
        
        # Total payload weight
        outputs['payload_weight'] = outputs['passenger_weight'] + outputs['cargo_weight']
        
        # Calculate Cargo C.G location
        # Use fuselage_length directly (computed by FuselageParameterLinks from num_passengers)
        fuselage_length_in = fuselage_length_ft * 12.0  # Convert to inches
        fs_start = self.options['fs_start']
        cg_percentage = self.options['cg_cargo_percentage']
        outputs['cg_cargo'] = fs_start + (fuselage_length_in * cg_percentage / 100.0)
        
        # Calculate Passenger + Carry-on C.G location
        # 1st OWEED passenger C.G = front_spar_location + oweed_row_offset
        first_oweed_cg = front_spar + oweed_row_offset
        
        # Use continuous values (no int() casting for optimization)
        num_rows_fwd_val = num_rows_fwd
        num_pax_val = num_pax
        pitch_val = pitch
        
        # Calculate total number of rows
        total_rows = num_pax_val / passengers_per_row
        
        # Build array of C.G locations for each row
        row_cgs = []
        
        # Forward rows: starting from OWEED and going forward
        # Forward Row 1 is immediately forward of OWEED at -first_row_offset
        # Forward Row 2 is next forward at -first_row_offset - pitch, etc.
        if num_rows_fwd_val > 0:
            # Build forward rows from closest to OWEED (Row 1) to farthest forward
            for i in range(int(num_rows_fwd_val)):
                # Row 1 (i=0): -first_row_offset, Row 2 (i=1): -first_row_offset - pitch, etc.
                row_cgs.append(first_oweed_cg - first_row_offset - i * pitch_val)
        
        # OWEED row (reference point)
        row_cgs.append(first_oweed_cg)
        
        # Aft rows: starting from OWEED and going aft
        # Aft Row 1 is immediately aft of OWEED at +first_row_offset
        # Aft Row 2 is next aft at +first_row_offset + pitch, etc.
        row_cgs.append(first_oweed_cg + first_row_offset)  # Aft Row 1
        
        # Remaining aft rows: total_rows - (num_rows_fwd + 2)
        # The +2 accounts for: OWEED row + 1 row aft at +first_row_offset
        num_aft_rows_remaining = int(total_rows - (num_rows_fwd_val + 2))
        for i in range(1, num_aft_rows_remaining + 1):
            # Aft Row 2 (i=1): +first_row_offset + pitch, Aft Row 3 (i=2): +first_row_offset + 2*pitch, etc.
            row_cgs.append(first_oweed_cg + first_row_offset + i * pitch_val)
        
        # Sort rows by C.G location (from most forward to most aft) for averaging
        row_cgs_sorted = sorted(row_cgs)
        
        # Calculate average C.G (all rows have equal weight, 4 passengers each)
        if len(row_cgs_sorted) > 0:
            outputs['cg_passenger'] = np.mean(row_cgs_sorted)
        else:
            outputs['cg_passenger'] = first_oweed_cg

    def compute_partials(self, inputs, partials):
        passenger_body_weight = self.options['passenger_weight']
        baggage_allowance = self.options['baggage_allowance']
        cargo_fraction = self.options['cargo_fraction']
        passengers_per_row = self.options['passengers_per_row']
        first_row_offset = self.options['first_row_offset']
        
        # Partials for weight outputs w.r.t. num_passengers
        carry_on_per_pax = (1.0 - cargo_fraction) * baggage_allowance
        checked_baggage_per_pax = cargo_fraction * baggage_allowance
        
        partials['passenger_weight', 'num_passengers'] = passenger_body_weight + carry_on_per_pax
        partials['cargo_weight', 'num_passengers'] = checked_baggage_per_pax
        partials['payload_weight', 'num_passengers'] = passenger_body_weight + baggage_allowance
        
        # Partials for C.G: cg = fs_start + (fuselage_length * 12 * percentage/100)
        cg_percentage = self.options['cg_cargo_percentage']
        partials['cg_cargo', 'fuselage_length'] = 12.0 * cg_percentage / 100.0
        
        # Partials for passenger C.G
        # d(cg_passenger)/d(front_spar_location) = 1.0 (all row C.Gs shift by same amount)
        partials['cg_passenger', 'front_spar_location'] = 1.0
        
        num_rows_fwd_val = inputs['num_rows_fwd']
        num_pax_val = inputs['num_passengers']
        pitch_val = inputs['pitch']
        
        # Use continuous total_rows for smooth gradients
        total_rows = num_pax_val / passengers_per_row
        
        # d(cg_passenger)/d(num_rows_fwd): surrogate gradient for optimization
        # When num_rows_fwd increases by 1, we add a forward row and lose an aft row
        # New forward row at: first_oweed_cg - first_row_offset - num_rows_fwd * pitch
        # Lost aft row was at: first_oweed_cg + first_row_offset + (num_aft_remaining) * pitch
        # Since num_aft_remaining = total_rows - num_rows_fwd - 2:
        # Change in sum = -2*first_row_offset - (total_rows - 2)*pitch
        # Change in mean = change_in_sum / total_rows
        if total_rows > 0:
            partials['cg_passenger', 'num_rows_fwd'] = (-2.0 * first_row_offset - (total_rows - 2.0) * pitch_val) / total_rows
        else:
            partials['cg_passenger', 'num_rows_fwd'] = 0.0
        
        # d(cg_passenger)/d(pitch): affects spacing of forward and aft rows
        # Use continuous approximation for num_aft_rows_remaining
        num_aft_rows_remaining = total_rows - (num_rows_fwd_val + 2.0)
        
        if total_rows > 0:
            # Forward rows contribution: row i at -first_row_offset - i*pitch, d/dpitch = -i
            # Sum for i=0 to n_fwd-1: sum(-i) = -n_fwd*(n_fwd-1)/2
            forward_contrib = -num_rows_fwd_val * (num_rows_fwd_val - 1.0) / 2.0
            
            # Aft rows contribution: row i at +first_row_offset + i*pitch, d/dpitch = +i  
            # Sum for i=1 to n_aft_rem: sum(i) = n_aft_rem*(n_aft_rem+1)/2
            aft_contrib = num_aft_rows_remaining * (num_aft_rows_remaining + 1.0) / 2.0
            
            partials['cg_passenger', 'pitch'] = (forward_contrib + aft_contrib) / total_rows
        else:
            partials['cg_passenger', 'pitch'] = 0.0
        
        # d(cg_passenger)/d(num_passengers): surrogate gradient for optimization
        # The compute uses int() for row counts (step function), but we use
        # a continuous approximation for gradient-based optimization.
        # When more passengers are added, more aft rows are added, shifting mean C.G. aft.
        if total_rows > 0:
            n_aft_rem = total_rows - num_rows_fwd_val - 2.0
            partials['cg_passenger', 'num_passengers'] = (pitch_val / passengers_per_row) * (n_aft_rem / total_rows) if n_aft_rem > 0 else 0.0
        else:
            partials['cg_passenger', 'num_passengers'] = 0.0


if __name__ == "__main__":
    prob = om.Problem(reports=False)
    model = prob.model

    ivc = model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('num_passengers', val=76, units=None)
    ivc.add_output('fuselage_length', val=97.89, units='ft')
    ivc.add_output('front_spar_location', val=948.0, units='inch')
    ivc.add_output('num_rows_fwd', val=9, units=None)
    ivc.add_output('pitch', val=32.0, units='inch')

    model.add_subsystem('payload', PayloadWeight(), promotes=['*'])

    prob.setup()
    prob.run_model()

    # Get values
    num_pax = int(prob.get_val('num_passengers')[0])
    num_rows_fwd = int(prob.get_val('num_rows_fwd')[0])
    pitch = prob.get_val('pitch', units='inch')[0]
    front_spar = prob.get_val('front_spar_location', units='inch')[0]
    first_oweed_cg = front_spar + 25.0
    total_rows = num_pax / 4.0
    num_aft_rows_remaining = int(total_rows - (num_rows_fwd + 2))
    
    # Print summary
    print("\n" + "=" * 70)
    print("PAYLOAD WEIGHT AND C.G SUMMARY")
    print("=" * 70)
    print(f"Passenger Weight:            {prob.get_val('passenger_weight', units='lbm')[0]:.2f} lbm")
    print(f"Passenger C.G:               {prob.get_val('cg_passenger', units='inch')[0]:.2f} in")
    print(f"Cargo Weight:                {prob.get_val('cargo_weight', units='lbm')[0]:.2f} lbm")
    print(f"Cargo C.G:                   {prob.get_val('cg_cargo', units='inch')[0]:.2f} in")
    print(f"Total Payload Weight:        {prob.get_val('payload_weight', units='lbm')[0]:.2f} lbm")
    
    # Calculate total payload C.G (weighted average)
    passenger_weight = prob.get_val('passenger_weight', units='lbm')[0]
    cargo_weight = prob.get_val('cargo_weight', units='lbm')[0]
    passenger_cg = prob.get_val('cg_passenger', units='inch')[0]
    cargo_cg = prob.get_val('cg_cargo', units='inch')[0]
    total_payload_weight = prob.get_val('payload_weight', units='lbm')[0]
    total_payload_cg = (passenger_weight * passenger_cg + cargo_weight * cargo_cg) / total_payload_weight if total_payload_weight > 0 else 0.0
    
    print(f"Total Payload C.G:            {total_payload_cg:.2f} in")
    print("=" * 70 + "\n")
    
    prob.check_partials(compact_print=True, method='fd', show_only_incorrect=True)

