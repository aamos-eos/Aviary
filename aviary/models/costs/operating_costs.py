"""
This file contains an OpenMDAO ExplicitComponent to calculate aircraft operating costs
based on electrical energy, fuel consumption, and CO2 emissions (carbon tax).
"""

from openmdao.api import ExplicitComponent


class AircraftOperatingCost(ExplicitComponent):
    """
    Calculates the operating cost for an aircraft mission based on:
    - Fuel consumption and fuel price
    - Electrical energy consumption and electricity price
    - Battery degradation cost (amortized over cycle life)
    - Carbon tax on CO2 emissions from fuel burn
    - Flight crew costs (pilots and flight attendants) based on block time
    - Maintenance costs (per hour, landing gear per cycle, airframe checks per month)
    """

    def initialize(self):
        # Electricity cost: 0.12 $/kWh
        self.options.declare(
            "electricity_cost_per_kwh",
            default=0.12,
            desc="Cost of electricity in USD per kWh",
        )
        # Battery cost: 500 $/kWh
        self.options.declare(
            "battery_cost_per_kwh",
            default=500.0,
            desc="Battery replacement cost in USD per kWh of capacity",
        )
        # Battery cycle life: 2400 cycles
        self.options.declare(
            "battery_cycle_life",
            default=2400,
            desc="Number of charge/discharge cycles before battery replacement",
        )
        # Fuel price: 3.2 $/gal
        self.options.declare(
            "fuel_price_per_gal",
            default=3.2,
            desc="Fuel price in USD per gallon",
        )
        # Carbon tax price: 90 $/tonne
        self.options.declare(
            "carbon_tax_per_tonne",
            default=90.0,
            desc="Carbon tax in USD per tonne of CO2 emitted",
        )
        # Fuel density for converting kg to gallons (Jet-A ~3.08 kg/gal)
        self.options.declare(
            "fuel_density_kg_per_gal",
            default=3.03,
            desc="Fuel density in kg per gallon (Jet-A ~3.08)",
        )
        # CO2 emission factor: 3.16 kg CO2 per kg of jet fuel
        self.options.declare(
            "co2_emission_factor",
            default=3.16,
            desc="Rate of CO2 production: kg of CO2 emitted per kg of jet fuel burned (standard value: 3.16)",
        )
        # Flight crew cost parameters
        self.options.declare(
            "pilot_rate",
            default=150.0,
            desc="Pilot hourly rate in USD per hour",
        )
        self.options.declare(
            "flight_attendant_rate",
            default=50.0,
            desc="Flight attendant hourly rate in USD per hour",
        )
        self.options.declare(
            "num_pilots",
            default=2,
            desc="Number of pilots per flight",
        )
        self.options.declare(
            "num_flight_attendants",
            default=2,
            desc="Number of flight attendants per flight",
        )
        # Maintenance cost parameters
        # Airframe costs - HOURLY (1C + 2C + 3C + structural checks)
        self.options.declare(
            "airframe_1c_check_cost_per_hour",
            default=16.0,
            desc="Airframe 1C check cost in USD per flight hour",
        )
        self.options.declare(
            "airframe_2c_check_cost_per_hour",
            default=9.0,
            desc="Airframe 2C check cost in USD per flight hour",
        )
        self.options.declare(
            "airframe_3c_check_cost_per_hour",
            default=35.0,
            desc="Airframe 3C check cost in USD per flight hour",
        )
        self.options.declare(
            "structural_check_cost_per_hour",
            default=14.0,
            desc="Structural check cost in USD per flight hour",
        )
        # Airframe costs - PER CYCLE (4yr + 8yr checks, allocated from monthly)
        self.options.declare(
            "airframe_4yr_check_cost_per_month",
            default=4583.0,
            desc="Airframe 4-year check cost in USD per month",
        )
        self.options.declare(
            "airframe_8yr_check_cost_per_month",
            default=2865.0,
            desc="Airframe 8-year check cost in USD per month",
        )
        # Airframe costs - MONTHLY (landing gear, allocated per flight)
        self.options.declare(
            "landing_gear_cost_per_flight",
            default=32.0,
            desc="Landing gear maintenance cost in USD per flight",
        )
        # Aircraft utilization parameters
        self.options.declare(
            "flights_per_day",
            default=8.0,
            desc="Number of flights per day",
        )
        self.options.declare(
            "flights_per_hour",
            default=0.0,
            desc="Number of flights per hour (if 0, calculated from flights_per_day / 24)",
        )
        self.options.declare(
            "load_factor",
            default=0.9,
            desc="Load factor (utilization factor) as a fraction (0.9 = 90%)",
        )
        self.options.declare(
            "days_per_month",
            default=30.0,
            desc="Average number of days per month for utilization calculations",
        )
        self.options.declare(
            "days_per_year",
            default=365.0,
            desc="Number of days per year for utilization calculations",
        )

        # Engine cost parameters (per hour)
        self.options.declare(
            "props_cost_per_hour",
            default=10.0,
            desc="Props maintenance cost in USD per flight hour",
        )
        self.options.declare(
            "engine_hot_section_cost_per_hour",
            default=50.0,
            desc="Engine hot section maintenance cost in USD per flight hour",
        )
        self.options.declare(
            "engine_overhaul_cost_per_hour",
            default=139.0,
            desc="Engine overhaul cost in USD per flight hour",
        )
        self.options.declare(
            "engine_llps_cost_per_hour",
            default=0.0,
            desc="Engine Life Limited Parts (LLPs) cost in USD per flight hour",
        )
        # Electric motor and busbar cost parameters
        self.options.declare(
            "electric_motor_cost_per_hour",
            default=40.0,
            desc="Electric motor cost in USD per hour",
        )

        self.options.declare(
            "busbar_cost_per_hour",
            default=5.0,
            desc="Busbar cost in USD per hour",
        )

        self.options.declare(
            "flight_hours_per_year",
            default=2000.0,
            desc="Flight hours per year (used to convert component life from years to hours)",
        )
        # Landing fee parameters
        self.options.declare(
            "landing_fee_cost_per_1000lbm",
            default=3.5,
            desc="Landing fee cost in USD per pound of MTOW",
        )
        self.options.declare(
            "nav_cost_per_100nm",
            default=61.75,
            desc="Landing fee cost in USD per 100 nautical miles of flight distance",
        )
        # Landing fee rebate (as a fraction, e.g., 0.5 = 50% rebate)
        self.options.declare(
            "landing_fee_rebate",
            default=0.5,
            desc="Landing fee rebate as a fraction (0.5 = 50% rebate, 0.0 = no rebate)",
        )

    def setup(self):
        # Retrieve options
        electricity_cost_per_kwh = self.options["electricity_cost_per_kwh"]
        battery_cost_per_kwh = self.options["battery_cost_per_kwh"]
        battery_cycle_life = self.options["battery_cycle_life"]
        fuel_price_per_gal = self.options["fuel_price_per_gal"]
        carbon_tax_per_tonne = self.options["carbon_tax_per_tonne"]
        fuel_density_kg_per_gal = self.options["fuel_density_kg_per_gal"]
        co2_emission_factor = self.options["co2_emission_factor"]

        # Derived cost factors
        fuel_cost_per_kg = fuel_price_per_gal / fuel_density_kg_per_gal
        carbon_tax_per_kg_fuel = co2_emission_factor * carbon_tax_per_tonne / 1000.0  # convert tonne to kg
        battery_cost_per_cycle_per_kwh = battery_cost_per_kwh / battery_cycle_life

        # --- Inputs ---
        self.add_input("block_fuel", val=0.0, units="kg", desc="Block fuel burned on the main mission")
        self.add_input("block_electrical_energy", val=0.0, units="kW*h", desc="Block electrical energy consumed during mission")
        self.add_input("block_time", val=0.0, units="h", desc="Block time for the mission")
        self.add_input("combined_block_turbine_time", val=0.0, units="h", desc="Block turbine time for engine maintenance calculations")
        self.add_input("battery_capacity", val=0.0, units="kW*h", desc="Installed battery capacity")
        self.add_input("MTOW", val=0.0, units="lbm", desc="Maximum Take-Off Weight in pounds")
        self.add_input("block_range", val=0.0, units="nmi", desc="Flight distance in nautical miles")
        self.add_input("num_seats", val=76.0, units=None, desc="Number of seats on the aircraft")

        # --- Outputs ---
        self.add_output("block_fuel_cost", val=0.0, units="USD", desc="Cost of block fuel burned")
        self.add_output("block_electricity_cost", val=0.0, units="USD", desc="Cost of block electricity consumed")
        self.add_output("battery_depreciation_cost", val=0.0, units="USD", desc="Battery wear cost per mission cycle")
        self.add_output("block_carbon_tax_cost", val=0.0, units="USD", desc="Carbon tax based on block CO2 emissions")
        self.add_output("block_crew_cost", val=0.0, units="USD", desc="Flight crew cost for block time")
        self.add_output("block_pilot_cost", val=0.0, units="USD", desc="Pilot cost for block time")
        self.add_output("block_flight_attendant_cost", val=0.0, units="USD", desc="Flight attendant cost for block time")
        # Airframe costs broken into 3 categories
        self.add_output("airframe_hourly_check_cost", val=0.0, units="USD", desc="Airframe hourly checks cost (1C + 2C + 3C + structural) per flight")
        self.add_output("airframe_cycle_check_cost", val=0.0, units="USD", desc="Airframe cycle checks cost (4yr + 8yr) per flight")
        self.add_output("airframe_monthly_check_cost", val=0.0, units="USD", desc="Airframe monthly cost (landing gear) per flight")
        self.add_output("total_airframe_maintenance_cost_per_flight", val=0.0, units="USD", desc="Total airframe maintenance cost per flight (hourly + cycle + monthly)")
        self.add_output("engine_cost_per_hour", val=0.0, units="USD/h", desc="Total engine cost per flight hour")
        self.add_output("block_engine_cost", val=0.0, units="USD", desc="Engine (turbine) cost for block time")
        self.add_output("engine_cost_per_flight", val=0.0, units="USD", desc="Engine (turbine) cost per flight")

        self.add_output("block_electric_motor_cost", val=0.0, units="USD", desc="Electric motor cost for block time (per flight)")
        self.add_output("electric_motor_cost_per_hour", val=0.0, units="USD/h", desc="Electric motor cost per flight hour")
        self.add_output("busbar_cost_per_hour", val=0.0, units="USD/h", desc="Busbar cost per flight hour")
        self.add_output("block_busbar_cost", val=0.0, units="USD", desc="Busbar cost for block time (per flight)")
        self.add_output("landing_fee_cost", val=0.0, units="USD", desc="Landing fee cost based on MTOW")
        self.add_output("nav_cost", val=0.0, units="USD", desc="Navigation cost based on flight distance")
        self.add_output("flights_per_month", val=0.0, units=None, desc="Calculated flights per month based on flights_per_day, days_per_month, and load_factor")
        self.add_output("flights_per_year", val=0.0, units=None, desc="Calculated flights per year based on flights_per_day, days_per_year, and load_factor")
        self.add_output("block_total_operating_cost", val=0.0, units="USD", desc="Total block operating cost for the mission")
        self.add_output("cash_operating_cost_per_seat", val=0.0, units="USD", desc="Cash operating cost per seat")
        self.add_output("cash_operating_cost_per_seat_mile", val=0.0, units="USD/nmi", desc="Cash operating cost per seat mile")
        self.add_output("block_co2_emissions", val=0.0, units="kg", desc="Block CO2 emissions from fuel burn")

        # --- Partials ---
        # block_fuel_cost = block_fuel * fuel_cost_per_kg
        self.declare_partials("block_fuel_cost", "block_fuel", val=fuel_cost_per_kg)

        # block_electricity_cost = block_electrical_energy * electricity_cost_per_kwh
        self.declare_partials("block_electricity_cost", "block_electrical_energy", val=electricity_cost_per_kwh)

        # battery_depreciation_cost = battery_capacity * battery_cost_per_cycle_per_kwh
        self.declare_partials("battery_depreciation_cost", "battery_capacity", val=battery_cost_per_cycle_per_kwh)

        # block_carbon_tax_cost = block_fuel * carbon_tax_per_kg_fuel
        self.declare_partials("block_carbon_tax_cost", "block_fuel", val=carbon_tax_per_kg_fuel)

        # block_co2_emissions = block_fuel * co2_emission_factor
        self.declare_partials("block_co2_emissions", "block_fuel", val=co2_emission_factor)

        # Crew costs
        pilot_rate = self.options["pilot_rate"]
        flight_attendant_rate = self.options["flight_attendant_rate"]
        num_pilots = self.options["num_pilots"]
        num_flight_attendants = self.options["num_flight_attendants"]
        pilot_cost_per_hour = num_pilots * pilot_rate
        flight_attendant_cost_per_hour = num_flight_attendants * flight_attendant_rate
        crew_cost_per_hour = pilot_cost_per_hour + flight_attendant_cost_per_hour
        # block_crew_cost = block_time * crew_cost_per_hour
        self.declare_partials("block_crew_cost", "block_time", val=crew_cost_per_hour)
        # block_pilot_cost = block_time * pilot_cost_per_hour
        self.declare_partials("block_pilot_cost", "block_time", val=pilot_cost_per_hour)
        # block_flight_attendant_cost = block_time * flight_attendant_cost_per_hour
        self.declare_partials("block_flight_attendant_cost", "block_time", val=flight_attendant_cost_per_hour)

        # Airframe costs - HOURLY (1C + 2C + 3C + structural checks)
        airframe_1c_check_cost_per_hour = self.options["airframe_1c_check_cost_per_hour"]
        airframe_2c_check_cost_per_hour = self.options["airframe_2c_check_cost_per_hour"]
        airframe_3c_check_cost_per_hour = self.options["airframe_3c_check_cost_per_hour"]
        structural_check_cost_per_hour = self.options["structural_check_cost_per_hour"]
        total_airframe_hourly_maint_cost = airframe_1c_check_cost_per_hour + airframe_2c_check_cost_per_hour + airframe_3c_check_cost_per_hour + structural_check_cost_per_hour
        
        # airframe_hourly_check_cost = block_time * total_airframe_hourly_maint_cost
        self.declare_partials("airframe_hourly_check_cost", "block_time", val=total_airframe_hourly_maint_cost)
        
        # For setup, we need flights_per_month for partials, so calculate it here
        flights_per_day = self.options["flights_per_day"]
        load_factor = self.options["load_factor"]
        days_per_month = self.options["days_per_month"]
        flights_per_month = flights_per_day * days_per_month * load_factor
        
        # Airframe costs - PER CYCLE (4yr + 8yr checks, allocated from monthly)
        # airframe_cycle_check_cost = (4yr + 8yr monthly costs) / flights_per_month
        # This is a constant per flight (no partials w.r.t. inputs)
        
        # Airframe costs - MONTHLY (landing gear, allocated per flight)
        # airframe_monthly_check_cost = landing_gear_cost_per_flight / flights_per_month
        # This is a constant per flight (no partials w.r.t. inputs)
        
        # total_airframe_maintenance_cost_per_flight = hourly + cycle + monthly
        # Partial w.r.t. block_time for the hourly check costs only
        self.declare_partials("total_airframe_maintenance_cost_per_flight", "block_time", val=total_airframe_hourly_maint_cost)

        # engine_cost_per_hour = props_cost_per_hour + engine_hot_section_cost_per_hour + engine_overhaul_cost_per_hour + engine_llps_cost_per_hour
        props_cost_per_hour = self.options["props_cost_per_hour"]
        engine_hot_section_cost_per_hour = self.options["engine_hot_section_cost_per_hour"]
        engine_overhaul_cost_per_hour = self.options["engine_overhaul_cost_per_hour"]
        engine_llps_cost_per_hour = self.options["engine_llps_cost_per_hour"]
        engine_cost_per_hour = props_cost_per_hour + engine_hot_section_cost_per_hour + engine_overhaul_cost_per_hour + engine_llps_cost_per_hour
        # engine_cost_per_hour is a constant (no inputs), so no partials needed

        # block_engine_cost = combined_block_turbine_time * engine_cost_per_hour
        self.declare_partials("block_engine_cost", "combined_block_turbine_time", val=engine_cost_per_hour)
        # engine_cost_per_flight = combined_block_turbine_time * engine_cost_per_hour (same calculation)
        self.declare_partials("engine_cost_per_flight", "combined_block_turbine_time", val=engine_cost_per_hour)

        # electric_motor_cost_per_hour = electric_motor_cost / (electric_motor_life_years * flight_hours_per_year)
        electric_motor_cost_per_hour = self.options["electric_motor_cost_per_hour"]
        busbar_cost_per_hour = self.options["busbar_cost_per_hour"]


        # block_electric_motor_cost = block_time * electric_motor_cost_per_hour
        self.declare_partials("block_electric_motor_cost", "block_time", val=electric_motor_cost_per_hour)

        # block_busbar_cost = block_time * busbar_cost_per_hour
        self.declare_partials("block_busbar_cost", "block_time", val=busbar_cost_per_hour)

        # landing_fee_cost = MTOW * landing_fee_cost_per_1000lbm / 1000 * (1 - landing_fee_rebate)
        landing_fee_cost_per_1000lbm = self.options["landing_fee_cost_per_1000lbm"]
        landing_fee_rebate = self.options["landing_fee_rebate"]
        landing_fee_cost_per_lbm = landing_fee_cost_per_1000lbm / 1000.0 * (1.0 - landing_fee_rebate)
        # Only declare landing fee partials if they are non-zero (avoid OpenMDAO deprecation warning)
        if landing_fee_cost_per_1000lbm != 0.0:
            self.declare_partials("landing_fee_cost", "MTOW", val=landing_fee_cost_per_lbm)
        
        # nav_cost = (block_range / 100) * nav_cost_per_100nm
        nav_cost_per_100nm = self.options["nav_cost_per_100nm"]
        if nav_cost_per_100nm != 0.0:
            self.declare_partials("nav_cost", "block_range", val=nav_cost_per_100nm / 100.0)

        # block_total_operating_cost = block_fuel_cost + block_electricity_cost + battery_depreciation_cost + block_carbon_tax_cost + block_crew_cost + total_airframe_maintenance_cost_per_flight + block_engine_cost + block_electric_motor_cost + block_busbar_cost + landing_fee_cost + nav_cost
        self.declare_partials("block_total_operating_cost", "block_fuel", val=fuel_cost_per_kg + carbon_tax_per_kg_fuel)
        self.declare_partials("block_total_operating_cost", "block_electrical_energy", val=electricity_cost_per_kwh)
        self.declare_partials("block_total_operating_cost", "battery_capacity", val=battery_cost_per_cycle_per_kwh)
        self.declare_partials("block_total_operating_cost", "block_time", val=crew_cost_per_hour + electric_motor_cost_per_hour + busbar_cost_per_hour + total_airframe_hourly_maint_cost)
        self.declare_partials("block_total_operating_cost", "combined_block_turbine_time", val=engine_cost_per_hour)
        # Only declare landing fee and nav cost partials for total cost if they are non-zero
        if landing_fee_cost_per_1000lbm != 0.0:
            self.declare_partials("block_total_operating_cost", "MTOW", val=landing_fee_cost_per_lbm)
        if nav_cost_per_100nm != 0.0:
            self.declare_partials("block_total_operating_cost", "block_range", val=nav_cost_per_100nm / 100.0)
        
        # cash_operating_cost_per_seat = block_total_operating_cost / num_seats
        # Partial w.r.t. num_seats = -block_total_operating_cost / num_seats^2
        # Partial w.r.t. block_total_operating_cost inputs = (1/num_seats) * d(block_total_operating_cost)/d(inputs)
        self.declare_partials("cash_operating_cost_per_seat", "num_seats")
        self.declare_partials("cash_operating_cost_per_seat", "block_fuel")
        self.declare_partials("cash_operating_cost_per_seat", "block_electrical_energy")
        self.declare_partials("cash_operating_cost_per_seat", "battery_capacity")
        self.declare_partials("cash_operating_cost_per_seat", "block_time")
        self.declare_partials("cash_operating_cost_per_seat", "combined_block_turbine_time")
        if landing_fee_cost_per_1000lbm != 0.0:
            self.declare_partials("cash_operating_cost_per_seat", "MTOW")
        if nav_cost_per_100nm != 0.0:
            self.declare_partials("cash_operating_cost_per_seat", "block_range")
        
        # cash_operating_cost_per_seat_mile = block_total_operating_cost / (num_seats * block_range)
        # Partial w.r.t. num_seats = -block_total_operating_cost / (num_seats^2 * block_range)
        # Partial w.r.t. block_range = -block_total_operating_cost / (num_seats * block_range^2) + (1/(num_seats * block_range)) * d(block_total_operating_cost)/d(block_range)
        # Partial w.r.t. block_total_operating_cost inputs = (1/(num_seats * block_range)) * d(block_total_operating_cost)/d(inputs)
        self.declare_partials("cash_operating_cost_per_seat_mile", "num_seats")
        self.declare_partials("cash_operating_cost_per_seat_mile", "block_range")
        self.declare_partials("cash_operating_cost_per_seat_mile", "block_fuel")
        self.declare_partials("cash_operating_cost_per_seat_mile", "block_electrical_energy")
        self.declare_partials("cash_operating_cost_per_seat_mile", "battery_capacity")
        self.declare_partials("cash_operating_cost_per_seat_mile", "block_time")
        self.declare_partials("cash_operating_cost_per_seat_mile", "combined_block_turbine_time")
        if landing_fee_cost_per_1000lbm != 0.0:
            self.declare_partials("cash_operating_cost_per_seat_mile", "MTOW")

    def compute(self, inputs, outputs):

        # Retrieve Inputs
        block_fuel = inputs["block_fuel"]
        block_electrical_energy = inputs["block_electrical_energy"]
        block_time = inputs["block_time"]
        combined_block_turbine_time = inputs["combined_block_turbine_time"]
        battery_capacity = inputs["battery_capacity"]
        MTOW = inputs["MTOW"]
        block_range = inputs["block_range"]
        num_seats = inputs["num_seats"]

        # Retrieve options
        electricity_cost_per_kwh = self.options["electricity_cost_per_kwh"]
        battery_cost_per_kwh = self.options["battery_cost_per_kwh"]
        battery_cycle_life = self.options["battery_cycle_life"]
        fuel_price_per_gal = self.options["fuel_price_per_gal"]
        carbon_tax_per_tonne = self.options["carbon_tax_per_tonne"]
        fuel_density_kg_per_gal = self.options["fuel_density_kg_per_gal"]
        co2_emission_factor = self.options["co2_emission_factor"]
        pilot_rate = self.options["pilot_rate"]
        flight_attendant_rate = self.options["flight_attendant_rate"]
        num_pilots = self.options["num_pilots"]
        num_flight_attendants = self.options["num_flight_attendants"]
        
        # Calculate flights_per_month and flights_per_year from flights_per_day and load_factor
        flights_per_day = self.options["flights_per_day"]
        load_factor = self.options["load_factor"]
        days_per_month = self.options["days_per_month"]
        days_per_year = self.options["days_per_year"]
        flights_per_month = flights_per_day * days_per_month * load_factor
        flights_per_year = flights_per_month  * load_factor * 12
        
        # Airframe costs - HOURLY (1C + 2C + 3C + structural)
        airframe_1c_check_cost_per_hour = self.options["airframe_1c_check_cost_per_hour"]
        airframe_2c_check_cost_per_hour = self.options["airframe_2c_check_cost_per_hour"]
        airframe_3c_check_cost_per_hour = self.options["airframe_3c_check_cost_per_hour"]
        structural_check_cost_per_hour = self.options["structural_check_cost_per_hour"]
        total_airframe_hourly_maint_cost = (airframe_1c_check_cost_per_hour + airframe_2c_check_cost_per_hour 
                                      + airframe_3c_check_cost_per_hour + structural_check_cost_per_hour)
        
        # Airframe costs - PER CYCLE (4yr + 8yr checks, allocated from monthly)
        airframe_4yr_check_cost_per_month = self.options["airframe_4yr_check_cost_per_month"]
        airframe_8yr_check_cost_per_month = self.options["airframe_8yr_check_cost_per_month"]
        airframe_cycle_check_cost_per_month = (airframe_4yr_check_cost_per_month + airframe_8yr_check_cost_per_month)
        airframe_cycle_check_cost_per_flight = airframe_cycle_check_cost_per_month / flights_per_month
        
        # Airframe costs - MONTHLY (landing gear, allocated per flight)
        landing_gear_cost_per_flight = self.options["landing_gear_cost_per_flight"]


        # Derived cost factors
        fuel_cost_per_kg = fuel_price_per_gal / fuel_density_kg_per_gal
        carbon_tax_per_kg_fuel = co2_emission_factor * carbon_tax_per_tonne / 1000.0
        battery_cost_per_cycle_per_kwh = battery_cost_per_kwh / battery_cycle_life
        crew_cost_per_hour = num_pilots * pilot_rate + num_flight_attendants * flight_attendant_rate
        
        # Engine cost per hour
        props_cost_per_hour = self.options["props_cost_per_hour"]
        engine_hot_section_cost_per_hour = self.options["engine_hot_section_cost_per_hour"]
        engine_overhaul_cost_per_hour = self.options["engine_overhaul_cost_per_hour"]
        engine_llps_cost_per_hour = self.options["engine_llps_cost_per_hour"]
        engine_cost_per_hour = props_cost_per_hour + engine_hot_section_cost_per_hour + engine_overhaul_cost_per_hour + engine_llps_cost_per_hour

        # Electric motor and busbar cost per hour
        electric_motor_cost_per_hour = self.options["electric_motor_cost_per_hour"]
        busbar_cost_per_hour = self.options["busbar_cost_per_hour"]

        # Landing fee parameters
        landing_fee_cost_per_1000lbm = self.options["landing_fee_cost_per_1000lbm"]
        nav_cost_per_100nm = self.options["nav_cost_per_100nm"]
        landing_fee_rebate = self.options["landing_fee_rebate"]
        
        # Energy costs
        block_fuel_cost = block_fuel * fuel_cost_per_kg
        block_electricity_cost = block_electrical_energy * electricity_cost_per_kwh
        battery_depreciation_cost = battery_capacity * battery_cost_per_cycle_per_kwh
        block_carbon_tax_cost = block_fuel * carbon_tax_per_kg_fuel
        block_co2_emissions = block_fuel * co2_emission_factor
        
        # Crew costs
        pilot_cost_per_hour = num_pilots * pilot_rate
        flight_attendant_cost_per_hour = num_flight_attendants * flight_attendant_rate
        block_pilot_cost = block_time * pilot_cost_per_hour
        block_flight_attendant_cost = block_time * flight_attendant_cost_per_hour
        block_crew_cost = block_pilot_cost + block_flight_attendant_cost
        
        # Airframe costs - 3 categories
        # 1. Hourly: 1C + 2C + 3C + structural checks
        airframe_check_cost_per_flight = block_time * total_airframe_hourly_maint_cost
        # 2. Per cycle: 4yr + 8yr checks (allocated from monthly)
        # 3. Monthly: Landing gear (allocated per flight)
        # Total airframe maintenance cost = hourly + cycle + monthly
        total_airframe_maintenance_cost_per_flight = (
            airframe_check_cost_per_flight
            + airframe_cycle_check_cost_per_flight
            + landing_gear_cost_per_flight
        )
        
        # Engine costs
        engine_cost_per_flight = combined_block_turbine_time * engine_cost_per_hour

        
        # Electric motor and busbar costs
        block_electric_motor_cost_per_flight = block_time * electric_motor_cost_per_hour
        block_busbar_cost_per_flight = block_time * busbar_cost_per_hour
        
        # Landing fees (with rebate applied)
        nav_cost_per_flight = (block_range / 100.0) * nav_cost_per_100nm
        landing_fee_cost_per_flight = MTOW * landing_fee_cost_per_1000lbm / 1000 * (1.0 - landing_fee_rebate)

        # Total Operating Cost = Crew + Maintenance + Energy + Landing & Navigation Fees
        maintenance_costs = (
            total_airframe_maintenance_cost_per_flight
            + engine_cost_per_flight
            + block_electric_motor_cost_per_flight
            + block_busbar_cost_per_flight
        )
        energy_costs = (
            block_fuel_cost
            + block_electricity_cost
            + battery_depreciation_cost
            + block_carbon_tax_cost
        )
        block_total_operating_cost = (
            block_crew_cost
            + maintenance_costs
            + energy_costs
            + landing_fee_cost_per_flight
            + nav_cost_per_flight
        )

        # Assign all outputs at the end
        outputs["block_fuel_cost"] = block_fuel_cost
        outputs["block_electricity_cost"] = block_electricity_cost
        outputs["battery_depreciation_cost"] = battery_depreciation_cost
        outputs["block_carbon_tax_cost"] = block_carbon_tax_cost
        outputs["block_co2_emissions"] = block_co2_emissions
        outputs["block_crew_cost"] = block_crew_cost
        outputs["block_pilot_cost"] = block_pilot_cost
        outputs["block_flight_attendant_cost"] = block_flight_attendant_cost
        outputs["airframe_hourly_check_cost"] = airframe_check_cost_per_flight
        outputs["airframe_cycle_check_cost"] = airframe_cycle_check_cost_per_flight
        outputs["airframe_monthly_check_cost"] = landing_gear_cost_per_flight
        outputs["total_airframe_maintenance_cost_per_flight"] = total_airframe_maintenance_cost_per_flight
        outputs["engine_cost_per_hour"] = engine_cost_per_hour
        outputs["block_engine_cost"] = engine_cost_per_flight
        outputs["engine_cost_per_flight"] = engine_cost_per_flight
        outputs["electric_motor_cost_per_hour"] = electric_motor_cost_per_hour
        outputs["busbar_cost_per_hour"] = busbar_cost_per_hour
        outputs["block_electric_motor_cost"] = block_electric_motor_cost_per_flight
        outputs["block_busbar_cost"] = block_busbar_cost_per_flight
        outputs["landing_fee_cost"] = landing_fee_cost_per_flight
        outputs["nav_cost"] = nav_cost_per_flight
        outputs["flights_per_month"] = flights_per_month
        outputs["flights_per_year"] = flights_per_year
        outputs["block_total_operating_cost"] = block_total_operating_cost
        
        # Cash operating cost per seat and per seat mile
        cash_operating_cost_per_seat = block_total_operating_cost / num_seats
        cash_operating_cost_per_seat_mile = block_total_operating_cost / (num_seats * block_range)
        outputs["cash_operating_cost_per_seat"] = cash_operating_cost_per_seat
        outputs["cash_operating_cost_per_seat_mile"] = cash_operating_cost_per_seat_mile

    def compute_partials(self, inputs, partials):
        """
        Compute all partial derivatives of outputs with respect to inputs.
        """
        # Retrieve options
        electricity_cost_per_kwh = self.options["electricity_cost_per_kwh"]
        battery_cost_per_kwh = self.options["battery_cost_per_kwh"]
        battery_cycle_life = self.options["battery_cycle_life"]
        fuel_price_per_gal = self.options["fuel_price_per_gal"]
        carbon_tax_per_tonne = self.options["carbon_tax_per_tonne"]
        fuel_density_kg_per_gal = self.options["fuel_density_kg_per_gal"]
        co2_emission_factor = self.options["co2_emission_factor"]
        pilot_rate = self.options["pilot_rate"]
        flight_attendant_rate = self.options["flight_attendant_rate"]
        num_pilots = self.options["num_pilots"]
        num_flight_attendants = self.options["num_flight_attendants"]
        props_cost_per_hour = self.options["props_cost_per_hour"]
        engine_hot_section_cost_per_hour = self.options["engine_hot_section_cost_per_hour"]
        engine_overhaul_cost_per_hour = self.options["engine_overhaul_cost_per_hour"]
        engine_llps_cost_per_hour = self.options["engine_llps_cost_per_hour"]
        electric_motor_cost_per_hour = self.options["electric_motor_cost_per_hour"]
        busbar_cost_per_hour = self.options["busbar_cost_per_hour"]

        # Derived cost factors
        fuel_cost_per_kg = fuel_price_per_gal / fuel_density_kg_per_gal
        carbon_tax_per_kg_fuel = co2_emission_factor * carbon_tax_per_tonne / 1000.0
        battery_cost_per_cycle_per_kwh = battery_cost_per_kwh / battery_cycle_life
        pilot_cost_per_hour = num_pilots * pilot_rate
        flight_attendant_cost_per_hour = num_flight_attendants * flight_attendant_rate
        crew_cost_per_hour = pilot_cost_per_hour + flight_attendant_cost_per_hour
        engine_cost_per_hour = props_cost_per_hour + engine_hot_section_cost_per_hour + engine_overhaul_cost_per_hour + engine_llps_cost_per_hour

        # Partial derivatives (all based on block operations)
        # block_fuel_cost = block_fuel * fuel_cost_per_kg
        partials["block_fuel_cost", "block_fuel"] = fuel_cost_per_kg

        # block_electricity_cost = block_electrical_energy * electricity_cost_per_kwh
        partials["block_electricity_cost", "block_electrical_energy"] = electricity_cost_per_kwh

        # battery_depreciation_cost = battery_capacity * battery_cost_per_cycle_per_kwh
        partials["battery_depreciation_cost", "battery_capacity"] = battery_cost_per_cycle_per_kwh

        # block_carbon_tax_cost = block_fuel * carbon_tax_per_kg_fuel
        partials["block_carbon_tax_cost", "block_fuel"] = carbon_tax_per_kg_fuel

        # block_co2_emissions = block_fuel * co2_emission_factor
        partials["block_co2_emissions", "block_fuel"] = co2_emission_factor

        # block_crew_cost = block_time * crew_cost_per_hour
        partials["block_crew_cost", "block_time"] = crew_cost_per_hour
        # block_pilot_cost = block_time * pilot_cost_per_hour
        partials["block_pilot_cost", "block_time"] = pilot_cost_per_hour
        # block_flight_attendant_cost = block_time * flight_attendant_cost_per_hour
        partials["block_flight_attendant_cost", "block_time"] = flight_attendant_cost_per_hour

        # block_engine_cost = combined_block_turbine_time * engine_cost_per_hour
        partials["block_engine_cost", "combined_block_turbine_time"] = engine_cost_per_hour
        # engine_cost_per_flight = combined_block_turbine_time * engine_cost_per_hour (same calculation)
        partials["engine_cost_per_flight", "combined_block_turbine_time"] = engine_cost_per_hour

        # block_electric_motor_cost = block_time * electric_motor_cost_per_hour
        partials["block_electric_motor_cost", "block_time"] = electric_motor_cost_per_hour

        # block_busbar_cost = block_time * busbar_cost_per_hour
        partials["block_busbar_cost", "block_time"] = busbar_cost_per_hour

        # Airframe costs partials
        airframe_1c_check_cost_per_hour = self.options["airframe_1c_check_cost_per_hour"]
        airframe_2c_check_cost_per_hour = self.options["airframe_2c_check_cost_per_hour"]
        airframe_3c_check_cost_per_hour = self.options["airframe_3c_check_cost_per_hour"]
        structural_check_cost_per_hour = self.options["structural_check_cost_per_hour"]
        total_airframe_hourly_maint_cost = airframe_1c_check_cost_per_hour + airframe_2c_check_cost_per_hour + airframe_3c_check_cost_per_hour + structural_check_cost_per_hour
        
        # airframe_hourly_check_cost = block_time * total_airframe_hourly_maint_cost
        partials["airframe_hourly_check_cost", "block_time"] = total_airframe_hourly_maint_cost
        # airframe_cycle_check_cost and airframe_monthly_check_cost are constants (no partials w.r.t. inputs)
        # total_airframe_maintenance_cost_per_flight = hourly + cycle + monthly (only hourly depends on block_time)
        partials["total_airframe_maintenance_cost_per_flight", "block_time"] = total_airframe_hourly_maint_cost

        # landing_fee_cost = MTOW * landing_fee_cost_per_1000lbm / 1000 * (1 - landing_fee_rebate)
        landing_fee_cost_per_1000lbm = self.options["landing_fee_cost_per_1000lbm"]
        landing_fee_rebate = self.options["landing_fee_rebate"]
        landing_fee_cost_per_lbm = landing_fee_cost_per_1000lbm / 1000.0 * (1.0 - landing_fee_rebate)
        # Only set landing fee partials if they are non-zero (to match declarations in setup)
        if landing_fee_cost_per_1000lbm != 0.0:
            partials["landing_fee_cost", "MTOW"] = landing_fee_cost_per_lbm
        
        # nav_cost = (block_range / 100) * nav_cost_per_100nm
        nav_cost_per_100nm = self.options["nav_cost_per_100nm"]
        if nav_cost_per_100nm != 0.0:
            partials["nav_cost", "block_range"] = nav_cost_per_100nm / 100.0

        # block_total_operating_cost = block_fuel_cost + block_electricity_cost + battery_depreciation_cost + block_carbon_tax_cost + block_crew_cost + total_airframe_maintenance_cost_per_flight + block_engine_cost + block_electric_motor_cost + block_busbar_cost + landing_fee_cost + nav_cost
        # d(block_total_operating_cost)/d(block_fuel) = d(block_fuel_cost)/d(block_fuel) + d(block_carbon_tax_cost)/d(block_fuel)
        partials["block_total_operating_cost", "block_fuel"] = fuel_cost_per_kg + carbon_tax_per_kg_fuel
        # d(block_total_operating_cost)/d(block_electrical_energy) = d(block_electricity_cost)/d(block_electrical_energy)
        partials["block_total_operating_cost", "block_electrical_energy"] = electricity_cost_per_kwh
        # d(block_total_operating_cost)/d(battery_capacity) = d(battery_depreciation_cost)/d(battery_capacity)
        partials["block_total_operating_cost", "battery_capacity"] = battery_cost_per_cycle_per_kwh
        # d(block_total_operating_cost)/d(block_time) = d(block_crew_cost)/d(block_time) + d(block_electric_motor_cost)/d(block_time) + d(block_busbar_cost)/d(block_time) + d(total_airframe_maintenance_cost_per_flight)/d(block_time)
        partials["block_total_operating_cost", "block_time"] = crew_cost_per_hour + electric_motor_cost_per_hour + busbar_cost_per_hour + total_airframe_hourly_maint_cost
        # d(block_total_operating_cost)/d(combined_block_turbine_time) = d(block_engine_cost)/d(combined_block_turbine_time)
        partials["block_total_operating_cost", "combined_block_turbine_time"] = engine_cost_per_hour
        # d(block_total_operating_cost)/d(MTOW) = d(landing_fee_cost)/d(MTOW) (only if non-zero)
        if landing_fee_cost_per_1000lbm != 0.0:
            partials["block_total_operating_cost", "MTOW"] = landing_fee_cost_per_lbm
        # d(block_total_operating_cost)/d(block_range) = d(nav_cost)/d(block_range) (only if non-zero)
        if nav_cost_per_100nm != 0.0:
            partials["block_total_operating_cost", "block_range"] = nav_cost_per_100nm / 100.0
        
        # Get inputs for per-seat calculations
        num_seats = inputs["num_seats"]
        block_range = inputs["block_range"]
        
        # Calculate block_total_operating_cost from inputs (same calculation as in compute)
        block_fuel = inputs["block_fuel"]
        block_electrical_energy = inputs["block_electrical_energy"]
        battery_capacity = inputs["battery_capacity"]
        MTOW = inputs["MTOW"]
        block_time = inputs["block_time"]
        combined_block_turbine_time = inputs["combined_block_turbine_time"]
        
        # Calculate intermediate values needed for block_total_operating_cost
        fuel_cost_per_kg = fuel_price_per_gal / fuel_density_kg_per_gal
        carbon_tax_per_kg_fuel = co2_emission_factor * carbon_tax_per_tonne / 1000.0
        battery_cost_per_cycle_per_kwh = battery_cost_per_kwh / battery_cycle_life
        block_fuel_cost = block_fuel * fuel_cost_per_kg
        block_electricity_cost = block_electrical_energy * electricity_cost_per_kwh
        battery_depreciation_cost = battery_capacity * battery_cost_per_cycle_per_kwh
        block_carbon_tax_cost = block_fuel * carbon_tax_per_kg_fuel
        block_crew_cost = block_time * crew_cost_per_hour
        block_engine_cost = combined_block_turbine_time * engine_cost_per_hour
        block_electric_motor_cost = block_time * electric_motor_cost_per_hour
        block_busbar_cost = block_time * busbar_cost_per_hour
        airframe_check_cost_per_flight = block_time * total_airframe_hourly_maint_cost
        
        # Get flights_per_month for cycle costs
        flights_per_day = self.options["flights_per_day"]
        load_factor = self.options["load_factor"]
        days_per_month = self.options["days_per_month"]
        flights_per_month = flights_per_day * days_per_month * load_factor
        airframe_4yr_check_cost_per_month = self.options["airframe_4yr_check_cost_per_month"]
        airframe_8yr_check_cost_per_month = self.options["airframe_8yr_check_cost_per_month"]
        airframe_cycle_check_cost_per_flight = (airframe_4yr_check_cost_per_month + airframe_8yr_check_cost_per_month) / flights_per_month
        landing_gear_cost_per_flight = self.options["landing_gear_cost_per_flight"]
        total_airframe_maintenance_cost_per_flight = (
            airframe_check_cost_per_flight
            + airframe_cycle_check_cost_per_flight
            + landing_gear_cost_per_flight
        )
        landing_fee_rebate = self.options["landing_fee_rebate"]
        landing_fee_cost_per_flight = MTOW * landing_fee_cost_per_1000lbm / 1000 * (1.0 - landing_fee_rebate)
        nav_cost_per_flight = (block_range / 100.0) * nav_cost_per_100nm
        
        block_total_operating_cost = (
            block_crew_cost
            + total_airframe_maintenance_cost_per_flight
            + block_engine_cost
            + block_electric_motor_cost
            + block_busbar_cost
            + block_fuel_cost
            + block_electricity_cost
            + battery_depreciation_cost
            + block_carbon_tax_cost
            + landing_fee_cost_per_flight
            + nav_cost_per_flight
        )
        
        # cash_operating_cost_per_seat = block_total_operating_cost / num_seats
        # d(cash_operating_cost_per_seat)/d(num_seats) = -block_total_operating_cost / num_seats^2
        # d(cash_operating_cost_per_seat)/d(block_total_operating_cost inputs) = (1/num_seats) * d(block_total_operating_cost)/d(inputs)
        if num_seats > 0:
            partials["cash_operating_cost_per_seat", "num_seats"] = -block_total_operating_cost / (num_seats ** 2)
            # Chain rule: d(cash_operating_cost_per_seat)/d(input) = (1/num_seats) * d(block_total_operating_cost)/d(input)
            for input_name in ["block_fuel", "block_electrical_energy", "battery_capacity", "block_time", "combined_block_turbine_time", "MTOW", "block_range"]:
                key = ("block_total_operating_cost", input_name)
                if key in partials:
                    partials["cash_operating_cost_per_seat", input_name] = partials[key] / num_seats
        
        # cash_operating_cost_per_seat_mile = block_total_operating_cost / (num_seats * block_range)
        # d(cash_operating_cost_per_seat_mile)/d(num_seats) = -block_total_operating_cost / (num_seats^2 * block_range)
        # d(cash_operating_cost_per_seat_mile)/d(block_range) = -block_total_operating_cost / (num_seats * block_range^2) + (1/(num_seats * block_range)) * d(block_total_operating_cost)/d(block_range)
        if num_seats > 0 and block_range > 0:
            partials["cash_operating_cost_per_seat_mile", "num_seats"] = -block_total_operating_cost / (num_seats ** 2 * block_range)
            # For block_range, need to account for both the denominator and numerator dependencies
            key_range = ("block_total_operating_cost", "block_range")
            if key_range in partials:
                partials["cash_operating_cost_per_seat_mile", "block_range"] = (
                    partials[key_range] / (num_seats * block_range)
                    - block_total_operating_cost / (num_seats * block_range ** 2)
                )
            else:
                partials["cash_operating_cost_per_seat_mile", "block_range"] = -block_total_operating_cost / (num_seats * block_range ** 2)
            # Chain rule for other inputs
            for input_name in ["block_fuel", "block_electrical_energy", "battery_capacity", "block_time", "combined_block_turbine_time", "MTOW"]:
                key = ("block_total_operating_cost", input_name)
                if key in partials:
                    partials["cash_operating_cost_per_seat_mile", input_name] = partials[key] / (num_seats * block_range)


if __name__ == "__main__":
    """
    Test and evaluate the AircraftOperatingCost component.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from openmdao.api import Problem, IndepVarComp, Group

    # Create a problem
    prob = Problem()

    # Create an independent variable component
    indeps = prob.model.add_subsystem("indeps", IndepVarComp(), promotes=["*"])

    # Add test inputs
    indeps.add_output("block_fuel", val=887.0*1.15, units="lbm")  # 850 kg block fuel burned
    indeps.add_output("block_electrical_energy", val=1944.0, units="kW*h")  # 500 kWh block electrical energy
    indeps.add_output("block_time", val=71/60, units="h")  # 2.5 hours block time
    indeps.add_output("combined_block_turbine_time", val=80/60, units="h")  # Combined operating time across all turbines for block mission
    indeps.add_output("battery_capacity", val=2500.0, units="kW*h")  # 1000 kWh battery capacity
    indeps.add_output("MTOW", val=86000.0, units="lbm")  # 86,000 lb MTOW
    indeps.add_output("block_range", val=210.0, units="nmi")  # 500 nautical miles
    indeps.add_output("num_seats", val=76.0, units=None)  # 76 seats

    # Add the operating cost component
    prob.model.add_subsystem(
        "operating_cost",
        AircraftOperatingCost(),
        promotes=["*"],
    )

    # Setup the problem
    prob.setup()

    # Run the problem
    prob.run_model()

    # Print results
    print("=" * 60)
    print("Aircraft Operating Cost Evaluation")
    print("=" * 60)
    # Get utilization parameters
    flights_per_day = prob.model.operating_cost.options["flights_per_day"]
    load_factor = prob.model.operating_cost.options["load_factor"]
    days_per_month = prob.model.operating_cost.options["days_per_month"]
    days_per_year = prob.model.operating_cost.options["days_per_year"]
    
    print(f"\nInputs:")
    print(f"  Block fuel:              {prob['block_fuel'][0]:.2f} kg")
    print(f"  Block electrical energy: {prob['block_electrical_energy'][0]:.2f} kWh")
    print(f"  Block time:              {prob['block_time'][0]:.2f} hours")
    print(f"  Block turbine time:      {prob['combined_block_turbine_time'][0]:.2f} hours")
    print(f"  Battery capacity:        {prob['battery_capacity'][0]:.2f} kWh")
    print(f"  MTOW:                    {prob['MTOW'][0]:.2f} lb")
    print(f"  Flight distance:         {prob['block_range'][0]:.2f} NM")
    
    print(f"\nUtilization Parameters:")
    print(f"  Flights per day:         {flights_per_day:.1f}")
    print(f"  Load factor:             {load_factor*100:.1f}%")
    print(f"  Days per month:          {days_per_month:.1f}")
    print(f"  Days per year:           {days_per_year:.1f}")
    print(f"  Flights per month:       {prob['flights_per_month'][0]:.1f} (calculated)")
    print(f"  Flights per year:        {prob['flights_per_year'][0]:.1f} (calculated)")

    print(f"\nOutputs:")
    print(f"  Block fuel cost:         ${prob['block_fuel_cost'][0]:.2f}")
    print(f"  Block electricity cost:  ${prob['block_electricity_cost'][0]:.2f}")
    print(f"  Battery depreciation:    ${prob['battery_depreciation_cost'][0]:.2f}")
    print(f"  Block carbon tax cost:   ${prob['block_carbon_tax_cost'][0]:.2f}")
    print(f"  Crew Costs:")
    print(f"    - Pilots:                       ${prob['block_pilot_cost'][0]:.2f}")
    print(f"    - Flight attendants:            ${prob['block_flight_attendant_cost'][0]:.2f}")
    print(f"    - Total crew cost:              ${prob['block_crew_cost'][0]:.2f}")
    print(f"  Airframe Costs:")
    print(f"    - Hourly (1C+2C+3C+structural): ${prob['airframe_hourly_check_cost'][0]:.2f}")
    print(f"    - Cycle (4yr+8yr):              ${prob['airframe_cycle_check_cost'][0]:.2f}")
    print(f"    - Monthly (landing gear):       ${prob['airframe_monthly_check_cost'][0]:.2f}")
    print(f"    - Total airframe per flight:    ${prob['total_airframe_maintenance_cost_per_flight'][0]:.2f}")
    print(f"  Engine (turbine) cost per hour: ${prob['engine_cost_per_hour'][0]:.2f}/h")
    print(f"  Engine (turbine) cost per flight: ${prob['engine_cost_per_flight'][0]:.2f}")
    print(f"  Electric motor cost per hour:     ${prob['electric_motor_cost_per_hour'][0]:.2f}/h")
    print(f"  Busbar cost per hour:             ${prob['busbar_cost_per_hour'][0]:.2f}/h")
    print(f"  Electric motor cost per flight:   ${prob['block_electric_motor_cost'][0]:.2f}")
    print(f"  Busbar cost per flight:           ${prob['block_busbar_cost'][0]:.2f}")
    print(f"  Landing fee cost:                 ${prob['landing_fee_cost'][0]:.2f}")
    print(f"  Navigation cost:                  ${prob['nav_cost'][0]:.2f}")
    print(f"  Block CO2 emissions:      {prob['block_co2_emissions'][0]:.2f} kg")
    print(f"  Block total operating cost: ${prob['block_total_operating_cost'][0]:.2f}")
    print(f"  Cash operating cost per seat:     ${prob['cash_operating_cost_per_seat'][0]:.2f}")
    print(f"  Cash operating cost per seat mile: ${prob['cash_operating_cost_per_seat_mile'][0]:.4f} USD/(seat*nmi)")

    print(f"\nCost Breakdown:")
    total = prob["block_total_operating_cost"][0]
    print(f"  Block fuel:              {prob['block_fuel_cost'][0]/total*100:.1f}%")
    print(f"  Block electricity:       {prob['block_electricity_cost'][0]/total*100:.1f}%")
    print(f"  Battery depreciation:    {prob['battery_depreciation_cost'][0]/total*100:.1f}%")
    print(f"  Block carbon tax:        {prob['block_carbon_tax_cost'][0]/total*100:.1f}%")
    print(f"  Pilots:                  {prob['block_pilot_cost'][0]/total*100:.1f}%")
    print(f"  Flight attendants:       {prob['block_flight_attendant_cost'][0]/total*100:.1f}%")
    print(f"  Airframe hourly:         {prob['airframe_hourly_check_cost'][0]/total*100:.1f}%")
    print(f"  Airframe cycle:          {prob['airframe_cycle_check_cost'][0]/total*100:.1f}%")
    print(f"  Airframe monthly:        {prob['airframe_monthly_check_cost'][0]/total*100:.1f}%")
    print(f"  Engine (turbine) cost:   {prob['block_engine_cost'][0]/total*100:.1f}%")
    print(f"  Electric motor cost:     {prob['block_electric_motor_cost'][0]/total*100:.1f}%")
    print(f"  Busbar cost:             {prob['block_busbar_cost'][0]/total*100:.1f}%")
    print(f"  Landing fees:            {prob['landing_fee_cost'][0]/total*100:.1f}%")
    print(f"  Navigation fees:        {prob['nav_cost'][0]/total*100:.1f}%")

    # Check partial derivatives
    print(f"\n" + "=" * 60)
    print("Partial Derivatives Check")
    print("=" * 60)
    prob.check_partials(compact_print=True)

    # Prepare data for plotting
    total = prob["block_total_operating_cost"][0]
    
    # Maintenance costs breakdown
    maintenance_costs = {
        "Airframe Hourly \n(1C+2C+3C+Structural)": prob['airframe_hourly_check_cost'][0],
        "Airframe Cycle \n(4yr+8yr)": prob['airframe_cycle_check_cost'][0],
        "Airframe Monthly \n(Landing Gear)": prob['airframe_monthly_check_cost'][0],
        "Engine (Turbine)": prob['block_engine_cost'][0],
        "Electric Motor": prob['block_electric_motor_cost'][0],
        "Busbar": prob['block_busbar_cost'][0],
    }
    maintenance_total = sum(maintenance_costs.values())
    
    # Crew costs breakdown
    crew_costs = {
        "Pilots": prob['block_pilot_cost'][0],
        "Flight Attendants": prob['block_flight_attendant_cost'][0],
    }
    crew_total = sum(crew_costs.values())
    
    # Energy costs breakdown
    energy_costs = {
        "Fuel": prob['block_fuel_cost'][0],
        "Electricity": prob['block_electricity_cost'][0],
        "Battery Depreciation": prob['battery_depreciation_cost'][0],
        "Carbon Tax": prob['block_carbon_tax_cost'][0],
    }
    energy_total = sum(energy_costs.values())
    
    # Landing/Navigation costs breakdown
    landing_fee_cost_per_flight = prob['landing_fee_cost'][0]
    nav_cost_per_flight = prob['nav_cost'][0]
    landing_nav_costs = {
        "Landing Fees": landing_fee_cost_per_flight,
        "Navigation Fees": nav_cost_per_flight,
    }
    landing_nav_total = sum(landing_nav_costs.values())
    
    # Create separate figures for each cost category
    
    # Figure 1: Maintenance Costs
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle('Maintenance Costs Breakdown', fontsize=14, fontweight='bold')
    
    if maintenance_total > 0:
        maintenance_labels = [k for k, v in maintenance_costs.items() if v > 0]
        maintenance_values = [v for v in maintenance_costs.values() if v > 0]
        # Pie Chart
        ax1.pie(maintenance_values, labels=maintenance_labels, autopct='%1.1f%%', startangle=90, radius=1.3)
        ax1.set_title('Pie Chart', fontsize=12, fontweight='bold')
        # Bar Chart
        bars = ax2.bar(range(len(maintenance_labels)), maintenance_values, color='steelblue')
        ax2.set_xticks(range(len(maintenance_labels)))
        ax2.set_xticklabels(maintenance_labels, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Cost ($)', fontsize=11)
        ax2.set_title('Bar Chart', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, maintenance_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${val:.0f}', ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No Maintenance Costs', ha='center', va='center', fontsize=12)
        ax1.set_title('Pie Chart', fontsize=12, fontweight='bold')
        ax2.text(0.5, 0.5, 'No Maintenance Costs', ha='center', va='center', fontsize=12)
        ax2.set_title('Bar Chart', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('operating_cost_maintenance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Crew Costs
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Crew Costs Breakdown', fontsize=14, fontweight='bold')
    
    if crew_total > 0:
        crew_labels = list(crew_costs.keys())
        crew_values = list(crew_costs.values())
        # Pie Chart
        ax3.pie(crew_values, labels=crew_labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
        ax3.set_title('Pie Chart', fontsize=12, fontweight='bold')
        # Bar Chart
        bars = ax4.bar(range(len(crew_labels)), crew_values, color=['lightcoral', 'lightblue'])
        ax4.set_xticks(range(len(crew_labels)))
        ax4.set_xticklabels(crew_labels, rotation=45, ha='right', fontsize=10)
        ax4.set_ylabel('Cost ($)', fontsize=11)
        ax4.set_title('Bar Chart', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, crew_values)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${val:.0f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No Crew Costs', ha='center', va='center', fontsize=12)
        ax3.set_title('Pie Chart', fontsize=12, fontweight='bold')
        ax4.text(0.5, 0.5, 'No Crew Costs', ha='center', va='center', fontsize=12)
        ax4.set_title('Bar Chart', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('operating_cost_crew.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Energy Costs
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('Energy Costs Breakdown', fontsize=14, fontweight='bold')
    
    if energy_total > 0:
        energy_labels = [k for k, v in energy_costs.items() if v > 0]
        energy_values = [v for v in energy_costs.values() if v > 0]
        # Pie Chart
        ax5.pie(energy_values, labels=energy_labels, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Pie Chart', fontsize=12, fontweight='bold')
        # Bar Chart
        bars = ax6.bar(range(len(energy_labels)), energy_values, color='forestgreen')
        ax6.set_xticks(range(len(energy_labels)))
        ax6.set_xticklabels(energy_labels, rotation=45, ha='right', fontsize=10)
        ax6.set_ylabel('Cost ($)', fontsize=11)
        ax6.set_title('Bar Chart', fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, energy_values)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${val:.0f}', ha='center', va='bottom', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No Energy Costs', ha='center', va='center', fontsize=12)
        ax5.set_title('Pie Chart', fontsize=12, fontweight='bold')
        ax6.text(0.5, 0.5, 'No Energy Costs', ha='center', va='center', fontsize=12)
        ax6.set_title('Bar Chart', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('operating_cost_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Landing/Navigation Costs
    fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle('Landing/Navigation Costs Breakdown', fontsize=14, fontweight='bold')
    
    if landing_nav_total > 0:
        landing_nav_labels = list(landing_nav_costs.keys())
        landing_nav_values = list(landing_nav_costs.values())
        # Pie Chart
        ax7.pie(landing_nav_values, labels=landing_nav_labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'orange'])
        ax7.set_title('Pie Chart', fontsize=12, fontweight='bold')
        # Bar Chart
        bars = ax8.bar(range(len(landing_nav_labels)), landing_nav_values, color=['gold', 'orange'])
        ax8.set_xticks(range(len(landing_nav_labels)))
        ax8.set_xticklabels(landing_nav_labels, rotation=45, ha='right', fontsize=10)
        ax8.set_ylabel('Cost ($)', fontsize=11)
        ax8.set_title('Bar Chart', fontsize=12, fontweight='bold')
        ax8.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, landing_nav_values)):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${val:.0f}', ha='center', va='bottom', fontsize=9)
    else:
        ax7.text(0.5, 0.5, 'No Landing/Navigation Costs', ha='center', va='center', fontsize=12)
        ax7.set_title('Pie Chart', fontsize=12, fontweight='bold')
        ax8.text(0.5, 0.5, 'No Landing/Navigation Costs', ha='center', va='center', fontsize=12)
        ax8.set_title('Bar Chart', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('operating_cost_landing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("Cost breakdown plots saved to separate files:")
    print("  - operating_cost_maintenance.png")
    print("  - operating_cost_crew.png")
    print("  - operating_cost_energy.png")
    print("  - operating_cost_landing.png")
    print("=" * 60)

    print(f"\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
