import numpy as np
import matplotlib.pyplot as plt
import tabulate as tab
import cmath as math
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga

# Extract data from input file (data parsing)

    ## read the input data

vehicle_data_df=pd.read_excel('Case2_vehicle.xlsx',usecols=[1])
track_data_df=pd.read_excel('Case2_track.xlsx')
fuel_data_df=pd.read_excel('Case2_fuel.xlsx',usecols=[1])
 
    ## dataframe to array

vehicle_data=vehicle_data_df.to_numpy()
fuel_data=fuel_data_df.to_numpy()

    ## create parameters
        ### Vehicle

vehicle_weight=vehicle_data[0] #kg
drag_coefficient=vehicle_data[1]
rolling_friction=vehicle_data[2]
frontal_area=vehicle_data[3] #m^2

powertrain_power=vehicle_data[4] #kW
powertrain_efficiency=0.95

battery_capacity=vehicle_data[5] #kWh
battery_power=vehicle_data[6] #kW
battery_weight=vehicle_data[7] #kg
battery_soc=1

regenerative_breaking_power=vehicle_data[8] #kW
regenerative_breaking_efficiency=vehicle_data[9]

cooling_system_power=vehicle_data[10]/1000 #kW
lights_power=vehicle_data[11]/1000 #kW
control_system_power=vehicle_data[12]/1000 #kW

available_area_pv=vehicle_data[13] #m2
pv_efficiency=vehicle_data[14] #kw/m2
pv_power=available_area_pv*pv_efficiency #kW

        ### environment

air_density=1.225 #kg/m^3
gravity=9.81 #m/s^2

        ### track
            #### note: check the titles of the excel file columns to make sure the excel file will work
number_of_laps=20 ## can be optimized for any number of laps!!!
track_max_speed_kmph=track_data_df['Max Speed (km/h)'].values #km/h
track_max_speed=track_max_speed_kmph*(1000/3600) #m/s
track_length_km=track_data_df['Length (km)'].values #km
track_length=track_length_km*1000 #m
track_slope_deg=track_data_df['Slope (º)'].values #º
track_slope=track_slope_deg*(math.pi/180) #radians

        ### fuel

Fuel_Efficiency=fuel_data[0]
co2_emissions=fuel_data[1] #kgCO2/kWh
average_irradiance=fuel_data[2] #w/m2

        ### check inputs


# Objective function and constraints

    ##initialize functions

time_minimum=(track_length/(track_max_speed)) #seconds for 20 laps



track_acceleration=np.array(np.zeros(42))

time_lap_minimum=0

i=0

for i in range(0, len(time_minimum)):    
   time_lap_minimum = time_lap_minimum + time_minimum[i]; #minimum time for one lap
 
time_total_minimum=time_lap_minimum*number_of_laps #minimum time for 20 laps

    ## Define objective function

def totaltime(track_speed):

        ### Globalize relevant variables

    global track_time, total_force, track_acceleration, power, energy, battery_final_capacity, total_energy

    track_time=(track_length/track_speed)

        ### Define acceleration
    i=0
    for i in range(0,len(track_time)):
       
        if i == 0:
            track_acceleration[i]=(track_speed[i+1]**2+0)/(2*track_length[i]) #m/s^2 ### note: the v at the begining might be non zero, but I use the acceleration for calc and considered 0 at the start as you can see
        elif (i % 2) == 0:
            track_acceleration[i]=(track_speed[i+1]-track_speed[i-1]**2)/(2*track_length[i])
        else:
            track_acceleration[i]=0

        ### create acceleration sign array
    
    i=0
    acceleration_sign = np.zeros(42)
    for i in range(0,len(track_time)):
        if track_acceleration[i] != 0:
            acceleration_sign[i] = np.sign(track_acceleration[i])
        if track_acceleration[i] == 0:
            acceleration_sign[i] = 1    
        
        ### Calculate energy
          
            #### forces

    aerodynamic_force=0.5*air_density*frontal_area*drag_coefficient*track_speed*track_speed*acceleration_sign #newtons
    slope_losses=(vehicle_weight+battery_weight)*gravity*np.sin(track_slope) #newtons
    acceleration_force=(battery_weight+vehicle_weight)*(track_acceleration) #newtons
    friction_losses=(vehicle_weight+battery_weight)*gravity*rolling_friction*np.cos(track_slope)*acceleration_sign #newtons
    total_force=acceleration_force+aerodynamic_force+friction_losses+slope_losses #newtons

            #### power and energy
    

    power=-(total_force*track_speed)/1000 - control_system_power - lights_power - cooling_system_power + pv_power #kw (watt to kw /1000)
    energy=((power*(track_time/3600))/powertrain_efficiency) #kwh

        ### calculate total energy expended per lap

    total_energy=sum(energy)

        ### calculate final capacity after 20 laps
    
    battery_final_capacity = battery_capacity - total_energy*number_of_laps # battery final capacity after 20 laps

        #Calculate total time per lap

    total_time=sum(track_time)
    
         ## fitness algorithim (penalties)
            ### max power penalty
    if (power[i]) > min(powertrain_power,battery_power):
            total_time = total_time + 9999
    #         #### run out of battery capacity penalty (for 20 laps)
    if battery_final_capacity < 0:
            total_time = total_time + 9999

    total_time=total_time*number_of_laps ##time for 20 laps!!
    
    return total_time

        ### Calculate speed 


#Adjust GA parameters

algorithm_param = {'max_num_iteration': 5000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.01,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv': None}

# set max speed boundary

varbound=np.stack((np.zeros(len(track_max_speed)),track_max_speed),axis=1)

# run genetic algorithm

model=ga(function=totaltime,dimension=len(track_length),variable_type='real',variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()
print(model.output_dict)