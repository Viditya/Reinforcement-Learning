# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = list(permutations([i for i in range(m)], 2)) + [(0,0)]
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for x in range (m+t+d)]  ## initialize vector state
        state_encod[state[0]] = 1  ## set the location value into vector
        state_encod[m+state[1]] = 1 ## set the day value into vector
        state_encod[m+t+state[2]] = 1## set time value into vector

        return state_encod

           

    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
            
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])
        
        # Update index for [0, 0]

        possible_actions_index.append(self.action_space.index((0,0)))

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        next_state, wait_time, transit_time, ride_time = self.next_state_func(state, action, Time_matrix)

        revenue_time = ride_time
        idle_time = wait_time + transit_time
        reward = (R * revenue_time) - (C * (revenue_time + idle_time))
        
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        ## Find current state of driver
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]

        ## reward depends on time and lets initialize these variables
        total_time = 0
        wait_time = 0
        ride_time = 0
        transit_time = 0
        ##If Driver refuses to take the ride(0,0)
        if (pickup_loc) == 0 and (drop_loc == 0):   
            wait_time = 1
            next_loc = curr_loc
        ## If Pickup location is same as current location    
        elif pickup_loc == curr_loc:
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            next_loc = drop_loc 
        ## If driver needs to transit to reach pickup location    
        else:
            transit_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.get_day_time_with_travel_duration(curr_time, curr_day, transit_time)
            ride_time =  Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc
            curr_time = new_time
            curr_day = new_day
        
        total_time = ride_time + wait_time
        new_time, new_day = self.get_day_time_with_travel_duration(curr_time, curr_day, total_time)
        next_state = [next_loc, new_time, new_day]
        
        return next_state, wait_time, transit_time, ride_time

    ## Derive day and time to include the transit time and also handle change in time of day or day in week.
    def get_day_time_with_travel_duration(self, time, day, transit):
        if time + int(transit) < 24:     ## New time is in the same day
            new_time = time + int(transit)
            new_day = day
        else:                           ## update day and time to the next day in calendar
            new_time = (time + int(transit)) % 24
            days_passed = (time + int(transit)) // 24
            new_day = (day + days_passed ) % 7
        return new_time, new_day

    def reset(self):
        self.state_init = random.choice(self.state_space)
        return self.action_space, self.state_space, self.state_init
