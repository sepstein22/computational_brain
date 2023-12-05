import os
import sys
import zipfile 
import pandas as pd
import warnings 
import numpy as np

class retrieve_file:
    def __init__(self, neuron_type= 'L5PC', num_ap = 1, V_data = None, I_data = None, t_data = None ):
        '''
        defines the ground truth for the optimization problem
        
        Args: 
            neuron_type (str): assume to be L5PC, HH, or manual otherwise raises an error
            num_ap (int): desired number of action potentials
            V_data (array): manually entered voltage time series
            I_data (array): manually entered stimulaiton time series
            t_data (array): manually entered time array
        '''
        self.neuron = neuron_type
        self.num_ap = num_ap
        self.V_data  = V_data
        self.I_data = I_data
        self.t_data = t_data
       
    def load(self):
        '''loads in file based on specified conditions due to naming framework and stores key global variables
        
        Returns: self.V_data, self.I_data, self.t_data, self.V0, self.dt, b (center) '''
        if self.neuron == 'L5PC':
            if self.num_ap >= 2:
                warnings.warn("For multiple action potentials, defaulted to general repetitive firing")
                fname = 'gt_multa'
            elif self.num_ap == 0:
                fname = 'gt_noap'
            else:
                print('Default chosen of 1 Action Potential')
                fname = 'gt_1a'
            with zipfile.ZipFile('./sim_data/' + fname + '.zip', 'r') as zip_ref:
                zip_ref.extractall()
            with open(fname + '.csv') as csvfile:
                df = pd.read_csv(csvfile)
            self.I_data = df['stim'].to_numpy()
            self.V_data = df['voltage'].to_numpy()
            self.t_data = df['time'].to_numpy() 
            self.dt = df['time'][1]-df['time'][0]
            self.V0 = df['voltage'][0]
            impulse = np.where(self.I_data != 0.0)
            if self.num_ap == 0:
                b = 150.0
            else:
                center = round((impulse[0][-1] - impulse[0][0])/2) + impulse[0][0]
                b = self.t_data[center]
        elif self.neuron == 'HH':
            if self.num_ap >= 2: 
                warnings.warn("For multiple action potentials, defaulted to general repetitive firing")
                fname = 'hh_multap'
            elif self.num_ap == 0:
                fname = 'hh_noap'
            else: 
                fname = 'hh_1ap'
            with zipfile.ZipFile('./sim_data/' + fname + '.zip', 'r') as zip_ref:
                csv_file_name = zip_ref.namelist()[0]
                with zip_ref.open(csv_file_name) as csv_file:
                    df = pd.read_csv(csv_file)
            self.V_data = df['voltage'].to_numpy()
            self.I_data = df['stim'].to_numpy()
            self.t_data = df['time'].to_numpy() 
            self.dt = df['time'][1]-df['time'][0]
            self.V0 = df['voltage'][0]
            impulse = np.where(self.I_data != 0.0)
            if self.num_ap == 0:
                b = 150.0
            else:
                center = round((impulse[0][-1] - impulse[0][0])/2) + impulse[0][0]
                b = self.t_data[center]
        elif self.neuron == 'manual':
            if self.V_data is None or self.I_data is None or self.t_data is None:
                raise Exception("Missing parameter, argument requires voltage, time, and input stim array ")
                pass
            else:
                self.V_data  = self.V_data.to_numpy()
                self.I_data = self.I_data.to_numpy()
                self.t_data = self.t_data.to_numpy()
                self.V0 = self.V_data[0]
                self.dt = self.t_data[1]-self.t_data[0]
                b = 150.0
        else:
            raise Exception("Sorry this is not a valid choice of neuron model for this problem, please choose HH, L5PC, or manually insert your data ")
            pass #raise warning here to input valid input
        return self.V_data, self.I_data, self.t_data, self.V0, self.dt, b
    
