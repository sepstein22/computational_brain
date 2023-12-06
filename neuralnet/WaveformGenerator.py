#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Noah's code for waveform generation in comp-phys project

#Import Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian

#The class waveform is used to return voltage and time data of a waveform we 
#wish to use in an experiment. The initialization of the waveform object
#Sets a duration and the number of data points. The methods within the
#Waveform class allow you to call specific types of waves with different
#characteristics. 

class Waveform:
    def __init__(self, duration=1, sampling_rate=1000):
        self.set_sampling_parameters(duration, sampling_rate)
        self.voltage_data = None

    def set_sampling_parameters(self, duration, sampling_rate):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    #This next part of the waveform object is all the methods
    #which produce different waveforms
        
    #Returns a sine wave of set frequency and amplitude
    def sine_wave(self, frequency, amplitude):
        self.voltage_data = amplitude * np.sin(2 * np.pi * frequency * self.time)

    #The square wave method also allows you to define a duty cycle
    def square_wave(self, frequency, amplitude, duty_cycle=0.5):
        periods = int(self.duration * frequency)
        samples_per_period = int(self.sampling_rate / frequency)
        on_samples = int(samples_per_period * duty_cycle)

        square_wave = np.zeros(len(self.time))
        for i in range(periods):
            start = i * samples_per_period
            square_wave[start:start + on_samples] = amplitude

        self.voltage_data = square_wave

    #Returns a basic triangle wave of set frequency and amplitude
    def triangular_wave(self, frequency, amplitude):
        self.voltage_data = amplitude * np.abs(2 * (self.time * frequency - np.floor(self.time * frequency + 0.5)))
    
    #This method returns a gaussian pulse of a set FWHM (in seconds) and amplitude
    def gaussian_pulse(self, amplitude, fwhm_seconds):
        center = self.duration / 2
        std = fwhm_seconds*1000 / (2 * np.sqrt(2 * np.log(2)))
        pulse = amplitude * gaussian(len(self.time), std=std)
        shift = int(center * self.sampling_rate) - len(self.time) // 2
        self.voltage_data = np.roll(pulse, shift)

    #This method returns a 2d array of time and voltage data respectfully
    def get_waveform_data(self):
        if self.voltage_data is None:
            raise ValueError("Waveform data has not been generated yet. Call a waveform generation method first.")
        return np.vstack((self.time, self.voltage_data)).T

    #This method plots your waveform
    def plot_waveform(self, title="Waveform"):
        if self.voltage_data is None:
            raise ValueError("Waveform data has not been generated yet. Call a waveform generation method first.")
        plt.plot(self.time, self.voltage_data)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage")
        plt.show()

# Example usage:
waveform_generator = Waveform(duration=2, sampling_rate=1000)

# Generate a Gaussian pulse with FWHM in seconds
waveform_generator.gaussian_pulse(amplitude=1, fwhm_seconds=1)

# Get the waveform data
waveform_data = waveform_generator.get_waveform_data()
print("Gaussian Pulse Data:")
print(waveform_data)

# Plot the Gaussian pulse
waveform_generator.plot_waveform(title="Gaussian Pulse")


# In[ ]:




