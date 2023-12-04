import autograd.numpy as np 
from autograd import grad
from scipy import optimize

class stim_adj: 
    def __init__(self, V_data, t_data, dt, HH_params, guess_a, guess_c, guess_b, bounds = [], method =  'CG'):
        '''
        args:
            V0 (float): defined in upload.py to be initial voltage
            V_data (array): defined in upload.py to be voltage time series
            t (array): defined in upload.py as all time steps
            data_steps (float): defined in upload.py to be empiracle time step size
            dt (float): retrieved from parent class specifying desired simulation time stepping
             
            HH_parms: 
                m, n, h (floats): retrieved from parent class as HH param
                g_Na, g_K, g_L, E_Na, E_K, E_K (floats): retrieved from parent class as HH param
            
            guess_a, guess_c, guess_b (floats): retrieved from parent class as optimization initializers 
                a: amplitude 
                c: frequency
                b: center location
            bounds (list of tuples): specifies bounds for each variable
            method (str): method for optimization
        '''
        
        #variables from empiracle data
        self.V0 = V_data[0]
        self.V_data = V_data
        self.t_data = t_data
        self.t_final = t_data[-1]
        #self.data_steps = data_steps
        
        
        #user specified or default
        self.dt = dt 
        self.g_Na, self.g_K, self.g_L, self.E_Na, self.E_K, self.E_L, self.C_m, self.m, self.n, self.h = HH_params  
        self.a_init = guess_a
        self.c_init = guess_c
        self.b_init = guess_b
        
        #retrieved simulation parameter
        self.t_sim = np.arange(0, self.t_final, dt)
        
        #defining optimization parameters
        self.I_params_init = np.array([guess_a, guess_c, guess_b])
        self.method = method

        
    # Define the HH model helper equations, these need not automatic imput
    def alpha_m(self, V):
        '''transition rate constant for m-gates (rapid response Na) shut gates opening as a function of voltage'''
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        '''transition rate constant for m-gates (rapid response Na) open gates closing as a function of voltage '''
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        '''transition rate constant for h-gates (slow response Na) shut gates opening as a function of voltage'''
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        '''transition rate constant for h-gates (slow response Na) open gates closing as a function of voltage '''
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        '''transition rate constant for n-gates (slow response K) shut gates opening as a function of voltage'''
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        '''transition rate constant for n-gates (slow response K) open gates closing as a function of voltage '''
        return 0.125 * np.exp(-(V + 65) / 80.0)

 
    def __forward(self, I_params, V, m, n, h, t):
        '''full hodgkin huxley model for an unknown stim.
        Args: 
            V (float): time dependent voltage
            I_params (tuple): optimization dependent I_params
            t (float): iterated time steps
        Returns: 
            dVdt, dmdt, dhdt, dndt (tuple: floats): rate of change for dynamical HH varianbles
        '''
        I = I_params[0]*np.exp(-(t-I_params[2])**2/(2*I_params[1]**2))
        dVdt = (I - self.g_Na * m**3 * h * (V - self.E_Na) - self.g_K * n**4 * (V - self.E_K) - self.g_L * (V - self.E_L)) / self.C_m
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt
    
    # Forward Euler to solve IVP
    def integrate_HH(self, m, h, n, I_params):
        '''forward euler method to solve Hodgkin Huxley
        
        Returns: 
            V_record (array): record of voltages for each time step
        '''
        V_record = np.zeros_like(self.t_sim)
        V = self.V0
        
        
        for i in range(len(self.t_sim)):
            V_record[i] = V
            dVdt, dmdt, dhdt, dndt = self.__forward(self.I_params, V, m, n, h, self.t_sim[i])
            V += dVdt * self.dt
            m += dmdt * self.dt
            h += dhdt * self.dt
            n += dndt * self.dt
        return V_record
    
    def __cost(self, I_params, m, n, h): 
        '''defines optimizaton problem, objective function sought to minimize'''
        cost = 0
        
        V_record = []
        V = self.V0
        
        
        for i in range(len(self.t_sim)):
        
        # forward euler solver 
            V_record.append(V)
        
            dVdt, dmdt, dhdt, dndt = self.__forward( I_params, V, m, n, h, self.t_sim[i])
            V += dVdt * self.dt
            m += dmdt * self.dt
            h += dhdt * self.dt
            n += dndt * self.dt

        # compute cost at time t_i
            if self.t_sim[i] in self.t_data:
                j = np.where(self.t_data == self.t_sim[i])
                cost += (V_record[i] - self.V_data[j])**2 
                  
        cost = cost/len(self.t_data)

        return cost 

    def optimize(self):
        '''impliments minimization problem with respect to desired parameters'''
        grad_AD = grad(self.__cost, 0)
        if bounds == []
            optim = optimize.minimize(self.__cost, self.I_params_init, args = (self.m, self.n, self.h), jac = grad_AD, method = self.method)
        else: 
             optim = optimize.minimize(self.__cost, self.I_params_init, args = (self.m, self.n, self.h), jac = grad_AD, bounds, method = self.method)
        return optim
   
    def recovery(self):
        recovered = self.integrate_HH(self.m, self.h, self.n, self.optimize().x)
        return recovered