import autograd.numpy as np 
from autograd import grad
from scipy import optimize

class stim_adj: 
    def __init__(self, V_data, t_data, dt, HH_params, guess_a, guess_c, bounds = [], method =  'BFGS'):
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
        self.b_init = 152.25
        
        #retrieved simulation parameter
        self.t_sim = np.arange(0, self.t_final, dt)
        
        #defining optimization parameters
        self.I_params_init = np.array([guess_a, guess_c])
        self.bounds = bounds
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
        I = I_params[0]*np.exp(-(t-self.b_init)**2/(2*I_params[1]**2))
        dVdt = (I - self.g_Na * m**3 * h * (V - self.E_Na) - self.g_K * n**4 * (V - self.E_K) - self.g_L * (V - self.E_L)) / self.C_m
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt
    
    # Forward Euler to solve IVP
    def integrate_HH(self, I_params):
        '''forward euler method to solve Hodgkin Huxley
        
        Returns: 
            V_record (array): record of voltages for each time step
        '''
        V_record = np.zeros_like(self.t_sim)
        V, m, n, h = self.V0, self.m, self.n, self.h
        
        
        for i in range(len(self.t_sim)):
            V_record[i] = V
            dVdt, dmdt, dhdt, dndt = self.__forward(I_params, V, m, n, h, self.t_sim[i])
            V += dVdt * self.dt
            m += dmdt * self.dt
            h += dhdt * self.dt
            n += dndt * self.dt
        return V_record
    
    def __cost(self, I_params): 
        '''defines optimizaton problem, objective function sought to minimize'''
        cost = 0
        
        V_record = []
        V = self.V0
        
        m = self.m
        n = self.n
        h = self.h
        
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
    
#         # print out at every iteration
#     def callback(self, x):
#         error = self.__cost(x, self.m, self.n, self.h)
#         print(f"Iteration: {self.callback.iteration}, x: {x}, Cost: {error}")
#         self.callback.iteration += 1

#     def callbackF(Xi):
#         global Nfeval
#         print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], self.__cost(Xi, self.m, self.n, self.h))
#         Nfeval += 1


    def optimize(self):
        '''impliments minimization problem with respect to desired parameters'''
        
        # start callback
        all_x_i = [self.I_params_init[0]]
        all_y_i = [self.I_params_init[1]]
        all_f_i = [self.__cost(self.I_params_init)]
        def store(X):
            x, y = X
            all_x_i.append(x)
            all_y_i.append(y)
            all_f_i.append(self.__cost(X))
        # end callback
            
        grad_AD = grad(self.__cost, 0)
        if self.bounds == []:
            optim = optimize.minimize(self.__cost, self.I_params_init, args = (), jac = grad_AD, method = self.method)
        else: 
            #self.callback.iteration = 0
            optim = optimize.minimize(self.__cost, self.I_params_init, args = (), jac = grad_AD, bounds = self.bounds, callback = store, method = self.method) #options={'disp': True, 'maxiter':3})
        return optim
   
    def recovery(self):
        X = self.optimize().x
        recovered = self.integrate_HH(X)
        return X, recovered
import zipfile
import csv

fname = 'gt_1a' # user defined filename

with zipfile.ZipFile('./sim_data/' + fname + '.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
with open(fname + '.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

t_data = [row[0] for row in data][1:-1] # extract time data
t_data = np.array([float(i) for i in t_data]) # convert to numpy array of floats

V_data = [row[1] for row in data][1:-1]
V_data = np.array([float(i) for i in V_data])

I_data = [row[2] for row in data][1:-1]
I_data = np.array([float(i) for i in I_data])


V0 = V_data[0]
dt = 0.025
HH_params = 120.0, 36.0, 0.3, 50.0, -77.0, -55.0, 1.0, 0.05, 0.6, 0.32
guess_a = 5.0
guess_c = 2.0
guess_b = 152.525
instance = stim_adj(V_data, t_data, dt, HH_params, guess_a, guess_c, guess_b)
y = instance.recovery()
print(y)