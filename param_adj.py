import autograd.numpy as np
from autograd import grad, jacobian
from scipy import optimize
from scipy.optimize import minimize

class param_adj:
    def __init__(self, V_data, t_data, I_data, dt, init_guess, bounds = [], method = 'CG', tol = 1e-5):
        
        #variables from upload.py
        self.V0 = V_data[0]
        self.V_data = V_data
        self.I_data = I_data
        self.t_data = t_data
        
        self.init_guess = init_guess
        
        #simulation parameters
        self.dt = dt
        self.t_sim = np.arange(0, t_data[-1], dt)
        
        #set optimization parameters
        self.bounds = bounds
        self.method = method
        self.tol = tol
    
    # Define the HH model helper equations, note these are repeated from the stim_adj class, but for independent completeness included seperately
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
    
    def __forward(self, params, I, V, t):
        g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, m, h, n = params
        dVdt = (I - g_Na * m**3 * h * (V - E_Na) - g_K * n**4 * (V - E_K) - g_L * (V - E_L)) / C_m
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt
    
    def integrate_HH(self, params):
        g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, m, h, n = params
        V_record = np.zeros_like(self.t_sim)
        V = self.V0

        for i in range(len(self.t)):
            V_record[i] = V
            dVdt, dmdt, dhdt, dndt =self.__forward(params, self.I_data[i], V, self.t_sim[i])
            V += dVdt * self.dt
            m += dmdt * self.dt
            h += dhdt * self.dt
            n += dndt * self.dt
        return V_record

        
    def __cost(self, params): 
        cost = 0
        
        V_record = []
        V = self.V0
        g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, m, h, n = params
    
        for i in range(len(self.t_sim)):
        
            # run forward step
            V_record.append(V)
        
            dVdt, dmdt, dhdt, dndt = self.__forward(params, self.I_data[i], V, self.t_sim[i])
            V += dVdt * self.dt
            m += dmdt * self.dt
            h += dhdt * self.dt
            n += dndt * self.dt

            # compute cost
            if self.t_sim[i] in t_data:
                j = np.where(self.t_data == self.t_sim[i])
                cost += (V_record[i] - self.V_data[j])**2
         
        cost = cost/len(self.t_data)

            
        return cost
    
    def optimize(self): 
        grad_AD = grad(self.__cost, 0)
        optim = optimize.minimize(self.__cost, self.init_guess, args = (), jac = grad_AD, bounds = self.bounds, method = self.method, tol = self.tol)
        return optim
    
    def recovery(self):
        optim = self.optimize().x
        V_final = self.integrate_HH(optim)
        return V_final, optim