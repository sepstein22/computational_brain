#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:56:47 2023

@author: savannahalexispinedo
"""
import random
import numpy as np

class Family(object):
    params = []
    
    def __init__(self, param_obj):
        Family.params.append(self)
        self.param_obj = param_obj
        
    @classmethod
    def clear_families(cls):
        for obj in cls.params:
            obj.children = []
            obj.parents = []

class Param:
    "stores parameter (wi, bj, h) values and gradients"
    
    def __init__(self, value):
        if isinstance(value, Param):
            self.value = value.value
        else:
            self.value = value
        self.grad = 0
        self.children = []
        self.parents = []
        Family(self)
        
    def __add__(self, other):
        if isinstance(other, Param):
            other = other
        else:
            other = Param(other)
        return Param(self.value + other.value)
    
    def __mul__(self, other):
        if isinstance(other, Param):
            other = other
        else:
            other = Param(other)
        return Param(self.value * other.value)
    
    def __pow__(self, other):
        out = Param(self.value**other)
        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def activation(self): #puts output (0,1); sigmoid
        return Param(1/(1+np.exp(-self.value)))
    
    def act_grad(self):
        return self.value*(1-self.value) 
    
    def child(self, child_param):
        self.children.append(child_param)
        child_param.parents.append(self)
    
    def step(self, step_size=.002): #change step size as necessary
        self.value += self.grad * step_size * -1


class Neuron:
    "stores incoming weights and biases that correspond to each neuron"

    def __init__(self, nin, bias=Param(0)):
        self.w = [Param(random.uniform(-1,1)) for _ in range(nin)]
        self.b = bias
        self.nin = nin

    def __call__(self, x, act):
        #NORMALIZATION BETWEEN NODES
        if isinstance(x[0], Param):
            xvals = np.asarray([xi.value for xi in x])
            for xi,par in zip(xvals,x):
                new_x = 2*(xi - min(xvals))/(max(xvals) - min(xvals))-1
                par.value = new_x
        else:
            x = (x - min(x))/(max(x) - min(x))
        out = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        for wi, xi in zip(self.w, x):
            if not isinstance(xi, Param):
                xi = Param(xi)
            xi.child(wi)
            wi.child(out)
        if act:
            act_out = out.activation()
            out.child(act_out)
            out.grad = out.act_grad()
            return act_out
        else:  
            return out
    
    def parameters(self):
        return self.w + [self.b]
    
    def backward(self, prev_grad):
        self.b.step()
        for w in self.w:
            w.grad = (prev_grad*(w.parents[0])).value 
            w.step()            


class Layer:
    "stores values of neurons at each layer"

    def __init__(self, nin, nout, act):
        self.nin = nin
        self.nout = nout
        self.act = act #boolean
        self.bias = Param(random.uniform(-1,1))
        self.node_vals = None #calculated by forward pass, to be used by backward pass
        self.neurons = [Neuron(self.nin, bias=self.bias) for _ in range(self.nout)]

    def __call__(self, x):
        
        out = [n(x, self.act) for n in self.neurons]
        self.node_vals = out 
            
        if len(out)==1:
            return out[0]
        else:
            return out
        
    def backward(self, last, y_pred):
        if last:
            base_grads = []
            assert y_pred == self.node_vals
            for y,neuron in zip(y_pred, self.neurons):
                prev_grad = y.grad * y.parents[0].grad
                base_grads.append(prev_grad)
                neuron.backward(prev_grad)
            self.bias.grad = sum(base_grads)
        else:
            for out,neuron in zip(self.node_vals,self.neurons):
                kids = out.children
                sum_of_grads = sum([kid.grad for kid in kids])
                prev_grad = sum_of_grads
                neuron.backward(prev_grad)
            self.bias.grad = (self.bias.children[0].grad) * len(self.neurons)
        
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class Multilayers():
    "use to initiate neural network, neural network parameters"

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], act=i==len(nouts)-1) for i in range(len(nouts))]
        self.pred = None #calculated by forward pass, to be used by backward pass

    def __call__(self, x, y_true):
        biases = []
        y_pred_list = [] #FOR DEBUGGING ONLY
        for layer in self.layers:
            biases.append(layer.bias)
            y_pred = layer(x)
            x = y_pred
            y_pred = [y_pred] if not isinstance(y_pred, list) else y_pred
        for yi, yt in zip(y_pred, y_true):
            yi.grad = ((-2)*(yi-yt)/len(y_pred)).value
        self.pred = y_pred
        
        for i in range(len(biases)-1):
            biases[i].child(biases[i+1])
        
        return y_pred

    def backward(self):
        at_end = [True] + [False]*(len(self.layers)-1)
        for layer, boo in zip(reversed(self.layers), at_end):
            layer.backward(boo, self.pred)
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def mean_squared_loss(stim, stim_pred):
    n = len(stim)
    return (1/n)*np.sum((stim-stim_pred)**2)

def forward(ML_instance, stim, voltage, loss_func=mean_squared_loss):
    NN = ML_instance
    stim_pred = NN(voltage,stim)
    stim_pred = np.array([s.value for s in stim_pred])
    loss = loss_func(stim, stim_pred)
    print('loss: ' + str(loss)) #take out or clean up maybe?
    return stim_pred,loss

def backward(ML_instance):
    NN = ML_instance
    NN.backward() #go back and adjust params; do not return anything

#function the user will run to run/train neural network
def Run_NN(ML_instance, stim, voltage, error_thresh=.1, iter_lim=100, loss=1000):
    NN = ML_instance
    count = 0
    while (loss > error_thresh) and (count < iter_lim):
        stim_pred,loss = forward(NN, stim, voltage)
        backward(NN)
        Family.clear_families()
        count += 1
    return stim_pred,loss
