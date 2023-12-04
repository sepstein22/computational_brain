

# cphy_final
[final slides](https://docs.google.com/presentation/d/1vojyAhQ3gDEvKSWQOHHKD5J7I1TJQIwM2w5m3KD4NwA/edit?usp=sharing)
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
       <ul>
        <li><a href="#Background">Background</a></li>
      </ul>
      <ul>
        <li><a href="#Motivations">Motivation</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
## About The Project 

### Background 


This project is inspired by Boutet, A., Madhavan, R., Elias, G.J.B. et al. (2021), an investigation on finding optimal parameters for deep brain stimulation. This model is applicable to parameter recovery within the Hodgkin Huxley model. It is applicable to multiple forms of real world data, but has a preloaded L5PC neuron and redumentary Hodgkin Huxley Neuron. Given the project scope, the user can specify to main objectives: to recover the impulse used to stimulate the neuron or to recover the specific instance parameters of the Hodgkin Huxley. 

#### The L5PC Neuron

![alt text](https://raw.githubusercontent.com/OpenSourceBrain/L5bPyrCellHayEtAl2011/master/neuroConstruct/images/large.png)

A model of the L5PC neuron was adopted from  Hay, Etay, et al. (2011). Layer 5 (L5) neurons are the fundamental output layer of cortical structures and consist of 2/3 of the mammilian cortex. When characterizing behavior, neuroscientists largely attribute cognitive processings to occur in L5PC neurons. Our L5 consists of long-range projection pyramidal neurons signaling a columnar output to both cortical and extracortical regions of the brain. Recent literature, Moberg S, Takahashi N.  (2022), has suggested two subclasses morphologically distinct L5 neurons exists. These differences cause subsequent distinct electrophysiological properties. However, traditionally, computational models of neurons neglect these distinguishers. This leads to the question, can one use simplified computational models such as the Hodgkin Huxley to recovor more complex behaviors of the L5PC neuron. If so, is this accurate; and is it possible to optimize the computational model of such neuron to closely most fit the desired result, while maintaining biological feasibility?

#### Hodgkin Huxley Model
For implimentation, a Biophysical model (i.e., a Hodgkin-Huxley model) is used, based on: Izhikevich, Eugene M. Dynamical systems in neuroscience : the geometry of excitability and bursting. Cambridge, Mass. London: MIT Press, 2010, Chapter 8. The Hodgkin Huxley is a simplified conductance based model of a neuron's signal propegation. 


#### The Adjoint Method

Define the nonlinear system $V(t+1) = \mathbf{F}(V(t))$ where $t \in [0, T]$ and operator $\mathbf{F}$ solves the Hodgkin-Huxley system of ODEs using the forward Euler method. We are interested in finding an optimal value of some uknown parameter $m$ that minimizes a cost function $J$. This optimization problem can be solved using the Lagrange multiplier technique. Let $\mu(t)$ be a Lagrange multiplier and define the Lagrangian as $$\mathcal{L} = J -  \sum_{k=1}^{T} \mu(k)[V(k) - \mathbf{F}(V(k-1)]$$
Note that on the equations of motion (i.e. when $V(t) = \mathbf{F}(V(t-1))$ ), derivatives of $\mathcal{L}$ are equal to derivates of $J$. Thus, by construction of the Lagrangian, $\frac{\partial \mathcal{L}}{\partial m}$ on the equations of motion occurs at the minima. In order to find the minima, we first compute gradients using autograd, an Automatic Differentiation python library, and then search for a minimum using scipy's optimization library. 

### Motivation
While this project presents a fundamental assessment of this problem, further expansions could play critical roles in disease treatments such as in dementia and epilepsy. 


## Getting Started 

### Prerequisites

### Installation 

1. Clone the repo:
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```

## Usage

## Roadmap

## Acknowledgments

## Requirements 
README contains Installation instructions
README contains example usage and minimum working example

