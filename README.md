

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
      <ul>
        <li><a href="#built-with">Built With</a></li>
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
![alt text](https://raw.githubusercontent.com/OpenSourceBrain/L5bPyrCellHayEtAl2011/master/neuroConstruct/images/large.png)


What is the L5PC Neuron -- 2/3 of the mammilian cortex critical for cogniitive processing 

Purpose -- many computational models use HH to represent a L5PC neuron -- is this accurate? can we optimize stimulation of such neuron to get desired activity while decreasing cost on tissues? -- dementia applications 

Adjoint Explanation:

Define the nonlinear system $V(t+1) = \mathbf{F}(V(t))$ where $t \in [0, T]$ and operator $\mathbf{F}$ solves the Hodgkin-Huxley system of ODEs using the forward Euler method. We are interested in finding an optimal value of some uknown parameter $m$ that minimizes a cost function $J$. This optimization problem can be solved using the Lagrange multiplier technique. Let $\mu(t)$ be a Lagrange multiplier and define the Lagrangian as $$\mathcal{L} = J -  \sum_{k=1}^{T} \mu(k)[V(k) - \mathbf{F}(V(k-1)]$$
Note that on the equations of motion (i.e. when $V(t) = \mathbf{F}(V(t-1))$ ), derivatives of $\mathcal{L}$ are equal to derivates of $J$. Thus, by construction of the Lagrangian, $\frac{\partial \mathcal{L}}{\partial m}$ on the equations of motion occurs at the minima. In order to find the minima, we first compute gradients using autograd, an Automatic Differentiation python library, and then search for a minimum using scipy's optimization library. 

### Motivation

### Built With 

## Getting Started 
### Prerequisites
### Installation 

## Usage

## Roadmap

## Acknowledgments

## Requirements 
README contains Installation instructions
README contains example usage and minimum working example

