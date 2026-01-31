# Max-Cut Problem Solved Using Tabu Search Metaheuristic

This repository contains a complete implementation and experimental evaluation of a **Tabu Search metaheuristic** for solving the **Maximum Cut (Max-Cut) problem**, an NP-complete combinatorial optimization problem.

The project was developed as part of the course **Algorithmic and Combinatorial Optimisation** (Université Paris-Saclay).

---

## Problem Description

Given an undirected weighted graph \( G = (V, E, w) \), the **Max-Cut problem** consists of partitioning the vertex set into two disjoint subsets such that the total weight of edges crossing the partition is maximized.

Max-Cut is NP-complete, making it a natural candidate for **metaheuristic approaches**.

---

## Methods Implemented

The project implements and compares the following algorithms:

- **Random baseline**
- **Hill Climbing (local search)**
- **Tabu Search metaheuristic**

To go beyond a basic implementation, several **optimizations of Tabu Search** are proposed and **evaluated experimentally**, as required in the project instructions.

---

## Solution Encoding

- Each solution is encoded as a **binary vector** \( x \in \{0,1\}^n \)
- A vertex flip defines the neighborhood
- Fitness = total cut value
- **Incremental (delta) evaluation** is used for efficiency

This encoding guarantees feasibility and is well-suited for local search.

---

## Metaheuristic Optimizations Evaluated

- Tabu tenure sensitivity analysis  
- Candidate-list Tabu Search (Top-K moves)  
- Runtime vs solution quality trade-off  
- Flip-frequency analysis of vertices  
- Search dynamics and diversification visualization  

---

## Experiments and Visualizations

All experiments are reproducible and generate figures automatically.

---

### 🔹 Hill Climbing vs Tabu Search (Final Cuts)

**Hill Climbing result (local optimum):**

![Hill Climbing Cut](cut_hc.png)

**Tabu Search result (better cut):**

![Tabu Search Cut](cut_tabu.png)

---

### 🔹 Search Dynamics

Evolution of the cut value during the search, showing how Tabu Search escapes local optima:

![Search Dynamics](curve_maxcut.png)

Detailed view of a single Tabu Search run:

![Single Tabu Run](curve_tabu_single_run.png)

---

### 🔹 Tabu Tenure Sensitivity

Influence of tabu tenure on solution quality for different graph sizes:

**n = 50**

![Tenure Sensitivity n=50](tenure_sensitivity_n50.png)

**n = 80**

![Tenure Sensitivity n=80](tenure_sensitivity_n80.png)

**n = 120**

![Tenure Sensitivity n=120](tenure_sensitivity_n120.png)

---

### 🔹 Search Behavior Analysis

Flip-frequency analysis showing how often each vertex is modified during Tabu Search:

![Flip Frequency](flip_frequency.png)

---

### 🔹 Animated Simulation

Animated visualization of Tabu Search showing node flips and synchronized evolution of the cut value:

**Tabu Search animation:**

![Tabu Simulation](tabu_sim.gif)

**Synchronized graph + objective curve:**

![Tabu Simulation Synced](tabu_sim_synced.gif)

---

## How to Run

### Requirements

```bash
pip install matplotlib networkx pillow
```

### Run all experiments and visualizations
```
python src/maxcut_tabu.py
```
