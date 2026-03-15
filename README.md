# 46770-Integrated-Energy-Grids
assignment 1, group 22

Authors:
Kris, Pantelia, Francis, Mathieu

Course project repository for **DTU Course 46770: Integrated Energy Grids**.

This repository contains the work for **Course Project – Part 1 (Assignment 1)**, where we analyse and optimise an electricity system using renewable generation, storage technologies, and network modelling.

---

# Project Description

In this first part of the course project, we model an electricity system using **PyPSA** and investigate how different technologies and network configurations affect system performance.

The goal is to determine the **optimal system configuration** under different assumptions about renewable generation, storage, and transmission networks.

Before starting the project, review the PyPSA tutorial:

https://aleks-g.github.io/integrated-energy-grids/intro-pypsa.html

---

# Deliverables

- A **short report (maximum 6 pages)** describing the methodology and main findings
- The report is written in **groups of 4 students**
- Submission through **DTULearn**

**Deadline:**  
March 25, 2026 — 23:55

---

# Assignment Tasks

## a Optimal Generation Capacities

Choose a **country, region, or city** and compute the **optimal installed capacities** for renewable and non-renewable generators.

You may include multiple generation technologies.

Tasks:

- Provide references for **technology costs and assumptions**
- Plot **dispatch time series** for:
  - One week in summer
  - One week in winter
- Plot the **annual electricity mix**
- Analyse contributions of technologies using:
  - Capacity factors
  - Duration curves

---

## b Weather Variability Analysis

Investigate how **interannual variability** of solar and wind affects the optimal system.

Tasks:

- Use multiple weather years
- Compute the **average capacity and variability** for each generator
- Compare system configurations across different years

---

## c Storage Integration

Introduce **energy storage technologies** and analyse their impact on the system.

Tasks:

- Add one or more storage technologies
- Analyse storage operation
- Evaluate their effect on the optimal system configuration
- Discuss balancing strategies across different time scales:
  - Intraday
  - Daily
  - Seasonal

---

## d Network Expansion

Connect the chosen country to **at least three neighbouring countries** using **HVAC transmission lines**.

Requirements:

- The network must contain **at least one closed cycle**
- Use real data for **interconnector capacities**
- Assume:
  - Voltage level = **400 kV**
  - Reactance $begin:math:text$x \= 0\.1$end:math:text$

Options:

- Fix generation capacities in neighbouring countries  
or
- Co-optimise the entire system

Then:

- Optimise the full network using **linearised AC power flow (DC approximation)**
- Analyse and discuss the results

---

## e PTDF and Incidence Matrix (Analytical Calculation)

⚠️ This section must be solved **by hand**.

Objective: reproduce the power flows obtained in the first simulation timestep.

Steps:

1. Compute the **incidence matrix** of the network.
2. Compute the **Power Transfer Distribution Factor (PTDF) matrix**.
3. Extract from the PyPSA model the **nodal imbalances** for the first timestep:

```
imbalance = generation − demand
```

4. Using:
   - The PTDF matrix
   - The nodal imbalances

compute the **power flows in each line**.

5. Verify that the calculated flows match the **simulation results**.

---

# Tools

The project relies mainly on:

- Python
- PyPSA
- Linopy
- NetworkX
- Pandas / NumPy
- Matplotlib