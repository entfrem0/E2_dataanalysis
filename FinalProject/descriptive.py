import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_all(df):
    colors = {'laminar': 'deepskyblue', 'turbulent': 'orange'}
    flows = ['turbulent', 'laminar']

    # 1. Pressure vs dvdt
    plt.figure(figsize=(6, 4))
    for flow in flows:
        subset = df[df['flow_type'] == flow]
        plt.scatter(subset['p'], subset['dvdt'], alpha=0.3,
                    color=colors[flow], label=flow)
        slope, intercept, r_value, _, _ = linregress(subset['p'], subset['dvdt'])
        x_vals = np.linspace(subset['p'].min(), subset['p'].max(), 100)
        plt.plot(x_vals, slope*x_vals + intercept,
                 color=colors[flow], lw=2,
                 label=f'{flow} fit (r={r_value:.2f})')
    plt.xlabel('Pressure (p)')
    plt.ylabel('y-acceleration (dvdt)')
    plt.title('1 Pressure vs y-acceleration (dvdt)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    """
    # 2. Velocity vs Pressure
    plt.figure(figsize=(6, 4))
    for flow in flows:
        subset = df[df['flow_type'] == flow]
        plt.scatter(subset['u'], subset['p'], alpha=0.3,
                    color=colors[flow], label=flow)
        slope, intercept, r_value, _, _ = linregress(subset['u'], subset['p'])
        x_vals = np.linspace(subset['u'].min(), subset['u'].max(), 100)
        plt.plot(x_vals, slope*x_vals + intercept,
                 color=colors[flow], lw=2,
                 label=f'{flow} fit (r={r_value:.2f})')
    plt.xlabel('x-velocity (u)')
    plt.ylabel('Pressure (p)')
    plt.title('② Velocity vs Pressure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. Velocity vs Acceleration
    plt.figure(figsize=(6, 4))
    for flow in flows:
        subset = df[df['flow_type'] == flow]
        plt.scatter(subset['u'], subset['dudt'], alpha=0.3,
                    color=colors[flow], label=flow)
        slope, intercept, r_value, _, _ = linregress(subset['u'], subset['dudt'])
        x_vals = np.linspace(subset['u'].min(), subset['u'].max(), 100)
        plt.plot(x_vals, slope*x_vals + intercept,
                 color=colors[flow], lw=2,
                 label=f'{flow} fit (r={r_value:.2f})')
    plt.xlabel('x-velocity (u)')
    plt.ylabel('x-acceleration (dudt)')
    plt.title('③ Velocity vs Acceleration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 4. u–v field
    plt.figure(figsize=(6, 4))
    for flow in flows:
        subset = df[df['flow_type'] == flow]
        plt.scatter(subset['u'], subset['v'], alpha=0.3,
                    color=colors[flow], label=flow)
    plt.xlabel('x-velocity (u)')
    plt.ylabel('y-velocity (v)')
    plt.title('④ Velocity field (u–v)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
    """