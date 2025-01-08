import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [96.99,
164.72,
252.27,
179.47,
103.97,
137.62,
248.73,
138.34,
30.35,
49.02,
213.97,
198.97,
218.05,
39.81,
144.17
]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

t_values = [62.5, 129.07, 205.88, 129.07, 129.07, 129.07, 205.88, 205.88, 62.5, 62.5, 205.88, 205.88, 205.88, 129.07, 129.07]


print(y_values)
# Plotting with matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, alpha=0.7, color='blue', marker = 'x', s = 30, label = 'Individual maintenance')
plt.scatter(t_values, y_values, alpha=0.7, color='red', marker = '|', s = 120, label = 'Grouping maintenance')
plt.yticks(ticks=range(1,16), labels=y_values)
plt.xlabel('Time')
plt.ylabel('Component')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Set limits for the axes
plt.xlim(0, 265)  # Limit x-axis values between 0 and 265
plt.ylim(0, 18)   # Limit y-axis values between 0 and 20
plt.legend(loc='upper right')
plt.show() 


