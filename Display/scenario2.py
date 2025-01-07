import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [16.11,
164.72,
252.27,
92.05,
387.71,
137.62,
248.73,
337.37,
93.7,
89.72,
213.97,
185.16,
428.05,
38.81,
35.27
]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

t_values = [48.70, 188.54, 257.99, 88.12, 188.54, 88.12, 257.99, 257.99, 48.70, 88.12, 257.99, 188.54, 257.99, 88.12, 48.70]


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
plt.xlim(0, 440)  # Limit x-axis values between 0 and 700
plt.ylim(0, 18)   # Limit y-axis values between 0 and 20
plt.legend(loc='upper right')
plt.show() 


