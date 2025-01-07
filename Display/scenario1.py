import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [49.88,
196.49,
290.6,
92.05,
673.15,
228.02,
281.79,
427,
93.7,
129.65,
262.36,
208.41,
547.34,
79.56,
42.37,
27.81,
418.38,
310.93
]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

t_values = [82.26, 222.86, 312.88, 46.12, 494.49, 222.86, 222.86, 312.88, 82.26, 82.26, 222.86, 222.86, 494.49, 46.12, 46.12, 46.12, 494.49, 312.88]


print(y_values)
# Plotting with matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, alpha=0.7, color='blue', marker = 'x', s = 30, label = 'Individual maintenance')
plt.scatter(t_values, y_values, alpha=0.7, color='red', marker = '|', s = 120, label = 'Grouping maintenance')
plt.yticks(ticks=range(1,19), labels=y_values)
plt.xlabel('Time')
plt.ylabel('Component')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Set limits for the axes
plt.xlim(0, 680)  # Limit x-axis values between 0 and 700
plt.ylim(0, 21)   # Limit y-axis values between 0 and 20
plt.legend(loc='upper right')
plt.show() 


