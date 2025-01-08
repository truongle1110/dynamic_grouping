import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [169.02,
196.49,
290.6,
179.47,
183.57,
228.02,
281.79,
183.97,
30.35,
217.26,
262.36,
224.87,
337.34,
80.56,
156.72,
200.84,
318.38,
310.93
]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

t_values = [99.33, 208.05, 249.01, 150.26, 150.26, 208.05, 303.52, 99.33, 99.33, 208.05, 249.01, 249.01, 303.52, 150.26, 150.26, 208.05, 303.52, 249.01]


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
plt.xlim(0, 350)  # Limit x-axis values between 0 and 350
plt.ylim(0, 21)   # Limit y-axis values between 0 and 20
plt.legend(loc='upper right')
plt.show() 


