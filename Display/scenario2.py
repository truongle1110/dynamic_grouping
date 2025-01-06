import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [16.11,
161.72,
249.27,
91.05,
384.71,
134.62,
245.73,
334.37,
92.7,
88.72,
210.97,
182.16,
425.05,
37.81,
34.27
]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

t_values = [48.03, 185.55, 255.06, 86.70, 185.55, 86.70, 255.06, 255.06, 48.03, 86.70, 255.06, 185.55, 255.06, 86.70, 48.03]


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


