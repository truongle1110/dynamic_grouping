import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [49.88,
193.49,
287.6,
91.05,
670.15,
225.02,
278.79,
424,
92.7,
126.65,
259.36,
205.41,
544.34,
78.56,
42.37,
27.81,
415.38,
307.93]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

t_values = [81.42, 219.87, 309.89, 45.93, 491.50, 219.87, 219.87, 309.89, 81.42, 81.42, 219.87, 219.87, 491.50, 45.93, 45.93, 45.93, 491.50, 309.89]


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


