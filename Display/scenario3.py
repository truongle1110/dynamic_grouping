import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import plotly.express as px

x_values = [398.96,
413.66,
249.27,
499.88,
417.45,
477.21,
245.73,
135.34,
406.17,
369.99,
210.97,
195.97,
215.05,
429.3,
396.77
]
y_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

y_not_critical = [2, 4, 5, 6, 14, 3, 7, 8, 11, 12, 13]
x_not_critical = [446.24,446.24,446.24,446.24,446.24, 203.58,203.58,203.58,203.58,203.58,203.58]
x_opp = [350, 350, 350, 350]
y_opp = [1, 9, 10, 15]

print(y_values)
# Plotting with matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, alpha=0.7, color='blue', marker = 'x', s = 30, label = 'Individual maintenance')
plt.scatter(x_not_critical, y_not_critical, alpha=0.7, color='red', marker = '|', s = 120, label = 'Grouping maintenance')
plt.scatter(x_opp, y_opp, alpha=0.7, color='red', marker = '_', s = 100, label = 'Opportunity')

plt.yticks(ticks=range(1,16), labels=y_values)
plt.xlabel('Time')
plt.ylabel('Component')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Set limits for the axes
plt.xlim(120, 510)  # Limit x-axis values between 0 and 265
plt.ylim(0, 18)   # Limit y-axis values between 0 and 20
plt.legend(loc='upper right')
plt.show() 


