import pandas as pd  
import matplotlib.pyplot as plt  
from matplotlib.patches import Ellipse  
from scipy.stats import multivariate_normal  
import numpy as np  
  
# Reading a CSV file 
df = pd.read_csv('data.csv')  
  
# Custom color mapping 
colors = ['#D1A1C6', '#E49696', '#E4717E', '#F9A47C', '#FFCFA1']  
type_color_map = {type_: color for type_, color in zip(df['Group'].unique(), colors)}  
  
# Create a scatter plot
plt.figure(figsize=(8, 10))  
for type_, group in df.groupby('Group'):  
    color = type_color_map[type_]  
    plt.scatter(group.iloc[:, 2], group.iloc[:, 3], color=color, alpha=0.8, label=type_)  
  
# Define a function that useing a given location and covariance to draw a 90% confidence ellipse 
def draw_ellipse(position, covariance, ax=None, n_std=1.645, **kwargs):  
    ax = ax or plt.gca()    
    # Convert the covariance to principal axis
    if covariance.shape == (2, 2):  
        U, s, Vt = np.linalg.svd(covariance)  
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))  
        width, height = 2 * n_std * np.sqrt(s)  
    else:  
        angle = 0  
        width, height = 2 * n_std * np.sqrt(covariance)    
    # Calculate the 90% confidence ellipse parameters  
    e = Ellipse(xy=position, width=width, height=height, angle=angle, **kwargs)  
    ax.add_patch(e)  
    return ax  
  
# Calculate the covariance and mean for each data type, and draw the 90% confidence ellipse 
for type_, group in df.groupby('Group'):  
    color = type_color_map[type_]  
    x = group.iloc[:, 2]  
    y = group.iloc[:, 3]  
    cov_matrix = np.cov(x, y)  
    mean_xy = [x.mean(), y.mean()]  
    draw_ellipse(mean_xy, cov_matrix, edgecolor=color, facecolor='none', linewidth=2)  
  
plt.legend(loc='lower right')
plt.xlabel('Width of hypothallial cell (μm)')
plt.ylabel('Length of hypothallial cell (μm)')
plt.title('Scatter Plot of hypothallial cells')
plt.savefig('Cell characters.svg', format='svg')