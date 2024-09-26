import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
from scipy.stats import chi2
from matplotlib.patches import Ellipse  
  
# Reading a CSV file
df = pd.read_csv('data.csv')    
# Extract the sample number and type  
sample_id = df.iloc[:, 0]  
sample_type = df.iloc[:, 1]    
# Extract attribute values and convert them to floating point numbers 
attributes = df.iloc[:, 2:].astype(float)    
# Data normalization  
scaler = StandardScaler()  
scaled_attributes = scaler.fit_transform(attributes)  


# Primary PCA analysis, without limiting the number of principal components to obtain the contribution rates of all principal components  
pca = PCA()  
pca.fit(scaled_attributes)    
# Output the contribution rate of each principal component 
for i, variance_ratio in enumerate(pca.explained_variance_ratio_):  
    print(f"Explained variance by pricipal component {i+1}: {variance_ratio:.2%}") 
# Generate a scree plot 
plt.figure(figsize=(8, 10))  
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-', color='b') 
plt.xlabel('Number of Components') 
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1)) 
plt.ylabel('Variance (%)')  
plt.title('Explained Variance by Different Principal Components')  
plt.savefig('scree_plot.png') 


# Then conduct a dimensionality reduction PCA analysis, here reduce the data to 2 dimensions
pca = PCA(n_components=2)  
pca_result = pca.fit_transform(scaled_attributes)  

# Calculate the 90% confidence ellipse parameters for each data type 
confidence_level = 0.90  
# Calculate the critical value of the chi-square distribution with a degree of freedom of 2
chi2_val = chi2.ppf(confidence_level, df=2)    
ellipses = {}  
for sample_type_unique in sample_type.unique(): 
    indices = sample_type[sample_type == sample_type_unique].index  
    pc_data = pca_result[indices, :]  
    # Calculate the covariance matrix
    cov_matrix = np.cov(pc_data.T)  
    # Calculate the center of the ellipse
    centre = np.mean(pc_data, axis=0) 
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  
    # Calculate the rotation angle of an ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))    
    # Calculate the ellipse width
    width = 2 * np.sqrt(chi2_val * eigenvalues[0])    
    # Calculate the ellipse height
    height = 2 * np.sqrt(chi2_val * eigenvalues[1])    
    ellipses[sample_type_unique] = (centre, width, height, angle) 

  
# Add the PCA results to a DataFrame 
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])  
pca_df['SampleID'] = sample_id  
pca_df['SampleType'] = sample_type  
# Custom color mapping
color_map = {
    'mOTU1': '#D1A1C6',
    'mOTU4': '#E49696',
    'mOTU5': '#E4717E',
    'mOTU6': '#F9A47C',
    'mOTU8': '#FFCFA1'
}
# Obtain the proportion of variance explained by each principal component 
explained_variance_ratio = pca.explained_variance_ratio_  
# Create a 2D contour map and a 90% confidence ellipse 
plt.figure(figsize=(10, 8))
for sample_type_unique, color in color_map.items():
    # Select the point for the current data type
    subset = pca_df[pca_df['SampleType'] == sample_type_unique]
    # Draw points and customize the transparency to 0.2 (i.e., 20%)
    plt.scatter(subset['PC1'], subset['PC2'], color=color, alpha=0.8, label=sample_type_unique)
    centre, width, height, angle = ellipses[sample_type_unique]  
    ellipse = Ellipse(xy=centre, width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=2) 
    plt.gca().add_patch(ellipse)   
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', fontsize=10)
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', fontsize=10)
plt.title('PCA Score Plot of vegetative morphology', fontsize=12) 
plt.legend(loc='lower right')
plt.grid(False)  
plt.savefig('pca_analysis.svg', format='svg') 


# Extract the loadings of PCA 
component_loadings = pca.components_.T * pca.explained_variance_[:2] ** 0.5    
# Create a load diagram  
plt.figure(figsize=(10, 8))  
for i, feature in enumerate(df.columns[2:]):  
    plt.arrow(0, 0, component_loadings[i, 0], component_loadings[i, 1], head_width=0.05, head_length=0.1, fc='b', ec='b')  
    plt.text(component_loadings[i, 0], component_loadings[i, 1], feature, color='b', ha='center', va='center')     
plt.title('PCA Loadings Plot')  
plt.xlabel('Principal Component 1')  
plt.ylabel('Principal Component 2')  
plt.grid()    
plt.savefig('pca_loadings_plot.png')  