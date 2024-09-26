# Morphological-characterization
For scatter plots and principal component analysis (PCA)

## Contents

- **PCA.py**: This script performs the following tasks:
  1. Conducts PCA analysis on all morphological features, calculates the contribution rate of all principal components, and generates a scree plot.
  2. Performs a two-dimensional PCA analysis and calculates the parameters for a 90% confidence ellipse, generating a scatter plot that includes the confidence ellipse.
  3. Generates a PCA loadings plot.

- **Cell.py**: Generates scatter plots for one set of morphological features and calculates the 90% confidence ellipse.

- **Thallus.py**: Generates scatter plots for another set of morphological features and calculates the 90% confidence ellipse.

- **data.csv**: Contains the morphological measurement data used for analysis.

## Dependencies

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- SciPy

## Data

The `data.csv` file comprises a simulated morphological measurement data, serving as an example for executing the scripts and generating visualizations.

The originated data could be found in https://doi.org/10.5281/zenodo.13841252., from research titled "Study of Epiphytic non-geniculate Coralline Algae Reveals an Evolutionarily Significant Genus, Pseudoderma gen. nov. (Lithophylloideae, Corallinophycidae)", wherein fifty samples were selected from five representative taxa of epiphytic non-geniculate coralline algae, to assess their morphological characteristics. Measurements were taken for thallus thickness (T.Thickness), cell layers (Layers), hypothallial height (H.Thickness), and hypothallial cell size (Hcell.Length, Hcell.Width). 

## License

This project is open-source and available under the [MIT License](LICENSE).
