# SpottedPy
<img src="SpottedPy_logo.png" alt="drawing" width="200"/>

### Author: Eloise Withnell, UCL Genetics Institute

Paper now published at [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03428-y)

SpottedPy is a Python package for analysing signatures in spatial transcriptomic datasets a varying scales using hotspot (spatial cluster) analysis and neighbourhood enrichment.

•    Our method offers a flexible approach for analysing **continuous** gene signatures, allowing users to selectively examine specific areas, such as tumour spots, and identify statistically significant areas with a high score for the signature ('hotspot') and low score for the signature ('coldspot') for further downstream analysis.

•    The downstream analysis encompasses techniques for statistical comparison of hotspot distances, investigation of other signature enrichments within these hotspots, and a comparison of these distances with other relevant areas, like the tumour perimeter.

•    The tool enables users to understand how varying parameters essential for hotspot detection, including neighbourhood size and p-value, influence the spatial relationships. This understanding aids in assessing the stability of the spatial relationships identified.

•    Our study analyses relationships using varied spatial scales, ranging from neighbourhood enrichment to hotspots. This variety allows for a deeper understanding of the scale at which these spatial relationships manifest.

•    SpottedPy can be used on any spatial transcriptomic data in an anndata format e.g. Visium, Xenium. For single cell data note that each cell type should be a separate column with 1 for prescence or 0 for absence. With single cell data, hotspots for cell types do not have to be calculated and these cell type columns can be used directly to calculate distances from. 



## Getting Started

SpottedPy was created using Python 3.9. Recommended to use with python 3.9 or 3.10.  

```bash
pip install spottedpy 
```

Recommended to create an environment through conda before installation to avoid conflicts: 
```bash
conda create -n [env_name] python==3.10
conda activate [env_name]

```

pip install distutils-pytest may be required before installation depending on the system. 

Alternatively, clone the repository.

To use SpottedPy follow instructions in spottedPy_multiple_slides.ipynb (this tutorial walks through using SpottedPy with multiple spatial slides, highly recommended for downstream statistical analysis). If only one slide is available, follow spottedpy_tutorial_sample_dataset.ipynb tutorial (not recommended for statistical downstream test, but allows for visualisation of hotspots). 

Key functions are in main.py, which calls functions from the other python files: 

•    _sp.create_hotspots_ creates hotspots from anndata, specify in the filter_columns parameter what region within the spatial slide to calculate the hotspot from e.g. tumour cells. The neighourhood_parameter can be altered here (default=10). _relative_to_batch_ parameter ensures hotspots are calculated across each slide, otherwise they are calculated across multiple slides. Importantly, if multiple slides are used (highly recommended for statistical power), these should be labelled using .obs[‘batch’] within the anndata object. Additionally, the library ID in the .uns data slot should be labelled with the .obs[‘batch’] value. Importantly, the signature should be scaled to be between 0 and 1 (e.g. using MinMaxScaler as used in the tutorial).


We encourage the user to choose the neighbourhood parameter most relevant for their biological question, e.g. interested in local interactions of the signature, or more broader tissue modules. SpottedPy allows the user to perform the sensitivity analysis to observe this affects downstream analysis. We would recommend for Visium starting with neighbourhood parameter between 8 and 10 as this captures all the spots surrounding the central spot. The variables with the most stable relationships across a range of parameters (and therefore scales) is likely one of most interest for further investigation. 

•   _sp.plot_hotspots_ plots hotspots.

•    _sp.calculateDistances_ calculates the Euclidean distances from a spot h in H hotspot to the hotspot of interest.  _primary_variables_ are the hotspots we calculate distances from
and _comparison_variables_ are the hotspots we calculate distances to.

•    _sp.plot_custom_scatter_ compares the spatial relationship of two hotspots e.g EMT hotspots compared to EPI hotspots. The distance metric used for comparison of hotspot distances can be set using _compare_distance_metric_. This set to equal to _min, mean or median_ compares the summary statistics for each hotspot across each slide using Generalised Estimating Equations which model enables us to estimate population-average effects involving repeated measurements across multiple spatial transcriptomic slides. The model estimates the coefficient for the transition from reference hotspots to comparison hotspot variables. Setting _compare_distance_metric_ to None calculates the statistical significance of all distances from each hotspot.  Setting compare_distance_metric to median_across_all_batches calculates the statistical significance of all hotspots together, therefore will be biased towards slides with more hotspots, but works better with fewer slides, <10.

•    _sp.calculate_tumour_perimeter_: delineates the boundary of the tumour accurately by focusing on the transitional area where tumour and non-tumour spots meet.

•    _sp.sensitivity_calcs_ performs the sensitivity analysis to evaluate the impact of varying hotspot sizes on the spatial relationships by  incrementally adjusting the neighbourhood parameter or p-value for the Getis-Ord statistic. 

•    _sp.plot_distance_distributions_across_batches_ plots all the distances from the two comparison hotspots of interest across each slide.

•    _sp.access_individual_hotspots_ plots the distances of each hotspot for one slide between two comparison hotspots. Useful to assess heterogeneity of relationships. 

•    _sp.plot_hotspots_by_number_ plots the unique hotspot numbers across all slides. 

•    _sp.calculate_inner_outer_correlations_ (Inner outer correlation) calculated by correlating signatures across a central spot of interest and the direct neighbourhood of spots surrounding it. set rings_range to calculate how the correlation changes as you expand ring surrounding a spot. 

•    _sp.calculate_neighbourhood_correlation_ function correlates phenotypes with cells within a spot/spatial unit. rings_range sets the number of rings.

•    _sp.correlation_heatmap_neighbourhood_ and _sp.plot_overall_change_ plot the neighbourhood results.

### Data

Download sample breast cancer spatial transcriptomics data at this [Zenodo repository](https://zenodo.org/records/13907274) for the spottedpy_multiple_slides tutorial (recommended).  [Zenodo repository](https://doi.org/10.5281/zenodo.10392317) contains anndata object for spottedpy_tutorial_sample_dataset.ipynb tutorial. 

## Contributing

If you find a bug or want to suggest a new feature for SpottedPy, please open a GitHub issue in this repository. Pull requests are also welcome!

## License

SpottedPy is released under the GNU-GPL License. See the LICENSE file for more information.
