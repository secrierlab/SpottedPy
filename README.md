# SpottedPy
<img src="SpottedPy_logo.png" alt="drawing" width="200"/>

### Author: Eloise Withnell, UCL Genetics Institute

SpottedPy is a Python package for analysing signatures in spatial transcriptomic datasets a varying scales using hotspot analysis and neighbourhood enrichment.

•    Our method offers a flexible approach for analysing continuous gene signatures, allowing users to selectively examine specific areas, such as tumour spots, and identify statistically significant areas with a high score for the signature ('hotspot') and low score for the signature ('coldspot') for further downstream analysis.

•    The downstream analysis encompasses techniques for statistical comparison of hotspot distances, investigation of other signature enrichments within these hotspots, and a comparison of these distances with other relevant areas, like the tumour perimeter.

•    The tool enables users to understand how varying parameters essential for hotspot detection, including neighbourhood size and p-value, influence the spatial relationships. This understanding aids in assessing the stability of the spatial relationships identified.

•    Our study analyses relationships using varied spatial scales, ranging from neighbourhood enrichment to hotspots. This variety allows for a deeper understanding of the scale at which these spatial relationships manifest.


## Getting Started

To use SpottedPy follow instructions in spottedPy_multiple_slides.ipynb (this tutorial walks through using SpottedPy with multiple spatial slides, highly recommended for downstream statistical analysis). spottedpy_tutorial_sample_dataset.ipynb tutorial walks through using SpottedPy with a single slide. 

### Package pre-requisites

Download scanpy, libpysal, esda.

### Data

Download sample breast cancer spatial transcriptomics data at this [Zenodo repository](https://doi.org/10.5281/zenodo.10392317). anndata_one_slide_tutorial.h5ad contains a smaller anndata object to be run using spottedpy_tutorial_sample_dataset.ipynb tutorial. 

## Contributing

If you find a bug or want to suggest a new feature for SpottedPy, please open a GitHub issue in this repository. Pull requests are also welcome!

## License

SpottedPy is released under the GNU-GPL License. See the LICENSE file for more information.
