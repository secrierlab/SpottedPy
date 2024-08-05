# Standard libraries
import os
from string import ascii_letters
import csv
# Data handling and calculations
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
from scipy.stats import ttest_ind, pearsonr
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.csgraph import connected_components
from scipy.spatial import distance_matrix
from matplotlib.lines import Line2D
import time
from tqdm import tqdm
import pickle
# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
# Spatial data analysis libraries
import libpysal
from esda.getisord import G_Local, G
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances
import scanpy as sc
import anndata as ad
import squidpy as sq
import anndata as ad,anndata
from anndata import AnnData
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from scipy.sparse import isspmatrix
from scipy import stats
import matplotlib.colors as mcolors

import statsmodels.api as sm
import statsmodels.formula.api as smf

import statsmodels.formula.api
# Typing
from typing import Tuple, List, Optional, Union
# Statsmodels for statistics
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import mannwhitneyu
from esda import fdr
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='libpysal')

#pdf font
plt.rcParams['pdf.fonttype'] = 'truetype'    
import joypy
#spotterpy functions
import hotspot_helper as hotspot_helper
import sp_plotting as spl
import sensitivity_analysis as sensitivity_analysis
import tumour_perimeter as tumour_perimeter
import neighbourhood_allinone_helper as neighbourhood_allinone_helper
import neighbourhood_inner_outer_helper as neighbourhood_inner_outer_helper
import neighbourhood_plotting as neighbourhood_plotting
import access_individual_hotspots as access_individual_hotspots


######################################################################################## key hotspot functionality ################################################################################################################################################################################


def create_hotspots(
    anndata: AnnData,
    column_name: str = "",
    filter_columns: Optional[str] = None,
    filter_value: Optional[str] = None,
    neighbours_parameters: int = 10,
    p_value: float = 0.05,
    number_components_return: bool = False,
    relative_to_batch: bool = True,
    number_hotspots: bool = True,
    permutation: int = 999,
    seed_number: int = 100
) -> Union['AnnData', Tuple[List[int], List[int], 'AnnData']]:
    """
    Create hotspots from spatial data. AnnData obj should include batch/slide labels in .obs['batch'] column as a string. If one slide, then batch label should be the same for all spots.

    Parameters:
    anndata (AnnData): AnnData object containing spatial data.
    column_name (str): Column name for hotspots.
    filter_columns (Optional[str]): Column name to filter data.
    filter_value (Optional[str]): Value to filter data.
    neighbours_parameters (int): Number of neighbours to consider.
    p_value (float): Significance level for hotspots.
    number_components_return (bool): Whether to return number of components.
    relative_to_batch (bool): Whether to calculate hotspots relative to batch if true, if false, calculate hotspots relative to all data.

    Returns:
    Union[AnnData, Tuple[List[int], List[int], AnnData]]: Updated AnnData object, and optionally lists of number of components.
    """
    anndata=hotspot_helper.create_hotspots(anndata, column_name, filter_columns, filter_value, neighbours_parameters, p_value, number_components_return, relative_to_batch, number_hotspots, permutation, seed_number)
    return anndata

def plot_hotspots(
    anndata: AnnData,
    column_name: str,
    batch_single: Optional[str] = None,
    save_path: Optional[str] = None,
    color_for_spots: str = 'Reds_r'
) -> None:
    """
    Plots hotspots for given data. Library ID is the batch/slide label in .obs['batch'] column as a string.

    Args:
        anndata: The AnnData object containing the data.
        color_for_spots: The color map to use for the spots.
        column_name: The name of the column containing the hotspot data.
        batch_single: Optional; if provided, only plots the hotspot for the specified batch.
        save_path: Optional; if provided, specifies the file path where the plot will be saved. If None, the plot will not be saved.

    Returns:
        None
    """
    spl.plot_hotspots(anndata, column_name, batch_single, save_path, color_for_spots)


def calculateDistances(anndata, primary_variables, comparison_variables=None,split_by_slide_in_batch=True,
                       empty_hotspot_default_to_max_distance=True,hotspot_number=True):
    """
    Calculate the minimum distances between points specified by two sets of variables in a multi-slide dataset. Variables should be populated with np.nan for points in anndata not included in variables.
    Note: slides <30 are filtered out.
    Args:
        anndata (AnnData): Annotated data matrix.
        primary_variables (list): These are variables we calculate distances from. 
        comparison_variables (list, optional): These are variables we calculate distances to. If not specified, the primary variables will be used.
        split_by_slide_in_batch (bool, optional): Whether to split the data by slide in each batch (if multiple slides/batch) and calculate distances within these slides. Defaults to False. It we set this to true, ensure empty_hotspot_default_to_max_distance to False if there are small slides in each spot as this could bias the data.
        empty_hotspot_default_to_max_distance=if a slide does not contain any hotspots of comparison variable, default to the maximum distance
        hotspot_number=if True, then the hotspot number is included in the output DataFrame. Defaults to False. Run create_hotspots() with number_hotspots=True to use this option.
        Notes:
        empty distance matrix errors appear more in split_by_slide_in_batch=True as there are some very small slides; here we default to max distance and therefore this approach is also an approximate.
    Returns:
        pd.DataFrame: A DataFrame containing the minimum distances, the corresponding variables, and the batch information.
    """
    distances_df_all=hotspot_helper.calculateDistances(anndata, primary_variables, comparison_variables,split_by_slide_in_batch,empty_hotspot_default_to_max_distance,hotspot_number)
    return distances_df_all


def plot_bubble_plot_mean_distances(distances_df, primary_vars, comparison_vars,normalise_by_row=False,fig_size=(5, 5),save_path=None):
    """
    Plot a bubble plot of mean distances between primary and comparison variables.
    
    Args:
        distances_df (pd.DataFrame): DataFrame containing the distance information from calculateDistances().
        primary_vars (list): List of primary variables to include in the plot. These are variables we calculate distances from.
        comparison_vars (list): List of comparison variables to include in the plot. These are variables we calculate distances to.
        save_path (str, optional): If provided, specifies the file path where the plot will be saved. If None, the plot will not be saved. Defaults to None.
    """
    spl.plot_bubble_plot_mean_distances(distances_df, primary_vars, comparison_vars,normalise_by_row,fig_size,save_path)



def plot_custom_scatter(data: pd.DataFrame, primary_vars: List[str], comparison_vars: List[str], fig_size: tuple = (10, 5),
                        bubble_size: tuple=(700, 700), file_save: bool = False,sort_by_difference: bool =True, 
                        compare_distribution_metric: Optional[str] = None, statistical_test: bool =False) -> None:
    """
    Plots a custom scatter plot comparing distances between two primary variables.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame containing the dataset to be plotted. This data should include columns for primary and comparison variables along with a 'min_distance' metric from calculateDistances().
        primary_vars (List[str]): A list of primary variable names. These variables are the main focus of the comparison and should include only two.
        Distances are calculated relative to first item in primary_vars e.g. negative means closer to primary_var[0] and positive means closer to primary_var[1]
        comparison_vars (List[str]): A list of comparison variable names. These variables are used to compare against the primary variables.
        fig_size (tuple, optional): The size of the figure for the scatter plot. Defaults to (10, 5).
        bubble_size (tuple, optional): The size of the bubbles in the scatter plot. Defaults to (700, 700).
        file_folder (Optional[str], optional): If provided, specifies the file path where the plot will be saved. If None, the plot will not be saved. Defaults to None. 
        compare_distribution_metric (Optional[str], optional): The metric to use for comparing the distribution of distances. Defaults to None. Options:  None, 'min', 'mean', 'median', 'ks_test','median_across_all_batches' is simplest approach (looks at all hotspots across batches, without weighting by batch ie. can be biased towards batches with more hotspots.)
        Please see paper for further description of these options. e.g. min means out of all distances from primary variable hotspot of interest, compare the minimum distance to hotspot of comparison variable.
        statistical_test (bool, optional): Whether to return df for further analysis.
    
    This function filters the data based on the specified primary and comparison variables, calculates mean distances, and plots these distances in a scatter plot. The plot illustrates the differences in distances between two primary variables across various comparison variables, with bubble size and color indicating statistical significance.
    """
    spl.plot_custom_scatter(data, primary_vars, comparison_vars, fig_size, bubble_size, file_save,sort_by_difference, compare_distribution_metric, statistical_test)


#for each comparison variable plot box plots of the distribution of min_distance for each primary variable
def plot_bar_plot_distance(distances,primary_variables,comparison_variables,fig_size):
    spl.plot_bar_plot_distance(distances,primary_variables,comparison_variables,fig_size)

######################################################################################## hotspot heatmaps #######################################################################################################################################

    
def score_genes_by_batch(adata, gene_lists, gene_list_names):
    """
    Scores genes in batches and adds the scores to the AnnData object in .obs.
    Returns the AnnData object and a list of the signature names in anndata object.

    Parameters:
    adata (AnnData): The AnnData object containing the data.
    gene_lists (list of list of str): A list of gene lists, each list contains the genes for a signature.
    gene_list_names (list of str): A list of names for each gene list.
    batch_column (str): The name of the column in adata.obs that contains the batch information.
    """
    adata, signatures=hotspot_helper.score_genes_by_batch(adata, gene_lists, gene_list_names)
    return adata, signatures


def plot_gene_heatmap(adata, signatures, states_to_loop_through, plot_score=False, normalize_values=False,fig_size=(5, 5),
                      score_by_batch=False,save_path=None):
    """
    Plots a heatmap of gene signatures across different states.
    
    Parameters:
        adata (AnnData): The AnnData object containing data and signtaure scores in .oba.
        signatures (list of str): List of gene signature names that have been scored in the AnnData object.
        states_to_loop_through (list of str): List of .obs labels names to be analyzed.
        plot_score (bool, optional): Whether to annotate the heatmap with the scores. Defaults to False.
        normalize_values (bool, optional): Whether to normalize the scores before plotting the heatmap. Defaults to False.
        score_by_batch (bool, optional): Whether to score the signatures by batch. Defaults to False.
    """
    data_for_heatmap=spl.plot_gene_heatmap(adata, signatures, states_to_loop_through, plot_score, normalize_values,fig_size,
                      score_by_batch)
    return data_for_heatmap

#bubble chart version of heatmap
def compare_gene_signatures(anndata_breast, gene_signatures, states, fig_size=(5, 2), bubble_size=500):
    """
    Plots a bubble chart comparing gene signatures between two states.

    :param anndata_breast: AnnData object containing gene expression data and signatures scores.
    :param gene_signatures: List of gene signatures to compare. These should be labelled in .obs column.
    :param states: List of two states to compare.
    :param fig_size: Tuple indicating the size of the figure.
    :param bubble_size: Size of the bubbles in the bubble plot.
    """
    data = spl.calculate_signature_differences(anndata_breast, gene_signatures, states)
    spl.plot_bubble_chart(data, states, fig_size, bubble_size)



def plot_signature_boxplot(anndata_breast,hotspot_variable,signature,fig_size=(3,3),file_save=None):
    """
    This function plots a boxplot for comparing responses to signatures based on two defined hotspots.

    Parameters:
    anndata_breast (AnnData): An AnnData object containing signatures scores in .obs column.
    hotspot_variable (list): A list of two strings representing the hotspot column names in anndata_breast.obs. 
    signature (str): The column name in anndata_breast.obs that contains the signature.
    fig_size (tuple): A tuple representing the figure size. Default is (3, 3).
    """
    spl.plot_signature_boxplot(anndata_breast,hotspot_variable,signature,fig_size,file_save)


############################################################# hotspot sensitivity #############################################################################################
def sensitivity_calcs(spatial_anndata, params):
    """
    This function calculates the sensitivity based on various parameters and returns several lists as output.

    Parameters:
    - spatial_anndata: AnnData object containing the spatial transcriptomics data.
    - params: A dictionary containing all the parameters required for sensitivity calculations.
    note: params['variable_comparison'] is the variable that is being compared to the other variables
        - params (dict): A dictionary with the following keys:
        * 'variable_comparison' (str): Name of the variable used for comparison.
        * 'variable_comparison_tumour' (bool): Indicates if data should be filtered for tumor cells for 'variable_comparison'.
        * 'sensitivity_parameter' (str): Specifies the sensitivity analysis parameter type ('pvalue' or 'neighbourhood').
        * 'variable_one' (str), 'variable_two' (str): Names of primary variables for distance calculation.
        * 'variable_one_two_is_tumour' (bool): Indicates if data should be filtered for tumor cells for 'variable_one' and 'variable_two'.
        * 'variable_three' (str): Reference variable for distance calculation. E.g all tumour cells. Do not calculate hotspot for this.
        * 'values_to_test' (list): List of values to test for the sensitivity analysis.
        * 'save_path' (str): Name of the file to save the plot.
        * 'variable_comparison_constant' (bool): Indicates if the comparison variable should be varied.
    Returns:
    
    - results (dict): A dictionary containing the calculated distances.

    """
    sensitivity_analysis.sensitivity_calcs(spatial_anndata, params)

def process_tumor_perimeter(adata):
    """
    Process the tumor perimeter cells in the dataset.

    Parameters:
    - adata: AnnData object containing single-cell spatial data with the following annotations:
        - adata.obs['batch']: A categorical column indicating the batch each cell belongs to.
        - adata.obs['tumour_cells']: A binary column indicating whether a cell is a tumour cell (1) or not (0).

    Returns:
    - adata: AnnData object with an updated column in `data.obs` indicating tumour perimeter cells.
        - adata.obs['tumour_perimeter']: A column indicating whether a cell is a tumour perimeter cell ('Yes') or not (NaN).
    """
    adata=tumour_perimeter.process_tumor_perimeter(adata)
    return adata
    

################################Plot differences by batch ################################


def plot_bubble_chart_by_batch(df, primary_variable_value, comparison_variable_values, reference_variable='tumour_cells', save_path=None, pval_cutoff=0.05, fig_size=(12,10),bubble_size=20,slide_order=None):
    """
    Plot a bubble chart showing the relationship between EMT variables and TME variables.

    Parameters:
    df (DataFrame): The distance df containing the distances to be plotted. Calculated from calculateDistances(). Primary variables should be in the 'primary_variable' column, 
    comparison variables should be in the 'comparison_variable' column, and distances should be in the 'min_distance' column.
    primary_variable_value (str): The primary variable of interest we want to comparise distances FROM (e.g., 'EMT_hallmarks_hot').
    comparison_variable_values (list): List of TME variables to include in the plot.
    reference_variable (str): The reference variable for comparison (default is 'tumour_cells'). Allows us to statistically compare distance distributions.
    save_path (str): Path to save the plot image. If None, the plot is not saved.
    pval_cutoff (float): P-value cutoff for significance (default is 0.05).
    fig_size (tuple): Size of the figure (default is (12,10)).

    Returns:
    DataFrame: The grouped and processed DataFrame used for plotting.
    """
    grouped_data=spl.plot_bubble_chart_by_batch(df, primary_variable_value, comparison_variable_values, reference_variable, save_path, pval_cutoff, fig_size,bubble_size,slide_order)

    return grouped_data

def plot_results_gene_signatures_heatmap(adata_vis, variable_one, comparison_variable, genes, file_path_plots,gene_signature_name,fig_size=(6, 7.2)):
    spl.plot_results_gene_signatures_heatmap(adata_vis, variable_one, comparison_variable, genes, file_path_plots,gene_signature_name,fig_size)

def plot_condition_differences(adata, variable_of_interest, conditions, save_path=None):
    spl.plot_condition_differences(adata, variable_of_interest, conditions, save_path)

#################################neighbourhood analysis plotting ##########################################



def calculate_neighbourhood_correlation(rings_range,adata_vis,neighbour_variables,source_nodes,
                                        neighbourhood_variable_filter_for_tumour_cells=None,
                                        split_by_batch=False):
    """

    Calculate the correlation between source nodes and neighbouring nodes for varying neighborhood sizes.

    Parameters:
    - rings_range: Iterable of integers specifying the range of neighborhood sizes (number of rings) to consider.
    - adata_vis: AnnData object containing spatial data with the following:
        - adata_vis.obs['batch']: A categorical column indicating the batch each cell belongs to.
    - neighbour_variables: DataFrame containing variables for the TME cells to be considered in the correlation.
    - source_nodes: List of source nodes for which the neighborhood correlation is calculated wrt. e.g. EMT
    - neighbourhood_variable_filter_for_tumour_cells: (Optional) Filter to apply on neighboring variables for tumour cells.
    - split_by_batch: Boolean indicating whether to calculate correlations separately for each batch.
    If split_by_batch is True, calculate the correlation between source nodes and neighbouring nodes for each batch. 
    As we take the median of the correlation values, the pvalues are not returned.

    Returns:
    - results: Dictionary with neighborhood size as keys and correlation matrices (and p-values if not split by batch) as values.
    """
    results=neighbourhood_allinone_helper.calculate_neighbourhood_correlation(rings_range,
                                                                adata_vis,
                                                                neighbour_variables,
                                                                source_nodes,
                                                                neighbourhood_variable_filter_for_tumour_cells,
                                                                split_by_batch)

    return results
    



def correlation_heatmap_neighbourhood(results,variables=None, save_path=None, pval_cutoff=0.05,fig_size=(5, 5)):
    """
    Plot heatmap from correlation dataframes.

    Parameters:
    - results (tuple): A tuple where the first element is a DataFrame of correlation coefficients 
      and the second is a DataFrame of p-values. This is returned from hotspot.calculate_neighbourhood_correlation. 
      Select ring number to plot by filtering the dict returned from hotspot.calculate_neighbourhood_correlation.
    - variables (list of str): List of variable names to include in the heatmap.
    - save_path (str, optional): Path to save the heatmap image. If None, the heatmap is not saved.
    - pval_cutoff (float): The p-value cutoff for significance.

    The function plots a heatmap and optionally saves it to a file.
    """
    neighbourhood_plotting.correlation_heatmap_neighbourhood(results, variables, save_path, pval_cutoff,fig_size)

    
def plot_correlation_shifts(ring_sensitivity_results1, correlation_primary_variable, save_path, ring_sensitivity_results2=None,
                            correlation_primary_variable2=None, fig_size=(10, 3), split_by_batch=False,
                            label_one=None,label_two=None,y_limits=None):
    """
    Plot the shifts in correlation over different 'ring' sizes for up to two different sets of results.

    Parameters:
    - ring_sensitivity_results1 (dict): First set of results, where each key is the number of rings used, and each value is a tuple containing the correlation matrix and p-values matrix.
    - correlation_primary_variable (str): The primary variable for which correlations are to be plotted.
    - save_path (str): Path to save the plot.
    - ring_sensitivity_results2 (dict, optional): Second set of results, similar structure as the first.
    - fig_size (tuple, optional): The size of the figure (width, height).
    - split_by_batch (bool, optional): Whether the correlation shifts are split by batch.

    The function plots and saves a multi-subplot figure showing correlation shifts for one or two sets of data.
    """
    neighbourhood_plotting.plot_correlation_shifts(ring_sensitivity_results1, correlation_primary_variable, save_path, ring_sensitivity_results2,
                            correlation_primary_variable2, fig_size, split_by_batch,
                            label_one,label_two,y_limits)

def plot_overall_change(ring_sensitivity_results,correlation_primary_variable,save_path,split_by_batch=False,fig_size=(10, 7)):
    """
    Plot the overall change in correlation values across different 'ring' sizes.

    Parameters:
    - ring_sensitivity_results (dict): A dictionary where keys are the number of rings and values are DataFrames 
      containing correlation data for those rings.
    - correlation_primary_variable (str): The primary variable for which the overall change in correlation is to be plotted.
    - save_path (str): Path to save the plot.
    - fig_size (tuple, optional): The size (width, height) of the figure.

    The function creates a bar plot showing the difference in correlation values for the primary variable across 
    the specified range of rings. Positive and negative changes are indicated with different color gradients.
    """
    neighbourhood_plotting.plot_overall_change(ring_sensitivity_results,correlation_primary_variable,save_path,split_by_batch,fig_size)
    


#returns cell abundance for inner and outer rings
def calculate_inner_outer_neighbourhood_enrichment(rings_range, adata_vis, neighbour_variables, source_nodes):

    """
    Calculates cell abundance for inner and outer rings and returns results for different ring ranges.

    Parameters:
    - rings_range (list): List of integer values representing different ring ranges to analyze.
    - adata_vis (AnnData): AnnData object.
    - neighbour_variables (DataFrame): DataFrame from .obs anndata obj. containing variables for correlation.
    - source_nodes (list): List of source nodes for analysis e.g. tumour_cells. We will loop through this for neighbourhood analysis.

    Returns:
    - Dict: Dictionary containing cell abundance mean results for inner and outer rings for each ring range.
    """
    inner_outer_results=neighbourhood_inner_outer_helper.calculate_inner_outer_neighbourhood_enrichment(rings_range, adata_vis, neighbour_variables, source_nodes)
    return inner_outer_results

#using cell abundance from calculate_inner_outer_neighbourhood_enrichment, we calculate corr_df and pval that we can use to plot using functions previoulsy defined
#correlation_key_variable is the variable that we are calculating the correlation for (e.g. EMT_hallmarks_hot) using the inner value for it and correlating how this value affects outer ring
#we keep this function separate from calculate_inner_outer_neighbourhood_enrichment to allow for flexibility in calculating correlation for different variables

def calculate_corr_pvalue_for_inner_outer_neighbourhood_enrichment(results,correlation_key_variable,rings_range,average_by_batch=False,adata_vis=None,source_nodes=None):
    """
    Calculates correlation and p-value for inner and outer neighborhood enrichment.

    Parameters:
    - results (dict): Dictionary containing inner and outer DataFrame cell abundance results from previous analysis (calculate_inner_outer_neighbourhood_enrichment())
    - correlation_key_variable (str): The key variable for which correlation is calculated in inner ring. We are comparing how a value of this variable in inner ring affects outer ring.
    - rings_range (list): List of ring ranges for which correlation and p-value are calculated.
    - average_by_batch (bool): If True, correlation is calculated for each batch and then averaged.
    - source_nodes (list): List of source nodes for analysis e.g. tumour_cells. We will loop through this for neighbourhood analysis. Must be same as in calculate_inner_outer_neighbourhood_enrichment().
    - adata_vis (AnnData): AnnData object. Only used if average_by_batch is True.

    Returns:
    - Dict: Dictionary containing correlation DataFrame and p-value for each ring. We can then run previous plots e.g. plot_correlation_shifts() on this. Equally accessing each element 
    of dict returns correlations and pvalue for that ring range.
    """
    corr_pval_results=neighbourhood_inner_outer_helper.calculate_corr_pvalue_for_inner_outer_neighbourhood_enrichment(results,correlation_key_variable,rings_range,average_by_batch,adata_vis,source_nodes)
    return corr_pval_results



def plot_correlation_coefficients_bar_chart(correlation_dict,ring_value,variable_to_compare,save_path=None,fig_size=(10,8)):
    """
    Plots a bar chart of correlation coefficients for a specified variable compared across different metrics.

    Parameters:
    - correlation_dict (dict): Dictionary containing correlation data for various ring sizes.
    - ring_value (str): The key within 'results' dict that refers to a specific set of correlation data.
    - variable_to_compare (str): The specific variable within the ring data to compare across metrics.
    """
    neighbourhood_plotting.plot_correlation_coefficients_bar_chart(correlation_dict,ring_value,variable_to_compare,save_path,fig_size)
    
def plot_correlation_coefficients_heatmap(correlation_dict, correlation_variable, save_path=None,fig_size=(15, 6)):
    """
    Plots a bar heatmap correlation coefficients for different ring sizes

    Parameters:
    - correlation_dict (dict): Dictionary containing correlation data for various ring sizes.
    - save_path (str): Path to save the heatmap image. If None, the heatmap is not saved.
    - correlation_variable (str): The specific variable  to compare across.
    """
    neighbourhood_plotting.plot_correlation_coefficients_heatmap(correlation_dict, correlation_variable, save_path,fig_size)

def add_genes_to_obs(adata, gene_list):
    """
    Check if genes are in anndata.var_names, and if so, add their expression to anndata.obs.
    
    Parameters:
    - adata: An anndata object.
    - gene_list: A list of gene names to check and add.
    
    Returns:
    - Updates the anndata object in place, no return value.
    """
    hotspot_helper.add_genes_to_obs(adata, gene_list)


############################### individual hotspot analysis ##########################################
def plot_distance_distributions_across_batches(distance_df, comparison_variable,fig_size=(3, 5),fig_name=None):
    """
    Plots distribution of minimum distances across different batches, comparing two primary variables.
    Parameters:
    - distance_df (DataFrame): The distance DataFrame containing data with 'comparison_variable', 'primary_variable', 
      'batch', and 'min_distance' columns.
    - comparison_variable (str): Variable based on which DataFrame is filtered to compare the 
      distributions of minimum distances.
    - fig_size (tuple): Figure size to set for the plot, e.g., (width, height).

    """
    access_individual_hotspots.plot_distance_distributions_across_batches(distance_df, comparison_variable,fig_size,fig_name)

def plot_distance_distributions_across_hotspots(df, comparison_variable,batch,fig_size=(3, 5),fig_name=None):
    """
        Plots the distribution of minimum distances across different hotspots within a specified batch,
        allowing comparison across unique primary variables.
        Parameters:
        - df (DataFrame): DataFrame containing the data with 'comparison_variable', 'primary_variable',
        'hotspot_number', and 'min_distance' columns.
        - comparison_variable (str): The variable based on which the DataFrame is filtered to compare the
        distributions of minimum distances.
        - batch (int or str): The batch number to filter the DataFrame on, isolating data for specific analysis.
        - fig_size (tuple): The size of the figure for the plot, specified as (width, height).
    """    
    access_individual_hotspots.plot_distance_distributions_across_hotspots(df, comparison_variable,batch,fig_size,fig_name)


def plot_hotspots_by_number(
    anndata: AnnData,
    column_name: str,
    batch_single: Optional[str] = None,
    save_path: Optional[str] = None,
    color_for_spots: str = 'Reds_r'
) -> None:
    access_individual_hotspots.plot_hotspots_by_number(anndata, column_name, batch_single, save_path, color_for_spots)