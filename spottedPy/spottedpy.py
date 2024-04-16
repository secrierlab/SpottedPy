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

# Typing
from typing import Tuple, List, Optional, Union
# Statsmodels for statistics
from statsmodels.sandbox.stats.multicomp import multipletests
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='libpysal')

#pdf font
plt.rcParams['pdf.fonttype'] = 'truetype'    

#spotterpy functions
import sp_plotting as spl

######################################################################################## key hotspot functionality ################################################################################################################################################################################

def find_connected_components(
    hotspot: pd.DataFrame,
    slides: AnnData
) -> Tuple[pd.DataFrame, int]:
    """
    Find connected components in a spatial graph.

    Parameters:
    hotspot (pd.DataFrame): DataFrame containing hotspot data.
    slides ('AnnData'): AnnData object containing spatial data.

    Returns:
    Tuple[pd.DataFrame, int]: Updated hotspot DataFrame and number of connected components.
    """
    anndata_scores_filtered_high_hotspot = slides[slides.obs.index.isin(hotspot.index)]
    row_count = anndata_scores_filtered_high_hotspot.obs.index.shape[0]

    if row_count < 2:
        neighbour_no = 1
    elif row_count < 21:
        neighbour_no = row_count - 1
    else:
        neighbour_no = 20

    sq.gr.spatial_neighbors(anndata_scores_filtered_high_hotspot, n_rings=1, coord_type="grid", n_neighs=neighbour_no)
    connectivity_matrix = pd.DataFrame.sparse.from_spmatrix(anndata_scores_filtered_high_hotspot.obsp['spatial_connectivities'])
    connectivity_matrix.index = anndata_scores_filtered_high_hotspot.obs.index
    connectivity_matrix.columns = anndata_scores_filtered_high_hotspot.obs.index
    connectivity_matrix_sparse = scipy.sparse.csr_matrix(connectivity_matrix.values)
    n_components, labels = connected_components(csgraph=connectivity_matrix_sparse, directed=False, return_labels=True)
    hotspot = hotspot.copy()
    hotspot['hotspot_label'] = labels
    hotspot = hotspot[~hotspot['hotspot_label'].isin(hotspot['hotspot_label'].value_counts()[hotspot['hotspot_label'].value_counts() < 5].index)]

    return hotspot, n_components


def calculate_hotspots_with_hotspots_numbered(
    anndata_filtered: AnnData,
    significance_level: float = 0.05,
    score_column: str = 'scores',
    neighbours_param: int = 5,
    return_number_components: bool = False,
    hotspots_relative_to_batch: bool = True,
    add_hotspot_numbers: bool = False
) -> Union['AnnData', Tuple[List[int], List[int], 'AnnData']]:
    """
    Calculate hotspots with numbered hotspots.

    Parameters:
    anndata_filtered (AnnData): Filtered AnnData object.
    significance_level (float): Significance level for hotspots.
    score_column (str): Column name for scores.
    neighbours_param (int): Number of neighbours to consider.
    return_number_components (bool): Whether to return number of components.

    Returns:
    Union[AnnData, Tuple[List[int], List[int], AnnData]]: Updated AnnData object, and optionally lists of number of components.
    """
    n_components_high_list = []
    n_components_low_list = []
    anndata_filtered=anndata_filtered.copy()
    anndata_filtered.obs[score_column + "_hot"] = np.nan
    anndata_filtered.obs[score_column + "_cold"] = np.nan

    #compute hotspots relative to batch
    if hotspots_relative_to_batch:
        for batch in anndata_filtered.obs['batch'].unique():
            score_df = anndata_filtered[anndata_filtered.obs['batch'] == str(batch)].obs
            score_df = score_df[~pd.isna(score_df[score_column])]
            pp = list(zip(score_df['array_row'], score_df['array_col']))
            kd = libpysal.cg.KDTree(np.array(pp))
            wnn2 = libpysal.weights.KNN(kd, neighbours_param)
            y = score_df[score_column]
            lg = G_Local(y, wnn2, seed=100)
            high_hotspot = score_df.loc[(lg.Zs > 0) & (lg.p_sim < significance_level)]
            low_hotspot = score_df.loc[(lg.Zs < 0) & (lg.p_sim < significance_level)]
            if high_hotspot.shape[0] > 1:
                high_hotspot, n_components_high = find_connected_components(high_hotspot, anndata_filtered)
            else:
                n_components_high = 0
            if low_hotspot.shape[0] > 1:
                low_hotspot, n_components_low = find_connected_components(low_hotspot, anndata_filtered)
            else:
                n_components_low = 0
            anndata_filtered.obs.loc[high_hotspot.index, score_column + "_hot"] = high_hotspot[score_column]
            anndata_filtered.obs.loc[low_hotspot.index, score_column + "_cold"] = low_hotspot[score_column]

            #add labels here
            if add_hotspot_numbers:
                anndata_filtered.obs.loc[high_hotspot.index, score_column + "_hot_number"] = high_hotspot['hotspot_label']
                anndata_filtered.obs.loc[low_hotspot.index, score_column + "_cold_number"] = low_hotspot['hotspot_label']
                
            
            if return_number_components:
                n_components_low_list.append(n_components_low)
                n_components_high_list.append(n_components_high)
    
    #compute hotspots relative to all data: unconventional approach, but useful for treatment effect analysis etc.
    else:
        #ensure that each batch have unique coordinates so they do not overlap
        unique_batches = sorted(anndata_filtered.obs['batch'].unique())
        # Create a mapping from batch to an incrementing number
        batch_to_prefix = {batch: i * 2 for i, batch in enumerate(unique_batches)}

        # Update 'array_row' by adding the prefix based on the batch
        anndata_filtered.obs['array_row'] = anndata_filtered.obs.apply(lambda row: str(batch_to_prefix[row['batch']]) + str(row['array_row']), axis=1)
        anndata_filtered.obs['array_col'] = anndata_filtered.obs.apply(lambda row: str(batch_to_prefix[row['batch']]) + str(row['array_col']), axis=1)
        #convert back to int
        anndata_filtered.obs['array_row'] = anndata_filtered.obs['array_row'].astype(int)
        anndata_filtered.obs['array_col'] = anndata_filtered.obs['array_col'].astype(int)


        score_df = anndata_filtered.obs
        score_df = score_df[~pd.isna(score_df[score_column])]
        pp = list(zip(score_df['array_row'], score_df['array_col']))
        kd = libpysal.cg.KDTree(np.array(pp))
        wnn2 = libpysal.weights.KNN(kd, neighbours_param)
        y = score_df[score_column]
        lg = G_Local(y, wnn2, seed=100)
        high_hotspot = score_df.loc[(lg.Zs > 0) & (lg.p_sim < significance_level)]
        low_hotspot = score_df.loc[(lg.Zs < 0) & (lg.p_sim < significance_level)]
        if high_hotspot.shape[0] > 1:
            high_hotspot, n_components_high = find_connected_components(high_hotspot, anndata_filtered)
        else:
            n_components_high = 0
        if low_hotspot.shape[0] > 1:
            low_hotspot, n_components_low = find_connected_components(low_hotspot, anndata_filtered)
        else:
            n_components_low = 0
        anndata_filtered.obs.loc[high_hotspot.index, score_column + "_hot"] = high_hotspot[score_column]
        anndata_filtered.obs.loc[low_hotspot.index, score_column + "_cold"] = low_hotspot[score_column]
        if return_number_components:
            n_components_low_list.append(n_components_low)
            n_components_high_list.append(n_components_high)
        
    if return_number_components:
        return n_components_low_list, n_components_high_list, anndata_filtered
    else:
        return anndata_filtered


def create_hotspots(
    anndata: AnnData,
    column_name: str = "",
    filter_columns: Optional[str] = None,
    filter_value: Optional[str] = None,
    neighbours_parameters: int = 10,
    p_value: float = 0.05,
    number_components_return: bool = False,
    relative_to_batch: bool = True,
    number_hotspots: bool = False
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
    anndata=anndata.copy()
    if filter_columns is not None:
        anndata_filtered = anndata[anndata.obs[filter_columns] == filter_value]
    else:
        anndata_filtered = anndata

    if any(anndata_filtered.obs[column_name] < 0):
        raise ValueError("Score values must be in the range 0 to 1. Ensure there are no negative values in the score column.")



    if number_components_return:
        n_components_low, n_components_high, anndata_filtered = calculate_hotspots_with_hotspots_numbered(
            anndata_filtered,
            significance_level=p_value,
            score_column=column_name,
            neighbours_param=neighbours_parameters,
            return_number_components=True,
            hotspots_relative_to_batch=relative_to_batch,
            add_hotspot_numbers=number_hotspots
        )

        anndata.obs[column_name + "_hot"] = anndata_filtered.obs[column_name + "_hot"]
        anndata.obs[column_name + "_cold"] = anndata_filtered.obs[column_name + "_cold"]

        if number_hotspots:
            anndata.obs[column_name + "_hot_number"] = anndata_filtered.obs[column_name + "_hot_number"]
            anndata.obs[column_name + "_cold_number"] = anndata_filtered.obs[column_name + "_cold_number"]
        
        return n_components_low, n_components_high, anndata
    
    else:
        anndata_filtered = calculate_hotspots_with_hotspots_numbered(
            anndata_filtered,
            significance_level=p_value,
            score_column=column_name,
            neighbours_param=neighbours_parameters,
            hotspots_relative_to_batch=relative_to_batch,
            add_hotspot_numbers=number_hotspots
        )
    
    anndata.obs[column_name + "_hot"] = anndata_filtered.obs[column_name + "_hot"]
    anndata.obs[column_name + "_cold"] = anndata_filtered.obs[column_name + "_cold"]

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
    if batch_single is not None:
        data_subset = anndata[anndata.obs['batch'] == str(batch_single)]
        sc.pl.spatial(data_subset, color=[column_name],  vmax='p0',color_map=color_for_spots, library_id=str(batch_single),save=f"_{save_path}",colorbar_loc=None,alpha_img= 0.5)
    else:
        for batch in anndata.obs['batch'].unique():
            data_subset = anndata[anndata.obs['batch'] == str(batch)]
            sc.pl.spatial(data_subset, color=[column_name],  vmax='p0',color_map=color_for_spots, library_id=str(batch),save=f"_{str(batch)}_{save_path}",colorbar_loc=None,alpha_img= 0.5)

def calculateDistancesHelper(batch_adata,primary_variables, comparison_variables,batch,empty_hotspot_default_to_max_distance):
    distances_per_batch = pd.DataFrame()
# Loop through each variable in the first set
    for primary_var in primary_variables:
        primary_points = batch_adata[batch_adata.obs[primary_var].notnull()].obs[['array_row', 'array_col']]
        slide_distances = distance_matrix(primary_points, batch_adata.obs[['array_row', 'array_col']])
        if empty_hotspot_default_to_max_distance:
            if slide_distances.size > 0:
                max_distances = np.amax(slide_distances, axis=1)
                #get a scalar value for max value within min_distances
                max_distance_of_slide=np.amax(max_distances)
        # Loop through each variable in the second set
        for comparison_var in comparison_variables:
            # Extract the points for the primary and comparison variables
            comparison_points = batch_adata[batch_adata.obs[comparison_var].notnull()].obs[['array_row', 'array_col']]
            # Calculate the distance matrix between the primary and comparison points
            dist_matrix = distance_matrix(primary_points, comparison_points)

            # Check if the distance matrix is empty
            if dist_matrix.size > 0:
                # Calculate the minimum distance for each primary point to the comparison points
                min_distances = np.amin(dist_matrix, axis=1)

                
                # Create a DataFrame to store the results
                temp_df = pd.DataFrame({
                    'min_distance': min_distances,
                    'primary_variable': primary_var,
                    'comparison_variable': comparison_var,
                    'primary_index': primary_points.index,
                    'batch': batch
                })
                # Concatenate the results DataFrame with the overall DataFrame
                distances_per_batch = pd.concat([distances_per_batch, temp_df])
            else:
                # Print a warning message if the distance matrix is empty
                print(f"Warning: Empty distance matrix for batch = {batch}, primary_var = {primary_var}, comparison_var = {comparison_var}")
                #if slide_distances.size > 0 then there are primary points, so add max_distance_of_slide to distances_per_batch
                if empty_hotspot_default_to_max_distance:
                    if slide_distances.size > 0:
                        #if primary variables exist, and no comparison value for that variable, make distance max length of slide                 
                        temp_df = pd.DataFrame({'min_distance': [max_distance_of_slide] * len(primary_points),
                                                'primary_variable': [primary_var] * len(primary_points),
                                                'comparison_variable': [comparison_var] * len(primary_points),
                                                'batch': [batch] * len(primary_points)})
                        # Concatenate the results DataFrame with the overall DataFrame                
                        distances_per_batch = pd.concat([distances_per_batch, temp_df]) 
    return distances_per_batch

def calculateDistances(anndata, primary_variables, comparison_variables=None,split_by_slide_in_batch=False,empty_hotspot_default_to_max_distance=False):
    """
    Calculate the minimum distances between points specified by two sets of variables in a multi-slide dataset. Variables should be populated with np.nan for points in anndata not included in variables.
    
    Args:
        anndata (AnnData): Annotated data matrix.
        primary_variables (list): These are variables we calculate distances from.
        comparison_variables (list, optional): These are variables we calculate distances to. If not specified, the primary variables will be used.
        split_by_slide_in_batch (bool, optional): Whether to split the data by slide in each batch (if multiple slides/batch) and calculate distances within these slides. Defaults to False. It we set this to true, ensure empty_hotspot_default_to_max_distance to False if there are small slides in each spot as this could bias the data.
        empty_hotspot_default_to_max_distance=if a slide does not contain any hotspots of comparison variable, default to the maximum distance
    
        Notes:
        empty distance matrix errors appear more in split_by_slide_in_batch=True as there are some very small slides; here we default to max distance and therefore this approach is also an approximate.
    Returns:
        pd.DataFrame: A DataFrame containing the minimum distances, the corresponding variables, and the batch information.
    """

    if comparison_variables is None:
        comparison_variables = primary_variables
    
    distances_df_all = pd.DataFrame()
    
    # Loop through each unique batch
    for batch in anndata.obs['batch'].unique():

        batch_data = anndata[anndata.obs['batch'] == batch]

        if split_by_slide_in_batch:
            #get connected components for batch
            sq.gr.spatial_neighbors(batch_data, n_rings=1, coord_type="grid", n_neighs=20)
            connectivity_matrix = pd.DataFrame.sparse.from_spmatrix(batch_data.obsp['spatial_connectivities'])
            connectivity_matrix.index = batch_data.obs.index
            connectivity_matrix.columns = batch_data.obs.index
            connectivity_matrix_sparse = scipy.sparse.csr_matrix(connectivity_matrix.values)
            n_components, labels = connected_components(csgraph=connectivity_matrix_sparse, directed=False, return_labels=True)
            #hotspot = hotspot[~hotspot['hotspot_label'].isin(hotspot['hotspot_label'].value_counts()[hotspot['hotspot_label'].value_counts() < 5].index)]
            batch_data.obs['connected_labels'] = labels
            #loop through slides within a batch (as sometimes they are saved within one image)
            for label in batch_data.obs['connected_labels'].unique():
                #filter batch_data to only include label
                batch_data_label=batch_data[batch_data.obs['connected_labels']==label]
                if batch_data_label.shape[0]<5:
                    continue
                distance_df=calculateDistancesHelper(batch_data_label, primary_variables, comparison_variables,batch,empty_hotspot_default_to_max_distance)
                distances_df_all = pd.concat([distances_df_all, distance_df])
        else:
            distance_df=calculateDistancesHelper(batch_data, primary_variables, comparison_variables,batch,empty_hotspot_default_to_max_distance)
            distances_df_all = pd.concat([distances_df_all, distance_df])

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
    # Filter the DataFrame based on the specified primary and comparison variables
    filtered_df = distances_df[
        distances_df['primary_variable'].isin(primary_vars) &
        distances_df['comparison_variable'].isin(comparison_vars)
    ]
    
    # Group by primary and comparison variables and calculate the mean distance
    mean_df = (
        filtered_df
        .groupby(['primary_variable', 'comparison_variable'])
        .min_distance
        .mean()
        .reset_index()
    )

    # Set up the figure and axes
    plt.figure(figsize=fig_size)
    with plt.rc_context():
        scatter = sns.scatterplot(
            x='primary_variable', 
            y='comparison_variable', 
            size='min_distance',
            sizes=(100, 2000),
            data=mean_df,
            hue='min_distance',
            palette="viridis",
            legend=False
        )
        
        # Set plot limits
        plt.xlim(-0.5, len(primary_vars) - 0.5)
        plt.ylim(-0.5, len(comparison_vars) - 0.5)

        # Set plot title and labels
        plt.title("Mean Distances", fontsize=15)
        plt.xlabel("Primary Variable")
        plt.ylabel("Comparison Variable")
        
        # Set x-axis tick rotation and tick label size
        plt.xticks(rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=15)

        # Adjust plot layout and show plot
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
    

        plt.show()
        plt.close()



def plot_custom_scatter(data: pd.DataFrame, primary_vars: List[str], comparison_vars: List[str], fig_size: tuple = (10, 5),bubble_size: tuple=(700, 700), file_save: bool = False) -> None:
    """
    Plots a custom scatter plot comparing distances between two primary variables.

    Parameters:
        data (pd.DataFrame): A pandas DataFrame containing the dataset to be plotted. This data should include columns for primary and comparison variables along with a 'min_distance' metric.
        primary_vars (List[str]): A list of primary variable names. These variables are the main focus of the comparison and should include only two.
        comparison_vars (List[str]): A list of comparison variable names. These variables are used to compare against the primary variables.
        fig_size (tuple, optional): The size of the figure for the scatter plot. Defaults to (10, 5).
        bubble_size (tuple, optional): The size of the bubbles in the scatter plot. Defaults to (700, 700).
        file_folder (Optional[str], optional): If provided, specifies the file path where the plot will be saved. If None, the plot will not be saved. Defaults to None. 
    
    This function filters the data based on the specified primary and comparison variables, calculates mean distances, and plots these distances in a scatter plot. The plot illustrates the differences in distances between two primary variables across various comparison variables, with bubble size and color indicating statistical significance.
    """
    # Filter the data
    filtered_df = data[data['primary_variable'].isin(primary_vars)]
    filtered_df = filtered_df[filtered_df['comparison_variable'].isin(comparison_vars)]
    
    # Calculate mean distances
    mean_df = (filtered_df.groupby(['primary_variable', 'comparison_variable'])
               .min_distance.mean()
               .reset_index())

    # Set variables for comparison
    comparison_var_one = primary_vars[0]
    comparison_var_two = primary_vars[1]

    # Update x-axis labels
    filtered_df['comparison_variable'] = filtered_df['comparison_variable'].str.replace('q05cell_abundance_w_sf_', '')
    
    # Set plot style and font
    sns.set_style("white")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    # Calculate differences and p-values
    pivot_df = mean_df.pivot(index='comparison_variable', columns='primary_variable', values='min_distance')
    pivot_df['difference'] = pivot_df[comparison_var_one] - pivot_df[comparison_var_two]
    pivot_df = pivot_df.reset_index()
    pivot_df['p_value'] = pivot_df['comparison_variable'].apply(lambda x: spl.calculate_pvalue(x, comparison_var_one, comparison_var_two, filtered_df))
    pivot_df['color'] = pivot_df['p_value'].apply(spl.custom_color)
    
    # Sort the DataFrame
    pivot_df = pivot_df.reindex(pivot_df['difference'].abs().sort_values(ascending=False).index)

    # Plot the data
    # Plot the data
    plt.figure(figsize=fig_size)
    ax = sns.scatterplot(x='comparison_variable', y='difference', size=1,
                        sizes=bubble_size, data=pivot_df, hue='color', 
                        palette={'#D53E4F': '#D53E4F', '#FDAE61': '#FDAE61', '#FEE08B': '#FEE08B', '#E6F598': '#E6F598'}, 
                        legend=None)
    plt.axhline(0, color='gray', linestyle='--')
    # Add ylabel with arrows and text
    plt.ylabel('Closer to\n{} ←→ {}'.format(comparison_var_one, comparison_var_two))
    plt.xticks(rotation=90)
    plt.tight_layout()
    # Add custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='p < 0.001', markersize=10, markerfacecolor='#D53E4F'),
                       Line2D([0], [0], marker='o', color='w', label='p < 0.01', markersize=10, markerfacecolor='#FDAE61'),
                       Line2D([0], [0], marker='o', color='w', label='p < 0.05', markersize=10, markerfacecolor='#FEE08B'),
                       Line2D([0], [0], marker='o', color='w', label='p >= 0.05', markersize=10, markerfacecolor='#E6F598')]
    plt.legend(handles=legend_elements, loc='lower right')
    sns.despine()

    # Save the plot
    if file_save: 
        plt.savefig(f"{comparison_var_one}_minus_{comparison_var_two}_scatterplot_hallmarks.pdf", dpi=300)
    
    plt.show()


#for each comparison variable plot box plots of the distribution of min_distance for each primary variable
def plot_bar_plot_distance(distances,primary_variables,comparison_variables,fig_size):
    #filter distances by primary_variable and comparison_variable
    filtered_df = distances[
        distances['primary_variable'].isin(primary_variables) &
        distances['comparison_variable'].isin(comparison_variables)
    ]
    
    #plot boxplot of min_distance
    for comparison_variable in comparison_variables:
        fig, ax = plt.subplots(figsize=fig_size)
        sns.boxplot(data=filtered_df[filtered_df['comparison_variable'] == comparison_variable],
                    x='primary_variable', y='min_distance', ax=ax)
        ax.set_title(comparison_variable)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()
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
    signatures = [f'{name}_score' for name in gene_list_names]

    unique_batches = adata.obs['batch'].unique()
    for batch in unique_batches:
        batch_data = adata[adata.obs['batch'] == batch]
        for i, genes in enumerate(gene_lists):
            sc.tl.score_genes(batch_data, genes, ctrl_size=200, n_bins=25,
                              score_name=signatures[i], random_state=0, copy=False, use_raw=None)
            
            # Save the scores back to the original AnnData object
            adata.obs.loc[batch_data.obs.index, signatures[i]] = batch_data.obs[signatures[i]]

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
    
    # Calculate mean scores for each state and signature.
    if score_by_batch:
        # Calculate mean scores for each state and signature.
        mean_scores, data_dict = spl.calculate_mean_scores_per_batch(adata, signatures, states_to_loop_through)
        data_for_heatmap = spl.create_heatmap_data_per_batch(mean_scores, states_to_loop_through, signatures)

    else:
         # Calculate mean scores for each state and signature within each batch.
        mean_scores, data_dict = spl.calculate_mean_scores(adata, signatures, states_to_loop_through)
        # Create a DataFrame for heatmap with states as rows and signatures as columns.
        data_for_heatmap = spl.create_heatmap_data(mean_scores, states_to_loop_through, signatures)

        

    # Normalize the data for heatmap if required.
    if normalize_values:
        data_for_heatmap = spl.normalize_heatmap_data(data_for_heatmap)

    # Plot the heatmap using the normalized data.
    spl.plot_heatmap(data_for_heatmap, signatures, states_to_loop_through, fig_size,plot_score,save_path)

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
    plot_bubble_chart(data, states, fig_size, bubble_size)



def plot_signature_boxplot(anndata_breast,hotspot_variable,signature,fig_size=(3,3)):
    """
    This function plots a boxplot for comparing responses to signatures based on two defined hotspots.

    Parameters:
    anndata_breast (AnnData): An AnnData object containing signatures scores in .obs column.
    hotspot_variable (list): A list of two strings representing the hotspot column names in anndata_breast.obs. 
    signature (str): The column name in anndata_breast.obs that contains the signature.
    fig_size (tuple): A tuple representing the figure size. Default is (3, 3).
    """
    hot_data = anndata_breast.obs[~anndata_breast.obs[hotspot_variable[0]].isna()][signature]
    cold_data = anndata_breast.obs[~anndata_breast.obs[hotspot_variable[1]].isna()][signature]
    # Plotting
    plt.figure(figsize=fig_size)
    sns.boxplot(data=[hot_data, cold_data],showfliers=False)
    plt.xticks([0, 1], [hotspot_variable[0], hotspot_variable[1]])
    plt.title('Response to Checkpoint Genes based on EMT Hallmarks')
    plt.ylabel('Response to Checkpoint Score')
    plt.show()

############################################################# hotspot sensitivity #############################################################################################
#helper function
def process_hotspots(adata, variable_name, parameter_size, sensitivity_parameter):
    if sensitivity_parameter=="pvalue":
        adata_hotspots = create_hotspots(
            adata, column_name=variable_name,
            neighbours_parameters=10,
            p_value=parameter_size,
            number_components_return=False
        )
    
    if sensitivity_parameter=="neighbourhood":
        adata_hotspots = create_hotspots(
            adata, column_name=variable_name,
            neighbours_parameters=parameter_size,
            p_value=0.05,
            number_components_return=False
        )
    return adata_hotspots

#helper function
def calculate_distances(adata_hotspots, variables):
    return calculateDistances(adata_hotspots, variables)

#helper function
def process_batches(spatial_anndata, params):
    distances_df_sensitivity = pd.DataFrame()

    for batch in spatial_anndata.obs['batch'].unique():
        adata_batch = spatial_anndata[spatial_anndata.obs['batch'] == batch]
        tumour_filtered = adata_batch[adata_batch.obs['tumour_cells'] == 1]

        # Processing for variable_comparison
        adata_hotspots = process_hotspots(adata_batch if not params['variable_comparison_tumour'] else tumour_filtered, 
                                          params['variable_comparison'].rsplit('_', 1)[0], 
                                          params['parameter_comparison_variable_neighbourhood'],params['sensitivity_parameter'])
        spatial_anndata=spl.add_hotspots_to_fullanndata(spatial_anndata,params['variable_comparison'].rsplit('_', 1)[0],batch,adata_hotspots)

        # Processing for variable_one and variable_two
        for variable in [params['variable_one'], params['variable_two']]:
            adata_hotspots = process_hotspots(tumour_filtered if params['variable_one_two_is_tumour'] else adata_batch, 
                                              variable.rsplit('_', 1)[0], 
                                              params['parameter_variables_neighbourhood'],params['sensitivity_parameter'])
            spatial_anndata=spl.add_hotspots_to_fullanndata(spatial_anndata,variable.rsplit('_', 1)[0],batch,adata_hotspots)
        # Calculate distances
        #filter spatial_anndata to only include batch
        #spatial_anndata_batch=spatial_anndata[spatial_anndata.obs['batch']==batch]
        #sc.pl.spatial(spatial_anndata_batch,color=params['variable_one'],library_id=str(batch))
        #sc.pl.spatial(spatial_anndata_batch,color=params['variable_two'],library_id=str(batch))
        #sc.pl.spatial(spatial_anndata_batch,color=params['variable_three'],library_id=str(batch))
    distances_batch = calculate_distances(spatial_anndata, [params['variable_one'], params['variable_two'],params['variable_comparison'],params['variable_three']])
    distances_df_sensitivity = pd.concat([distances_df_sensitivity, distances_batch])


    return distances_df_sensitivity


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
    results = {
        "distance_variable_one": [],
        "distance_variable_two": [],
        "distance_variable_three": []
    }

    for neighbourhood_val in params['values_to_test']:
        if params['sensitivity_parameter']=="pvalue":
            params['parameter_comparison_variable_neighbourhood'] = neighbourhood_val if not params['variable_comparison_constant'] else 0.05
        
        if params['sensitivity_parameter']=="neighbourhood":
            params['parameter_comparison_variable_neighbourhood'] = neighbourhood_val if not params['variable_comparison_constant'] else 10
        params['parameter_variables_neighbourhood'] = neighbourhood_val

        distances_df_sensitivity = process_batches(spatial_anndata, params)
        
        distance_variable_one = distances_df_sensitivity[
            (distances_df_sensitivity['comparison_variable'] == params['variable_comparison']) & 
            (distances_df_sensitivity['primary_variable'] == params['variable_one'])
        ].groupby('batch')['min_distance'].mean()

        distance_variable_two = distances_df_sensitivity[
            (distances_df_sensitivity['comparison_variable'] == params['variable_comparison']) & 
            (distances_df_sensitivity['primary_variable'] == params['variable_two'])
            ].groupby('batch')['min_distance'].mean()

        distance_variable_three = distances_df_sensitivity[
            (distances_df_sensitivity['comparison_variable'] == params['variable_comparison']) & 
            (distances_df_sensitivity['primary_variable'] == params['variable_three'])
        ].groupby('batch')['min_distance'].mean()


        paired_df = pd.DataFrame({'variable_one': distance_variable_one, 'variable_two': distance_variable_two,'variable_three':distance_variable_three }).dropna()
        results["distance_variable_one"].append(paired_df['variable_one'].mean())
        results["distance_variable_two"].append(paired_df['variable_two'].mean())
        results["distance_variable_three"].append(paired_df['variable_three'].mean())

    spl.plot_sensitivity(params['values_to_test'], results, params['variable_comparison'],
                     params['variable_one'], params['variable_two'], params['variable_three'], params['save_path'],params['sensitivity_parameter'])

    return results




################################Plot differences by batch ################################


def plot_bubble_chart_by_batch(df, primary_variable_value, comparison_variable_values, reference_variable='tumour_cells', save_path=None, pval_cutoff=0.05, fig_size=(12,10),bubble_size=20):
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
    # Prepare the data for plotting
    grouped_data = spl.prepare_data_hotspot(df, primary_variable_value, comparison_variable_values,reference_variable)
    fig, ax = plt.subplots(figsize=fig_size)
    slides = df['batch'].unique()
    slide_positions = {slide: idx for idx, slide in enumerate(slides)}
    # Bonferroni correction
    n_tests = len(slides)
    bonferroni_alpha = pval_cutoff / n_tests
    # Iterate through the grouped data and plot each point
    for idx, row in grouped_data.iterrows():
        if row['primary_variable'] != reference_variable:
            # Determine position and color based on data values
            y_pos = slide_positions.get(row['batch'], 0)
            color = 'blue' if row['Difference'] > 0 else 'red'
            alpha = 1 if row['Pvalue'] < bonferroni_alpha else 0  # Transparency based on significance
            # Plot the bubble
            ax.scatter(row['comparison_variable'], y_pos, s=abs(row['Difference'])*bubble_size, color=color, alpha=alpha, edgecolors='white', linewidth=0.3)
    ax.set_yticks(list(slide_positions.values()))
    ax.set_yticklabels(list(slide_positions.keys()))
    labels=grouped_data['comparison_variable'].unique()
    ax.set_xticklabels(labels, rotation=90)

    #ax.set_xticklabels([label for label in comparison_variable_values], rotation=90)
    red_patch = mpatches.Patch(color='red', label='Closer to {}'.format(primary_variable_value))
    blue_patch = mpatches.Patch(color='blue', label='Closer to {}'.format(reference_variable))

    # Place the legend at the top of the figure
    ax.legend(handles=[red_patch, blue_patch], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, borderaxespad=0.)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    # Return the processed DataFrame
    return grouped_data

#################################neighbourhood analysis plotting ##########################################

def get_neighboring_node_abundance(node_name,connectivity_matrix,cell_abundance,source_nodes,neighbourhood_variable_filter_for_tumour_cells):
    """
    Calculate the normalized abundance of neighboring nodes for a given node.

    Parameters:
    - node_name (str): The name of the node for which neighbors' abundance is to be calculated.
    - connectivity_matrix (DataFrame): A pandas DataFrame representing the connectivity matrix where indices 
      and columns are node names. A value of 1 indicates a connection between nodes.
    - cell_abundance (DataFrame): A pandas DataFrame where each row represents a node and columns represent 
      different cell abundances.

    Returns:
    - mean (Series): A pandas Series representing the mean abundance of neighboring nodes.
    """
    # Filter connectivity matrix for the given node
    filtered=connectivity_matrix[node_name]
    # Identify neighboring nodes
    neighbouring_nodes=filtered[(filtered == 1)].index.values
    # Calculate mean abundance for neighboring nodes
    cell_abundance_filtered=cell_abundance[cell_abundance.index.isin(neighbouring_nodes)]
    

    if neighbourhood_variable_filter_for_tumour_cells is not None:
        cell_abundance_filtered = cell_abundance[cell_abundance.index.isin(neighbouring_nodes)]
        # Filter source_nodes to include only those present in cell_abundance_filtered
        filtered_source_nodes = [node for node in source_nodes if node in cell_abundance_filtered.index]
        # Calculate mean for columns in 'neighbourhood_variable_filter_for_tumour_cells' only for source nodes present in cell_abundance_filtered
        if filtered_source_nodes:
        # Calculate mean for columns in 'neighbourhood_variable_filter_for_tumour_cells' only for source nodes present in cell_abundance_filtered
            means_for_filtered_columns = cell_abundance_filtered.loc[filtered_source_nodes, neighbourhood_variable_filter_for_tumour_cells].mean(axis=0)
        else:
            # If filtered_source_nodes is empty, create an empty Series
            means_for_filtered_columns = pd.Series(np.nan, index=[neighbourhood_variable_filter_for_tumour_cells])
        if not isinstance(means_for_filtered_columns, pd.Series):
            means_for_filtered_columns = pd.Series([means_for_filtered_columns], index=[neighbourhood_variable_filter_for_tumour_cells])
        # Calculate mean for all other columns across all neighbouring nodes
        other_columns = [col for col in cell_abundance_filtered.columns if col not in neighbourhood_variable_filter_for_tumour_cells]
        means_for_other_columns = cell_abundance_filtered[other_columns].mean(axis=0)
        # Combine the two results
        mean_results = pd.concat([means_for_filtered_columns, means_for_other_columns])
    else:
        mean_results=cell_abundance_filtered.mean(axis=0)
        
    return mean_results

def calculate_neighbourhood_correlation(rings_range,adata_vis,neighbour_variables,source_nodes,neighbourhood_variable_filter_for_tumour_cells=None):
    
    results = {}
    for rings_for_neighbours_value in rings_range:
        sq.gr.spatial_neighbors(adata_vis, n_rings=rings_for_neighbours_value, coord_type="grid", n_neighs=6)
        connectivity_matrix=pd.DataFrame.sparse.from_spmatrix(adata_vis.obsp['spatial_connectivities'])
        connectivity_matrix.index=adata_vis.obs.index
        connectivity_matrix.columns=adata_vis.obs.index

        node_abundance_list = []
        for node_name in tqdm(source_nodes):
            sum_cell_abundance=get_neighboring_node_abundance(node_name,connectivity_matrix,neighbour_variables,source_nodes,neighbourhood_variable_filter_for_tumour_cells)
            node_abundance_list.append(sum_cell_abundance)
        node_df_cell_abundance = pd.DataFrame(node_abundance_list, index=source_nodes,columns=neighbour_variables.columns)
        node_df_cell_abundance.index=source_nodes
        corr_df = node_df_cell_abundance.corr()
        pval = node_df_cell_abundance.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_df.shape)

        results[rings_for_neighbours_value] = corr_df, pval
    return results



def correlation_heatmap_neighbourhood(results, *variables, save_path=None, pval_cutoff=0.05,fig_size=(5, 5)):
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
    corr_df=results[0]
    pvalue_df=results[1]
    sub_corr = corr_df.loc[variables, :]
    sub_pval = pvalue_df.loc[variables, :]
    annot = sub_pval.applymap(spl.format_pval_annotation)
    #select fig_height from fig_size
    plt.figure(figsize=fig_size)
    ax = sns.heatmap(sub_corr, cmap="RdBu_r", center=0, annot=annot, cbar=True, square=True, 
                     linewidth=0.5, linecolor='black',fmt="s")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticks(np.arange(sub_corr.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(sub_corr.columns, rotation=90)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
def plot_correlation_shifts(ring_sensitivity_results,correlation_primary_variable,save_path,fig_size=(10, 3)):
    """
    Plot the shifts in correlation over different 'ring' sizes.

    Parameters:
    - ring_sensitivity_results (dict): A dictionary where each key is the number of rings used for neighbor calculation, and each 
      value is a tuple containing the correlation matrix and the corresponding p-values matrix for those rings.
    - correlation_primary_variable (str): The primary variable for which correlations are to be plotted.
    - save_path (str): Path to save the plot.
    - fig_size (tuple, optional): The size of the figure (width, height).

    The function plots and saves a multi-subplot figure showing correlation shifts.
    """
    sns.set_style("white")  
    x_values = list(ring_sensitivity_results.keys())
    all_columns = ring_sensitivity_results[x_values[0]][0].columns
    x_indices = range(len(x_values))
    # Figure setup
    n_cols = 5
    n_rows = int(np.ceil(len(all_columns) / n_cols))
    fig_width = fig_size[0]
    fig_height = fig_size[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=False)
    fig.suptitle(f'{correlation_primary_variable} correlation shifts', fontsize=16)
    for idx, column in enumerate(all_columns):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if n_rows == 1:
            ax = axes[col_idx]
        else:
            ax = axes[row_idx, col_idx]
        y_values = [ring_sensitivity_results[key][0].loc[correlation_primary_variable, column] for key in x_values]
        y_mean = np.mean(y_values)
        ax.plot(x_indices, y_values, '-o', color='black', markersize=6, markerfacecolor='red')
        ax.set_ylim(y_mean - 0.2, y_mean + 0.2)    
        ax.set_title(column, fontsize=12)
        # Set x-ticks and labels for all rows
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_values, rotation=45, fontsize=10)
        #add x-axis label
        ax.set_xlabel("Number of rings", fontsize=12)
        #add y-axis label
        ax.set_ylabel("Correlation", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)  # Adjust y-tick font size
    # Remove any extra subplots
    if len(all_columns) % n_cols != 0:
        for j in range(len(all_columns) % n_cols, n_cols):
            fig.delaxes(axes[n_rows - 1, j])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_change(ring_sensitivity_results,correlation_primary_variable,save_path,fig_size=(10, 7)):
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
    differences = []
    x_values = list(ring_sensitivity_results.keys())
    all_columns = ring_sensitivity_results[x_values[0]][0].columns
    
    for column in all_columns:
        first_value = ring_sensitivity_results[x_values[0]][0].loc[correlation_primary_variable, column]
        last_value = ring_sensitivity_results[x_values[-1]][0].loc[correlation_primary_variable, column]
        differences.append(last_value - first_value)
    sorted_indices = np.argsort(differences)
    sorted_differences = np.array(differences)[sorted_indices]
    sorted_columns = np.array(all_columns)[sorted_indices]
    # Create gradients for both negative and positive values
    negative_values = [val for val in sorted_differences if val < 0]
    positive_values = [val for val in sorted_differences if val > 0]
    negative_colors = sns.color_palette("Blues_r", n_colors=len(negative_values))  # reversed gradient
    positive_colors = sns.color_palette("Reds", n_colors=len(positive_values)+1)  # reversed gradient
    gradient_colors = negative_colors + positive_colors

    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_style("white")

    bars = ax.bar(sorted_columns, sorted_differences, color=gradient_colors)  # Vertical bar plot
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Conrrelation change')
    ax.set_title(f'Difference in Correlation Values when varying ring size for {correlation_primary_variable}')
    ax.tick_params(axis='x', rotation=90, labelsize=12)  # Rotate x-axis labels by 45 degrees & increase font size
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


###################neighbourhood enrichment zones ############################################################
def get_neighboring_node_abundance_enrichment(node_name, cell_abundance,neighbors_matrix,source_nodes):
    # Get indices of neighboring nodes
    neighboring_nodes_indices = neighbors_matrix.loc[node_name, :]
    inner_nodes = neighboring_nodes_indices[neighboring_nodes_indices > 0].index.values
    #ensure inner nodes are tumour cells
    inner_nodes=[node for node in inner_nodes if node in source_nodes]

    # Calculate mean abundance for neighboring nodes
    mean = cell_abundance.loc[inner_nodes].mean(axis=0)
    #print("inner",inner_nodes)
    #print("inner abundance",cell_abundance.loc[inner_nodes])
    return mean,inner_nodes

# Function to identify nodes in the next outer ring
def get_outer_ring_nodes(node_name,outer_neighbours_matrix,inner_nodes):
    # Get indices of neighboring nodes for ring range + 1
    outer_ring_nodes = outer_neighbours_matrix.loc[node_name]
    outer_ring_nodes = outer_ring_nodes[outer_ring_nodes > 0].index.values
    # Exclude nodes that are also in the inner ring
    outer_ring_nodes = [node for node in outer_ring_nodes if node not in inner_nodes]
    return outer_ring_nodes

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
    inner_outer_results={}
    for ring_range in rings_range:
        # Set the maximum ring range for spatial_neighbors computation
        max_ring_range = ring_range + 1
        # Precompute spatial neighbors with the maximum ring range
        sq.gr.spatial_neighbors(adata_vis, n_rings=max_ring_range, coord_type="grid", n_neighs=6)
        neighbors_matrix_outer = pd.DataFrame.sparse.from_spmatrix(adata_vis.obsp['spatial_connectivities'])
        neighbors_matrix_outer.index = adata_vis.obs.index
        neighbors_matrix_outer.columns = adata_vis.obs.index
        if ring_range > 0:
            # Precompute spatial neighbors with the ring range
            sq.gr.spatial_neighbors(adata_vis, n_rings=ring_range, coord_type="grid", n_neighs=6)
            neighbors_matrix_inner = pd.DataFrame.sparse.from_spmatrix(adata_vis.obsp['spatial_connectivities'])
            neighbors_matrix_inner.index = adata_vis.obs.index
            neighbors_matrix_inner.columns = adata_vis.obs.index
            inner_ring_means = []
            outer_ring_means = []
            for node_name in tqdm(source_nodes):
                # Process each node
                inner_mean,neighboring_nodes = get_neighboring_node_abundance_enrichment(node_name, neighbour_variables, neighbors_matrix_inner,source_nodes)
                outer_nodes = get_outer_ring_nodes(node_name, neighbors_matrix_outer, neighboring_nodes)
                outer_mean = neighbour_variables.loc[outer_nodes].mean(axis=0)
                inner_ring_means.append(inner_mean)
                outer_ring_means.append(outer_mean)
        else:
            inner_ring_means = []
            outer_ring_means = []
            for node_name in tqdm(source_nodes):
                # Process each node
                inner_mean = neighbour_variables.loc[node_name].mean(axis=0)
                print(inner_mean)
                inner_ring_means.append(inner_mean)
                #empty neighbouring nodes
                outer_nodes = get_outer_ring_nodes(node_name, neighbors_matrix_outer, [])
                outer_ring_means.append(inner_mean)
                print(outer_ring_means)
        # Convert lists to DataFrames
        inner_df_1 = pd.DataFrame(inner_ring_means, index=source_nodes, columns=neighbour_variables.columns)
        outer_df_1 = pd.DataFrame(outer_ring_means, index=source_nodes, columns=neighbour_variables.columns)
        inner_outer_results[ring_range] = inner_df_1,outer_df_1

    return inner_outer_results

#using cell abundance from calculate_inner_outer_neighbourhood_enrichment, we calculate corr_df and pval that we can use to plot using functions previoulsy defined
#correlation_key_variable is the variable that we are calculating the correlation for (e.g. EMT_hallmarks_hot) using the inner value for it and correlating how this value affects outer ring
def calculate_corr_pvalue_for_inner_outer_neighbourhood_enrichment(results,correlation_key_variable,rings_range):
    """
    Calculates correlation and p-value for inner and outer neighborhood enrichment.

    Parameters:
    - results (dict): Dictionary containing inner and outer DataFrame cell abundance results from previous analysis (calculate_inner_outer_neighbourhood_enrichment())
    - correlation_key_variable (str): The key variable for which correlation is calculated in inner ring. We are comparing how a value of this variable in inner ring affects outer ring.
    - rings_range (list): List of ring ranges for which correlation and p-value are calculated.

    Returns:
    - Dict: Dictionary containing correlation DataFrame and p-value for each ring. We can then run previous plots e.g. plot_correlation_shifts() on this. Equally accessing each element 
    of dict returns correlations and pvalue for that ring range.
    """
    corr_pval_results={}
    for ring in rings_range:
        inner_df,outer_df=results[ring]
        outer_df[f'{correlation_key_variable}_inner_values']=inner_df[correlation_key_variable]
        corr_df=outer_df.corr()
        #if index contains 'Cancer' or 'Luminal' or 'Myoepithelial' remove from corr_df
        corr_df=corr_df[~corr_df.index.str.contains('Cancer')]
        corr_df=corr_df[~corr_df.index.str.contains('Luminal')]
        corr_df=corr_df[~corr_df.index.str.contains('Myoepithelial')]
        corr_df=outer_df.corr()
        p_val=outer_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_df.shape)
        corr_pval_results[ring] = corr_df, p_val
    return corr_pval_results


def plot_results_gene_signatures_heatmap(adata_vis, variable_one, comparison_variable, genes, file_path_plots,gene_signature_name):
    spl.plot_results_gene_signatures_heatmap(adata_vis, variable_one, comparison_variable, genes, file_path_plots,gene_signature_name)

def plot_condition_differences(adata, variable_of_interest, conditions, save_path=None):
    spl.plot_condition_differences(adata, variable_of_interest, conditions, save_path)


def add_genes_to_obs(adata, gene_list):
    """
    Check if genes are in anndata.var_names, and if so, add their expression to anndata.obs.
    
    Parameters:
    - adata: An anndata object.
    - gene_list: A list of gene names to check and add.
    
    Returns:
    - Updates the anndata object in place, no return value.
    """
    # Check each gene in the provided list
    for gene in gene_list:
        if gene in adata.var_names:
            # Get the index of the gene in var_names
            gene_index = adata.var_names.get_loc(gene)
            # Extract the expression values from the .X attribute

            gene_expression = adata.X[:, gene_index].toarray().flatten() if isspmatrix(adata.X) else adata.X[:, gene_index]
            # Add to .obs with gene name as the new column name
            adata.obs[gene] = gene_expression
        else:
            print(f"Gene {gene} not found in var_names.")
