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

# Typing
from typing import Tuple, List, Optional, Union
# Statsmodels for statistics
from statsmodels.sandbox.stats.multicomp import multipletests
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='libpysal')

#pdf font
plt.rcParams['pdf.fonttype'] = 'truetype' 


def custom_color(pvalue):
    """Define custom color based on p-value."""
    if pvalue < 0.001:
        return "#D53E4F"  # Red (Significant)
    elif pvalue < 0.01:
        return "#FDAE61"  # Orange
    elif pvalue < 0.05:
        return "#FEE08B"  # Yellow (Less significant)
    else:
        return "#E6F598"  # Light Green (Not significant)

def calculate_pvalue(tme_var, emt_var_one, emt_var_two, df):
    """Calculate p-value between two groups."""
    group1 = df[(df['comparison_variable'] == tme_var) & (df['primary_variable'] == emt_var_one)]['min_distance']
    group2 = df[(df['comparison_variable'] == tme_var) & (df['primary_variable'] == emt_var_two)]['min_distance']
    _, p_value = ttest_ind(group1, group2, equal_var=False)
    return p_value


# Calculate the mean scores for each state and signature; helper function
def calculate_mean_scores(adata, signatures, states_to_loop_through):
    mean_scores = {}
    data_dict = {}
    for state in states_to_loop_through:
        mean_scores[state] = {}
        for signature in signatures:
            scores = adata.obs.loc[adata.obs[state].notna(), signature]
            data_dict[(state, signature)] = scores
            mean_scores[state][signature] = scores.mean()
    return mean_scores, data_dict

# Create a DataFrame for heatmap with states as rows and signatures as columns; helper function
def create_heatmap_data(mean_scores, states_to_loop_through, signatures):
    data_for_heatmap = pd.DataFrame(index=states_to_loop_through, columns=signatures)
    for state in states_to_loop_through:
        for signature in signatures:
            data_for_heatmap.loc[state, signature] = mean_scores[state][signature]
    
    # Convert values to numeric
    data_for_heatmap = data_for_heatmap.apply(pd.to_numeric, errors='coerce')
    
    return data_for_heatmap

# Normalize the data for heatmap; helper function; helper function
def normalize_heatmap_data(data_for_heatmap):
    data_for_heatmap_normalized = data_for_heatmap.subtract(data_for_heatmap.min(axis=0), axis=1)
    data_for_heatmap_normalized = data_for_heatmap_normalized.divide(data_for_heatmap_normalized.max(axis=0), axis=1)
    return data_for_heatmap_normalized

# Plot heatmap using Seaborn; helper function
def plot_heatmap(data_for_heatmap_normalized, signatures, states_to_loop_through,fig_size, plot_score=False):
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(data_for_heatmap_normalized, cmap=cmap, annot=plot_score, fmt=".2f" if plot_score else None, ax=ax,
                cbar_kws={'label': 'Normalized Mean Score'}, linewidths=1)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    ax.set_title('Mean Scores Heatmap')
    rect = Rectangle((0, 0), len(signatures), len(states_to_loop_through), linewidth=1, edgecolor='black',
                     facecolor='none')
    ax.add_patch(rect)
    ax.set_aspect('equal')
    plt.show()

    #helper function
def calculate_signature_differences(anndata_breast, gene_signatures, states):
    """
    Calculates differences in gene signatures between two states.

    :param anndata_breast: AnnData object containing gene expression data.
    :param gene_signatures: List of gene signatures to compare.
    :param states: List of two states to compare.
    :return: DataFrame with the calculated differences and p-values.
    """
    results = []
    for signature in gene_signatures:
        state_scores = [anndata_breast.obs.loc[anndata_breast.obs[state].notna(), signature] for state in states]
        means = [scores.mean() for scores in state_scores]
        difference = means[0] - means[1]
        _, pvalue = ttest_ind(*[scores.dropna() for scores in state_scores])

        results.append({
            'Signature': signature,
            'State 1 Mean': means[0],
            'State 2 Mean': means[1],
            'Difference': difference,
            'Pvalue': pvalue
        }) 
    return pd.DataFrame(results)

#helper function
def plot_bubble_chart(data, states, fig_size, bubble_size):
    fig, ax = plt.subplots(figsize=fig_size)
    for _, row in data.iterrows():
        color = 'red' if row['Difference'] > 0 else ('blue' if row['Difference'] < 0 else 'white')
        ax.scatter(row['Signature'], 0, s=bubble_size, color=color, edgecolors='black', linewidth=0.5)
    
    ax.set_yticks([])
    ax.set_title('Gene Signatures Comparison')
    plt.xticks(rotation=90)
    add_legends(ax, states)
    plt.subplots_adjust(right=0.9)
    plt.savefig("enrichment.pdf", bbox_inches='tight')
    plt.show()

#helper function
def add_legends(ax, states):
    labels = [states[0], states[1], 'No Difference']
    colors = ['red', 'blue', 'white']
    handles = [Patch(facecolor=color, edgecolor='black') for color in colors]
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(1.1, -0.2), frameon=True, title='Signature')




def plot_sensitivity(x_values, cell_data, variable_comparison, variable_one, variable_two, variable_three, filename,parameter_changing):
    """
    Plot the sensitivity analysis results.

    Parameters:
    - x_values: List of neighbourhood values tested.
    - cell_data: Dictionary containing the calculated distances.
    - variable_comparison, variable_one, variable_two, variable_three: Hotspot variables .
    - filename: Name of the file to save the plot.
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.plot(x_values, cell_data['distance_variable_one'], label=variable_one, linewidth=2)
    ax.plot(x_values, cell_data['distance_variable_two'], label=variable_two, linewidth=2)
    ax.plot(x_values, cell_data['distance_variable_three'], label=variable_three, linewidth=2)
    ax.set_xlabel(parameter_changing, fontsize=10)
    ax.set_ylabel("Distance", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title("Hotspot Sensitivitiy Test to "+variable_comparison)
    #ax.set_xticks(x_values)
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='y', labelsize=10)
    #plt.show()
    plt.savefig(f"{filename}_{parameter_changing}.pdf", bbox_inches='tight')


def add_hotspots_to_fullanndata(spatial_anndata,hotspot_to_add,batch,old_anndata):
    mask = spatial_anndata.obs['batch'] == batch
    new_column_name = hotspot_to_add + "_hot"
    spatial_anndata.obs.loc[mask, new_column_name] = old_anndata.obs[new_column_name]
    new_column_name = hotspot_to_add + "_cold"
    spatial_anndata.obs.loc[mask, new_column_name] = old_anndata.obs[new_column_name]
    return spatial_anndata



#helper function
def filter_dataframe(df, primary_vars, comp_vars):
    return df[df['primary_variable'].isin(primary_vars) & df['comparison_variable'].isin(comp_vars)]

#helper function
def calculate_differences(df, ref_values):
    #negative value equals closer to primary variable
    return df.apply(lambda row: row['min_distance'] - ref_values.get((row['batch'], row['comparison_variable']), row['min_distance']), axis=1)

#helper function
def calculate_pvalues(df, primary_variable_value, reference_variable):
    pvalues = {}
    grouped = df.groupby(['batch', 'comparison_variable'])
    for (slide, tme_var), group in grouped:
        emt_values = group[group['primary_variable'] == primary_variable_value]['min_distance']
        ref_values = group[group['primary_variable'] == reference_variable]['min_distance']
        _, pvalue = ttest_ind(emt_values, ref_values)
        pvalues[(slide, tme_var)] = pvalue
    return pvalues

#helper function
def prepare_data_hotspot(df, primary_variable_value, comparison_variable_values, reference_variable='tumour_cells'):
    filtered_df = filter_dataframe(df, [primary_variable_value, reference_variable], comparison_variable_values)
    mean_df = filtered_df.groupby(['primary_variable', 'comparison_variable', 'batch']).min_distance.mean().reset_index()
    ref_values = mean_df[mean_df['primary_variable'] == reference_variable].set_index(['batch', 'comparison_variable'])['min_distance']
    mean_df['Difference'] = calculate_differences(mean_df, ref_values)
    pvalues = calculate_pvalues(filtered_df, primary_variable_value, reference_variable)
    mean_df['Pvalue'] = mean_df.apply(lambda row: pvalues.get((row['batch'], row['comparison_variable']), np.nan), axis=1)
    return mean_df

#helper function
def format_pval_annotation(pval):
    if pval <= 0.0001:
        return '****'
    elif pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    else:
        return ''