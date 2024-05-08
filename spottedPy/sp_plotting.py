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
from scipy.stats import mannwhitneyu

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
    elif pvalue < 0.1:
        return "#DED9A9" # Light Green (Not significant)
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

def calculate_mean_scores_per_batch(adata, signatures, states_to_loop_through):
    mean_scores = {}
    data_dict = {}
    unique_batches = adata.obs['batch'].unique()

    for batch in unique_batches:
        mean_scores[batch] = {}
        batch_data = adata[adata.obs['batch'] == batch]

        for state in states_to_loop_through:
            mean_scores[batch][state] = {}
            for signature in signatures:
                scores = batch_data.obs.loc[batch_data.obs[state].notna(), signature]
                data_dict[(batch, state, signature)] = scores
                mean_scores[batch][state][signature] = scores.mean()
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

def create_heatmap_data_per_batch(mean_scores, states_to_loop_through, signatures):
    # Prepare a list to hold all rows of the DataFrame
    rows = []
    # Iterate over each batch, state, and signature to populate the rows list
    for batch, states in mean_scores.items():
        for state in states_to_loop_through:
            row = {'Batch': batch, 'State': state}
            for signature in signatures:
                row[signature] = states.get(state, {}).get(signature, None)
            rows.append(row)
    # Create a DataFrame from the rows
    data_for_heatmap = pd.DataFrame(rows)
    # Set index to Batch and State
    data_for_heatmap.set_index(['Batch', 'State'], inplace=True)
    # Convert values to numeric, ignoring errors
    data_for_heatmap = data_for_heatmap.apply(pd.to_numeric, errors='coerce')
    return data_for_heatmap

# Normalize the data for heatmap; helper function; helper function
def normalize_heatmap_data(data_for_heatmap):
    data_for_heatmap_normalized = data_for_heatmap.subtract(data_for_heatmap.min(axis=0), axis=1)
    data_for_heatmap_normalized = data_for_heatmap_normalized.divide(data_for_heatmap_normalized.max(axis=0), axis=1)
    return data_for_heatmap_normalized

def plot_heatmap(data_for_heatmap_normalized, signatures, states_to_loop_through, fig_size, plot_score=False,save_path=None):
    fig, ax = plt.subplots(figsize=fig_size)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Check if the DataFrame has a MultiIndex (batch data included)
    if isinstance(data_for_heatmap_normalized.index, pd.MultiIndex):
        # Use the levels of the MultiIndex to adjust the plot
        data_for_plot = data_for_heatmap_normalized.unstack(level=0)
        sns.heatmap(data_for_plot, cmap=cmap, annot=plot_score, fmt=".2f" if plot_score else None, ax=ax,
                    cbar_kws={'label': 'Normalized Mean Score'}, linewidths=1)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha="right")
        ax.set_title('Mean Scores Heatmap per Batch')
    else:
        # Plot as usual for non-batch data
        sns.heatmap(data_for_heatmap_normalized, cmap=cmap, annot=plot_score, fmt=".2f" if plot_score else None, ax=ax,
                    cbar_kws={'label': 'Normalized Mean Score'}, linewidths=1)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha="right")
        ax.set_title('Mean Scores Heatmap')
        rect = Rectangle((0, 0), len(signatures), len(states_to_loop_through), linewidth=1, edgecolor='black',
                         facecolor='none')
        ax.add_patch(rect)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

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
    #add hotspot label
    spatial_anndata.obs.loc[mask, hotspot_to_add+"_hot_number"] = old_anndata.obs[hotspot_to_add+"_hot_number"]
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
def prepare_data_hotspot(df, primary_variable_value, comparison_variable_values, reference_variable):
    filtered_df = filter_dataframe(df, [primary_variable_value, reference_variable], comparison_variable_values)
    mean_df = filtered_df.groupby(['primary_variable', 'comparison_variable', 'batch']).min_distance.mean().reset_index()
    ref_values = mean_df[mean_df['primary_variable'] == reference_variable].set_index(['batch', 'comparison_variable'])['min_distance']
    mean_df['Difference'] = calculate_differences(mean_df, ref_values)
    pvalues = calculate_pvalues(filtered_df, primary_variable_value, reference_variable)
    mean_df['Pvalue'] = mean_df.apply(lambda row: pvalues.get((row['batch'], row['comparison_variable']), np.nan), axis=1)
    return mean_df

#helper function
def format_pval_annotation(pval):

    if pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    else:
        return ''
    


def calculate_upper_whisker(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    return data[data <= upper_bound].max()




def plot_condition_differences(adata, variable_of_interest, conditions, save_path=None):
    if len(conditions) != 2:
        raise ValueError("Exactly two conditions must be provided.")
    

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Batch': adata.obs['batch'],
        variable_of_interest: adata.obs[variable_of_interest]
    })
    

    for condition in conditions:
        plot_data[condition] = adata.obs[condition]

    # Melt the DataFrame to long format for easier plotting with seaborn
    plot_data_melted = plot_data.melt(id_vars=['Batch', variable_of_interest], value_vars=conditions, var_name='Condition', value_name='Level')

    # Filter out rows where Quiescence_Level is NaN
    plot_data_filtered = plot_data_melted.dropna(subset=['Level'])

    # Start with a larger figure size for better readability
    plt.figure(figsize=(10, 5))
    # Create the boxplot
    boxplot = sns.boxplot(x='Batch', y=variable_of_interest, hue='Condition', data=plot_data_filtered, showfliers=False)
    plt.ylabel('')

    # Adjust font size for better visibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Annotation setup
    batch_levels = plot_data_filtered['Batch'].unique()
    condition_levels = plot_data_filtered['Condition'].unique()

    # Loop through batches to calculate p-values and determine positions
    for i, batch in enumerate(batch_levels):
        batch_data = plot_data_filtered[plot_data_filtered['Batch'] == batch]
        data1 = batch_data[batch_data['Condition'] == conditions[0]][variable_of_interest]
        data2 = batch_data[batch_data['Condition'] == conditions[1]][variable_of_interest]

        # Calculate whisker values for each condition
        upper_whisker_data1 = calculate_upper_whisker(data1)
        upper_whisker_data2 = calculate_upper_whisker(data2)

        # Determine the highest point for annotation
        max_value = max(upper_whisker_data1, upper_whisker_data2)
        y_offset = 0.05 * max_value  # Slight offset above the max value

        # Calculate p-value between conditions
        pvalue = mannwhitneyu(data1, data2).pvalue
        significance = format_pval_annotation(pvalue)

        plt.text(i, max_value + y_offset, f'{significance}', ha='center', va='bottom', fontsize=14)

    plt.title(f'{variable_of_interest} in Conditions across Batches', fontsize=16)
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Adjust layout to ensure everything fits without clipping
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def prepare_data(adata_vis, variable_one, comparison_variable):
    mask_one = pd.notna(adata_vis.obs[variable_one])
    mask_two = pd.notna(adata_vis.obs[comparison_variable])
    return mask_one, mask_two

def compute_values_for_genes(adata_vis, genes, batches, mask_one, mask_two):
    gene_positions = {gene: idx for idx, gene in enumerate(genes)}
    results = []
    for batch in batches:
        batch_mask = adata_vis.obs['batch'] == batch
        for gene_name in genes:
            gene_index = adata_vis.var.index.get_loc(gene_name)
            gene_values = adata_vis.X[:, gene_index].toarray().flatten()
            mask_combined = batch_mask & mask_one
            difference = gene_values[mask_combined].mean() - gene_values[batch_mask & mask_two].mean()
            _, pvalue = ttest_ind(gene_values[mask_combined], gene_values[batch_mask & mask_two])
            results.append({'Batch': batch, 'Gene': gene_name, 'Difference': difference, 'Pvalue': pvalue})
    return pd.DataFrame(results)

#comparison_variable='tumour_cells' to compare values between
def plot_results_gene_signatures_heatmap(adata_vis, variable_one, comparison_variable, genes, file_path_plots,gene_signature_name):
    # Prepare data
    batches=adata_vis.obs['batch'].unique()

    mask_one, mask_two = prepare_data(adata_vis, variable_one, comparison_variable)
    gene_positions = {gene: idx for idx, gene in enumerate(genes)}
    # Compute values for genes
    df_results = compute_values_for_genes(adata_vis, genes, batches, mask_one, mask_two)
    heatmap_data = np.full((len(batches), len(genes)), np.nan)  # Initialize with NaN

    for batch in batches:
        data = df_results[df_results['Batch'] == batch]
        for _, row in data.iterrows():
            if row['Pvalue'] < 0.05:
                x_pos = gene_positions[row['Gene']]
                heatmap_data[list(batches).index(batch), x_pos] = row['Difference']

    # Create colormap
    cmap = plt.get_cmap('bwr')
    cmap.set_bad('white')  # Set NaN color to white

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(6, 7.2))
        # Determine the maximum absolute value in your heatmap data (ignoring NaNs)
    max_abs_value = np.nanmax(np.abs(heatmap_data))

    # Create the heatmap ensuring that the colormap is symmetric around zero
    cax = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, vmin=-max_abs_value, vmax=max_abs_value)

    # Labeling, etc.
    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right",fontsize=15)
    ax.set_yticks(np.arange(len(batches)))
    ax.set_yticklabels([f'{batch}' for batch in batches],fontsize=15)
    ax.grid(False)

    # Colorbar for reference
    #cbar = plt.colorbar(cax, orientation='vertical', pad=0.01)
    #
    # cbar.set_label('Difference', rotation=270, labelpad=15)
    plt.tight_layout()

    path = os.path.join(file_path_plots, variable_one + "_" + comparison_variable + "_" + gene_signature_name + "_heatmap.pdf")
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()
