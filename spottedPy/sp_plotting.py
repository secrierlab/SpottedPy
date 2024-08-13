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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import combine_pvalues



# Typing
from typing import Tuple, List, Optional, Union
# Statsmodels for statistics
from statsmodels.sandbox.stats.multicomp import multipletests
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='libpysal')

#pdf font
plt.rcParams['pdf.fonttype'] = 'truetype' 


def plot_hotspots(anndata, column_name, batch_single, save_path, color_for_spots):
    if batch_single is not None:
        data_subset = anndata[anndata.obs['batch'] == str(batch_single)].copy()
        sc.pl.spatial(data_subset, color=[column_name],  vmax='p0',color_map=color_for_spots, library_id=str(batch_single),save=f"_{save_path}",colorbar_loc=None,alpha_img= 0.5)
    else:
        for batch in anndata.obs['batch'].unique():
            data_subset = anndata[anndata.obs['batch'] == str(batch)].copy()
            sc.pl.spatial(data_subset, color=[column_name],  vmax='p0',color_map=color_for_spots, library_id=str(batch),save=f"_{str(batch)}_{save_path}",colorbar_loc=None,alpha_img= 0.5)

def custom_color(pvalue):
    """Define custom color based on p-value."""
    if pvalue < 0.001:
        return "#D53E4F"  # Red (Significant)
    elif pvalue < 0.01:
        return "#FDAE61"  # Orange
    elif pvalue < 0.05:
        return "#FEE08B"  # Yellow (Less significant)
    elif pvalue < 0.1:
        return "#DED9A9" #  (Not significant)
    else:#pvalue > 0.1:
        return "#E6F598"  # Light Green (Not significant)
###custom scatter plot
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

def plot_bubble_plot_mean_distances(distances_df, primary_vars, comparison_vars,normalise_by_row=False,fig_size=(5, 5),save_path=None):
    # Filter the DataFrame based on the specified primary and comparison variables
    filtered_df = distances_df[
        distances_df['primary_variable'].isin(primary_vars) &
        distances_df['comparison_variable'].isin(comparison_vars)
    ]
    mean_df=filtered_df.groupby(['batch','primary_variable', 'hotspot_number']).min_distance.median().reset_index()
    # Group by primary and comparison variables and calculate the mean distance
    mean_df = (
        filtered_df
        .groupby(['primary_variable', 'comparison_variable'])
        .min_distance
        .mean()
        .reset_index()
    )
    #sort by mean distance
    mean_df=mean_df.sort_values(by='min_distance',ascending=False)
    #normalise by min_distance
    if normalise_by_row:
        mean_df['min_distance']=mean_df['min_distance']/mean_df['min_distance'].max()
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
        plt.xticks(rotation=45, ha='right')
        plt.tick_params(axis='both', which='major', labelsize=15)

        #add rotation to y axis
        #plt.yticks(rotation=-45, ha='right')

        # Adjust plot layout and show plot
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()



def plot_custom_scatter(data, primary_vars, comparison_vars, fig_size, bubble_size, file_save,sort_by_difference, compare_distribution_metric, statistical_test):
    # Set plot style and font
    sns.set_style("white")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    
    if compare_distribution_metric not in [None, 'min', 'mean', 'median', 'ks_test','median_across_all_batches']:
        raise ValueError("compare_distribution_metric must be one of 'min', 'mean', 'median','median_across_all_batches' or 'ks_test'.")

    if compare_distribution_metric is None:
        # Filter the data
        filtered_df = data[data['primary_variable'].isin(primary_vars)]
        filtered_df = filtered_df[filtered_df['comparison_variable'].isin(comparison_vars)]
        mean_df = (filtered_df.groupby(['primary_variable', 'comparison_variable'])
                .min_distance.mean()
                .reset_index())
        comparison_var_one = primary_vars[0]
        comparison_var_two = primary_vars[1]
        # Calculate differences and p-values
        pivot_df = mean_df.pivot(index='comparison_variable', columns='primary_variable', values='min_distance')
        pivot_df['difference'] = pivot_df[comparison_var_one] - pivot_df[comparison_var_two]
        pivot_df = pivot_df.reset_index()
        pivot_df['p_value'] = pivot_df['comparison_variable'].apply(lambda x: calculate_pvalue(x, comparison_var_one, comparison_var_two, filtered_df))
        pivot_df['color'] = pivot_df['p_value'].apply(custom_color)
        # Sort the DataFrame
        if sort_by_difference:
            pivot_df = pivot_df.reindex(pivot_df['difference'].abs().sort_values(ascending=False).index)
        else:
            #make comparison_variable a categorical variable
            pivot_df['comparison_variable'] = pd.Categorical(pivot_df['comparison_variable'], categories=comparison_vars, ordered=True)
            pivot_df.sort_values(by='comparison_variable', inplace=True)

    #test different metric of distance distributions.. min, median, mean
    #df with index as comparison_variable, column as primary_variable, and dufference as coefficient and p_value as p_value and sort by coefficient
    if compare_distribution_metric is not None and compare_distribution_metric in ['mean', 'median', 'min']:
        results_df = pd.DataFrame(columns=['comparison_variable', 'coefficient', 'p_value'])

        for comp_var in comparison_vars:
            comparison_var_two = primary_vars[1]
            comparison_var_one = primary_vars[0]
            #filter for comp_var
            distance_vals_filtered=data[data['comparison_variable']==comp_var].copy()

            distance_vals_filtered.loc[:, 'batch'] = distance_vals_filtered['batch'].astype('category')
            #if compare_distribution_metric is mean, median or min ensure there are multiple batches, otherwise not enough power
            if len(distance_vals_filtered['batch'].unique())<2:
                #raise error
                raise ValueError("Not enough slides to perform statistical test. Requires more than one slide. Otherwise, group all distances from hotspots by setting compare_distribution_metric to median_across_all_batches")
            if compare_distribution_metric=="mean":
                min_distances = distance_vals_filtered.groupby(['batch','primary_variable', 'hotspot_number']).min_distance.mean().reset_index()
            if compare_distribution_metric=="median":
                min_distances = distance_vals_filtered.groupby(['batch','primary_variable', 'hotspot_number']).min_distance.median().reset_index()
            if compare_distribution_metric=="min":
                min_distances = distance_vals_filtered.groupby(['batch','primary_variable', 'hotspot_number']).min_distance.min().reset_index()


            model_formula = f"min_distance ~ C(primary_variable, Treatment(reference='{comparison_var_two}'))"
            model = smf.gee(
                model_formula,
                "batch",  
                data=min_distances,
                family=sm.families.Gaussian()
            )
            result = model.fit()
            coef_key = f'C(primary_variable, Treatment(reference=\'{comparison_var_two}\'))[T.{comparison_var_one}]'

            coefficient = result.params.get(coef_key)
            p_value = result.pvalues.get(coef_key)
            # Append results to the DataFrame
            new_row = pd.DataFrame([{
                'comparison_variable': comp_var,
                'difference': coefficient,
                'p_value': p_value
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            #calcluate color
            results_df['color'] = results_df['p_value'].apply(custom_color)

        if sort_by_difference:
            pivot_df=results_df.reindex(results_df['difference'].abs().sort_values(ascending=False).index)
        else: 
            results_df['comparison_variable'] = pd.Categorical(results_df['comparison_variable'], categories=comparison_vars, ordered=True)
            # Sort by the column
            results_df.sort_values(by='comparison_variable', inplace=True)
            pivot_df=results_df
            #sort by coefficient

    ###simplest approach; compare the medians of all hotspots within all batches
    if compare_distribution_metric== "median_across_all_batches":
        comparison_var_one = primary_vars[0]
        comparison_var_two = primary_vars[1]
        filtered_df = data[data['primary_variable'].isin([comparison_var_one, comparison_var_two])]
        # Group by 'comparison_variable', 'primary_variable', and 'hotspot_number' and calculate the median 'min_distance'
        median_distances = filtered_df.groupby(["primary_variable","hotspot_number",'comparison_variable'])['min_distance'].median().reset_index()
        #now calculate min_distance for each primary_variable and comparison_variable
        median_distances_group_hotspots = median_distances.groupby(["primary_variable","comparison_variable",])['min_distance'].median().reset_index()
        # Pivot the data to have 'EMT_hallmarks_hot' and 'EMT_hallmarks_cold' as columns
        pivot_df = median_distances_group_hotspots.pivot_table(index=['comparison_variable'], columns='primary_variable', values='min_distance').reset_index()
        pivot_df['difference']=pivot_df[comparison_var_one]-pivot_df[ comparison_var_two]
        # Perform a t-test between 'EMT_hallmarks_hot' and 'EMT_hallmarks_cold' for each 'comparison_variable'
        ttest_results = {}
        for comparison_variable in median_distances['comparison_variable'].unique():
            comp_df = median_distances[median_distances['comparison_variable'] == comparison_variable]
            hot_values=comp_df[comp_df['primary_variable'] == comparison_var_one]
            cold_values = comp_df[comp_df['primary_variable'] ==  comparison_var_two]
            t_stat, p_val = ttest_ind(hot_values['min_distance'], cold_values['min_distance'])
            ttest_results[comparison_variable] = {'t_stat': t_stat, 'p_val': p_val}
        # Convert t-test results into a DataFrame
        ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index').reset_index()
        ttest_df.columns = ['comparison_variable', 't_stat', 'p_value']
        # Merge the t-test results with the pivot_df
        results_df = pd.merge(pivot_df, ttest_df, on='comparison_variable', how='left')
        results_df['color'] = results_df['p_value'].apply(custom_color)  
        if sort_by_difference:
            pivot_df=results_df.reindex(results_df['difference'].abs().sort_values(ascending=False).index)
        else: 
            results_df['comparison_variable'] = pd.Categorical(results_df['comparison_variable'], categories=comparison_vars, ordered=True)
            # Sort by the column
            results_df.sort_values(by='comparison_variable', inplace=True)
            pivot_df=results_df

    #Kolmogorov-Smirnov Test to analyse how different the distance distributions are, but note: this doesnt compare at the hotspot level
    if compare_distribution_metric=="ks_test":
        results_df = pd.DataFrame(columns=['comparison_variable', 'coefficient', 'p_value'])
        for comp_var in comparison_vars:
            comparison_var_two = primary_vars[1]
            comparison_var_one = primary_vars[0]
            # Filter data for comp_var
            distance_vals_filtered = data[data['comparison_variable'] == comp_var]
            distance_vals_filtered.loc[:, 'batch'] = distance_vals_filtered['batch'].astype('category')

            # List to store results for combining later
            ks_statistics = []
            p_values = []
            # Perform KS test for each batch and primary variable pair
            for batch in distance_vals_filtered['batch'].cat.categories:
                batch_data = distance_vals_filtered[distance_vals_filtered['batch'] == batch]
                data1 = batch_data[batch_data['primary_variable'] == comparison_var_one]['min_distance']
                data2 = batch_data[batch_data['primary_variable'] == comparison_var_two]['min_distance']
                statistic, p_value = mannwhitneyu(data1, data2, alternative='less')

                #ks_statistic, p_value = stats.ks_2samp(data1, data2)
                ks_statistics.append(statistic)
                p_values.append(p_value)
            # Combine results across batches
            average_ks = np.mean(ks_statistics)
            combined_p_value = stats.combine_pvalues(p_values, method='fisher')[1]  # Using Fisher's method
            
            # Append results to the DataFrame
            results_df = results_df.append({
                'comparison_variable': comp_var,
                'difference': average_ks,  # Using average KS as the 'difference'
                'p_value': combined_p_value
            }, ignore_index=True)
            
            # Calculate color based on p-value
        results_df['color'] = results_df['p_value'].apply(custom_color)  
        if sort_by_difference:
            pivot_df=results_df.reindex(results_df['difference'].abs().sort_values(ascending=False).index)
        else: 
            results_df['comparison_variable'] = pd.Categorical(results_df['comparison_variable'], categories=comparison_vars, ordered=True)
# Sort by the column
            results_df.sort_values(by='comparison_variable', inplace=True)
            pivot_df=results_df

    # Plot the data
    # Plot the data
    plt.figure(figsize=fig_size)
    ax = sns.scatterplot(x='comparison_variable', y='difference', size=1,
                        sizes=bubble_size, data=pivot_df, hue='color', 
                        palette={'#D53E4F': '#D53E4F', '#FDAE61': '#FDAE61', '#FEE08B': '#FEE08B', '#E6F598': '#E6F598','#DED9A9':'#DED9A9'}, 
                        legend=None)
    plt.axhline(0, color='gray', linestyle='--')
    plt.ylabel('Closer to\n{} ←→ {}'.format(comparison_var_one, comparison_var_two))
    plt.xticks(rotation=90, ha='right', rotation_mode='anchor')
    plt.tick_params(axis='x', which='both', direction='in', length=6, width=2)    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='p < 0.001', markersize=10, markerfacecolor='#D53E4F'),
                       Line2D([0], [0], marker='o', color='w', label='p < 0.01', markersize=10, markerfacecolor='#FDAE61'),
                       Line2D([0], [0], marker='o', color='w', label='p < 0.05', markersize=10, markerfacecolor='#FEE08B'),
                       Line2D([0], [0], marker='o', color='w', label='p < 0.1', markersize=10, markerfacecolor='#DED9A9'),
                       Line2D([0], [0], marker='o', color='w', label='p >= 0.05', markersize=10, markerfacecolor='#E6F598')]
    plt.legend(handles=legend_elements, loc='lower right')
    sns.despine()
    # Save the plot
    if file_save: 
        plt.savefig(f"{file_save}_scatterplot_hallmarks.pdf", dpi=300,bbox_inches='tight')
    plt.show()
    if statistical_test:
        return results_df
    
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
                    x='primary_variable', y='min_distance', ax=ax,palette='viridis')
        ax.set_title(comparison_variable)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()

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

def plot_gene_heatmap(adata, signatures, states_to_loop_through, plot_score=False, normalize_values=False,fig_size=(5, 5),
                      score_by_batch=False,save_path=None):
    # Calculate mean scores for each state and signature.
    if score_by_batch:
        # Calculate mean scores for each state and signature.
        mean_scores, data_dict = calculate_mean_scores_per_batch(adata, signatures, states_to_loop_through)
        data_for_heatmap = create_heatmap_data_per_batch(mean_scores, states_to_loop_through, signatures)
    else:
         # Calculate mean scores for each state and signature within each batch.
        mean_scores, data_dict = calculate_mean_scores(adata, signatures, states_to_loop_through)
        # Create a DataFrame for heatmap with states as rows and signatures as columns.
        data_for_heatmap = create_heatmap_data(mean_scores, states_to_loop_through, signatures)

    # Normalize the data for heatmap if required.
    if normalize_values:
        data_for_heatmap = normalize_heatmap_data(data_for_heatmap)
    # Plot the heatmap using the normalized data.
    plot_heatmap(data_for_heatmap, signatures, states_to_loop_through, fig_size,plot_score,save_path)

    return data_for_heatmap


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

def plot_signature_boxplot(anndata_breast,hotspot_variable,signature,fig_size,file_save):
    hot_data = anndata_breast.obs[~anndata_breast.obs[hotspot_variable[0]].isna()][signature]
    cold_data = anndata_breast.obs[~anndata_breast.obs[hotspot_variable[1]].isna()][signature]
    # Plotting
    data = pd.DataFrame({
    hotspot_variable[0]: hot_data,
    hotspot_variable[1]: cold_data
    })

    # Melting the DataFrame to long format for seaborn
    data_melted = data.melt(var_name='Hotspot', value_name='Response to Checkpoint Score')
    # Plotting
    plt.figure(figsize=fig_size)
    sns.boxplot(x='Hotspot', y='Response to Checkpoint Score', data=data_melted, showfliers=False)
    plt.title('Response to Checkpoint Genes based on EMT Hallmarks')
    plt.ylabel('Response to Checkpoint Score')
    if file_save: 
        plt.savefig(f"{file_save}_overall_comparison.pdf", dpi=300)
    plt.show()

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
    #add new_column_name to spatial_anndata as empty column
    spatial_anndata.obs.loc[mask, new_column_name] = old_anndata.obs[new_column_name]
    new_column_name = hotspot_to_add + "_cold"
    spatial_anndata.obs.loc[mask, new_column_name] = old_anndata.obs[new_column_name]
   
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
def plot_results_gene_signatures_heatmap(adata_vis, variable_one, comparison_variable, genes, file_path_plots,gene_signature_name,fig_size):
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
    fig, ax = plt.subplots(figsize=fig_size)
        # Determine the maximum absolute value in your heatmap data (ignoring NaNs)
    max_abs_value = np.nanmax(np.abs(heatmap_data))

    # Create the heatmap ensuring that the colormap is symmetric around zero
    cax = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, vmin=-max_abs_value, vmax=max_abs_value)

    # Labeling, etc.
    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes,rotation=45, ha="right",fontsize=15)
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
    # Prepare the data for plotting
    grouped_data = prepare_data_hotspot(df, primary_variable_value, comparison_variable_values,reference_variable)
    fig, ax = plt.subplots(figsize=fig_size)
    slides = df['batch'].unique()

    if slide_order:
        slide_positions = slide_order
    else:
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


def plot_correlation_coefficients_bar_chart(correlation_dict,ring_value,variable_to_compare,save_path=None,fig_size=(10,8)):
    """
    Plots a bar chart of correlation coefficients for a specified variable compared across different metrics.

    Parameters:
    - correlation_dict (dict): Dictionary containing correlation data for various ring sizes.
    - ring_value (str): The key within 'results' dict that refers to a specific set of correlation data.
    - variable_to_compare (str): The specific variable within the ring data to compare across metrics.
    """

    series=correlation_dict[ring_value][variable_to_compare]
    try:
        series = series.drop(index=variable_to_compare)
    #if the row does not exist, pass
    except:
        pass
    # Sort series by absolute values while preserving the sign
    sorted_series = series.sort_values(ascending=False) 

    # Plot
    plt.figure(figsize=fig_size)
    sorted_series.plot(kind='bar', color=sorted_series.map(lambda x: 'g' if x > 0 else 'r'))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('Bar Chart of Values Ranked by Correlation Coefficient')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_correlation_coefficients_heatmap(correlation_dict, correlation_variable, save_path=None,fig_size=(15, 6)):
    """
    Plots a bar heatmap correlation coefficients for different ring sizes

    Parameters:
    - correlation_dict (dict): Dictionary containing correlation data for various ring sizes.
    - save_path (str): Path to save the heatmap image. If None, the heatmap is not saved.
    - correlation_variable (str): The specific variable  to compare across.
    """
    
    
    
    heatmap_data = pd.DataFrame()
    for ring_value, corr_matrix in correlation_dict.items():
        # Extract the row for ['0']_inner_values
        if correlation_variable in corr_matrix.index:
            heatmap_data[ring_value] = corr_matrix.loc[correlation_variable]
    # Transpose the DataFrame to have batches as columns and variables as rows
    heatmap_data = heatmap_data.transpose()
    # Plotting the heatmap
    plt.figure(figsize=fig_size)  # Set the figure size as necessary
    sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap with ['0']_inner_values Across Batches")
    plt.xlabel("Variables")
    plt.ylabel("Ring sizes")
    plt.xticks(rotation=90)  # Rotate variable names for better visibility
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
