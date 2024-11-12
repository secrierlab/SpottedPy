import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squidpy as sq
from . import sp_plotting as spl

def correlation_heatmap_neighbourhood(results, variables, save_path=None, pval_cutoff=0.05,fig_size=(5, 5)):
    """
    Plot heatmap from correlation dataframes.

    Parameters:
    - results (tuple): A tuple where the first element is a DataFrame of correlation coefficients 
      and the second is a DataFrame of p-values. This is returned from hotspot.calculafte_neighbourhood_correlation. 
      Select ring number to plot by filtering the dict returned from hotspot.calculate_neighbourhood_correlation.
    - variables (list of str): List of variable names to include in the heatmap.
    - save_path (str, optional): Path to save the heatmap image. If None, the heatmap is not saved.
    - pval_cutoff (float): The p-value cutoff for significance.

    The function plots a heatmap and optionally saves it to a file.
    """
    corr_df=results[0]
    pvalue_df=results[1]
    sub_corr = corr_df.loc[corr_df.index.intersection(variables), :]
    sub_pval = pvalue_df.loc[pvalue_df.index.intersection(variables), :]

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
    sns.set_style("white")
    x_values = list(ring_sensitivity_results1.keys())
    if split_by_batch:
        all_columns = ring_sensitivity_results1[x_values[0]].columns
    else:
        all_columns = ring_sensitivity_results1[x_values[0]][0].columns
    x_indices = range(len(x_values))

    # Figure setup
    n_cols = 5
    n_rows = int(np.ceil(len(all_columns) / n_cols))
    fig_width = fig_size[0]
    fig_height = fig_size[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=True)
    fig.suptitle(f'{correlation_primary_variable} correlation shifts', fontsize=16)

    for idx, column in enumerate(all_columns):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

        # Plotting for first results set
        if label_one is None:
            label_one='Set 1'
        y_values1 = [ring_sensitivity_results1[key][0].loc[correlation_primary_variable, column] for key in x_values] if not split_by_batch else [ring_sensitivity_results1[key].loc[correlation_primary_variable, column] for key in x_values]
        ax.plot(x_indices, y_values1, '-o', color='black', markersize=6, markerfacecolor='red')

        # Optionally plot second results set
        if ring_sensitivity_results2:
            if label_two is None:
                label_two='Set 2'
            y_values2 = [ring_sensitivity_results2[key][0].loc[correlation_primary_variable2, column] for key in x_values] if not split_by_batch else [ring_sensitivity_results2[key].loc[correlation_primary_variable2, column] for key in x_values]
            ax.plot(x_indices, y_values2, '-o', color='black', markersize=6, markerfacecolor='blue')

        ax.set_title(column, fontsize=12)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_values, rotation=45, fontsize=10)
        ax.set_xlabel("Number of rings", fontsize=12)
        ax.set_ylabel("Correlation", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.legend()

    # Remove any extra subplots
    #if len(all_columns) % n_cols != 0:
    #    for j in range(len(all_columns) % n_cols, n_cols):
    #        fig.delaxes(axes[n_rows - 1, j])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

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
    differences = []
    x_values = list(ring_sensitivity_results.keys())
    if split_by_batch:
        all_columns = ring_sensitivity_results[x_values[0]].columns
    else:
        all_columns = ring_sensitivity_results[x_values[0]][0].columns
    
    for column in all_columns:
        if split_by_batch:
            first_value = ring_sensitivity_results[x_values[0]].loc[correlation_primary_variable, column]
            last_value = ring_sensitivity_results[x_values[-1]].loc[correlation_primary_variable, column]
        else:
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
    #rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
