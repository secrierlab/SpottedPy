import pandas as pd
import numpy as np
import squidpy as sq
from scipy.stats import pearsonr
from tqdm import tqdm


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

def calculate_neighbourhood_correlation(rings_range,adata_vis,neighbour_variables,source_nodes,
                                        neighbourhood_variable_filter_for_tumour_cells=None,
                                        split_by_batch=False):
    """
    If split_by_batch is True, calculate the correlation between source nodes and neighbouring nodes for each batch. 
    As we take the median of the correlation values, the pvalues are not returned.
    """
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
    
        #split by batch here
        if split_by_batch:
            node_df_cell_abundance['batch'] = adata_vis.obs['batch']
            temp_results={}
            #MAYBE NEED TO FILTER FOR BATCH HERE!!!!!
            for name, group in node_df_cell_abundance.groupby('batch'):
                # Calculate pairwise correlations and p-values within each batch
                # Let's assume you're interested in correlations between all pairs of the first four columns as an example
                corr_matrix = group.corr()
                temp_results[name] = corr_matrix
            average_corrs = pd.concat([r for r in temp_results.values()]).groupby(level=0).median()
            results[rings_for_neighbours_value] = average_corrs
        else:   
            corr_df = node_df_cell_abundance.corr()
            pval = node_df_cell_abundance.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_df.shape)
            results[rings_for_neighbours_value] = corr_df, pval

    return results
    

