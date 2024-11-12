import scanpy as sq
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import squidpy as sq


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
                inner_ring_means.append(inner_mean)
                #empty neighbouring nodes
                outer_nodes = get_outer_ring_nodes(node_name, neighbors_matrix_outer, [])
                outer_ring_means.append(inner_mean)
        # Convert lists to DataFrames
        inner_df_1 = pd.DataFrame(inner_ring_means, index=source_nodes, columns=neighbour_variables.columns)
        outer_df_1 = pd.DataFrame(outer_ring_means, index=source_nodes, columns=neighbour_variables.columns)
        inner_outer_results[ring_range] = inner_df_1,outer_df_1

    return inner_outer_results

def calculate_corr_pvalue_for_inner_outer_neighbourhood_enrichment(results,correlation_key_variable,rings_range,average_by_batch=False,adata_vis=None,source_nodes=None):
    corr_pval_results={}
    for ring in rings_range:
        inner_df,outer_df=results[ring]
        outer_df[f'{correlation_key_variable}_inner_values']=inner_df[correlation_key_variable]

        if average_by_batch:
            #if index contains 'Cancer' or 'Luminal' or 'Myoepithelial' remove from corr_df            
            outer_df['batch'] = adata_vis.obs['batch']
            temp_results={}
            for name, group in outer_df.groupby('batch'):
                # Calculate pairwise correlations and p-values within each batch
                # Let's assume you're interested in correlations between all pairs of the first four columns as an example
                corr_matrix = group.corr()
                temp_results[name] = corr_matrix
            average_corrs = pd.concat([r for r in temp_results.values()]).groupby(level=0).median()
            
            corr_pval_results[ring] = average_corrs

            
        else:
            inner_df,outer_df=results[ring]
            outer_df[f'{correlation_key_variable}_inner_values']=inner_df[correlation_key_variable]
            corr_df=outer_df.corr()
            #if index contains 'Cancer' or 'Luminal' or 'Myoepithelial' remove from corr_df
            
            corr_df=outer_df.corr()
            p_val=outer_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_df.shape)
            corr_pval_results[ring] = corr_df, p_val
    return corr_pval_results
