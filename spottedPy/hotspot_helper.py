import numpy as np
import pandas as pd
import scipy.sparse
import squidpy as sq
from anndata import AnnData
from typing import List, Tuple, Union
from scipy.sparse.csgraph import connected_components
from esda import G_Local
import libpysal
import scanpy as sc
from statsmodels.stats.multitest import multipletests as fdr
from scipy.spatial import distance_matrix

def find_connected_components(
    hotspot: pd.DataFrame,
    slides: AnnData
) -> Tuple[pd.DataFrame, int]:
    anndata_scores_filtered_high_hotspot = slides[slides.obs.index.isin(hotspot.index)]
    row_count = anndata_scores_filtered_high_hotspot.obs.index.shape[0]
    if row_count < 8:
        neighbour_no = row_count - 1
    else:
        neighbour_no = 6
    sq.gr.spatial_neighbors(anndata_scores_filtered_high_hotspot, n_rings=1, coord_type="grid", n_neighs=neighbour_no)
    connectivity_matrix = pd.DataFrame.sparse.from_spmatrix(anndata_scores_filtered_high_hotspot.obsp['spatial_connectivities'])
    connectivity_matrix.index = anndata_scores_filtered_high_hotspot.obs.index
    connectivity_matrix.columns = anndata_scores_filtered_high_hotspot.obs.index
    connectivity_matrix_sparse = scipy.sparse.csr_matrix(connectivity_matrix.values)
    n_components, labels = connected_components(csgraph=connectivity_matrix_sparse, directed=False, return_labels=True)
    hotspot = hotspot.copy()
    hotspot['hotspot_label'] = labels
    #append batch to each hotspot label
    hotspot['hotspot_label'] = hotspot['hotspot_label'].astype(str) + "_" + hotspot['batch'].astype(str)
    hotspot = hotspot[~hotspot['hotspot_label'].isin(hotspot['hotspot_label'].value_counts()[hotspot['hotspot_label'].value_counts() < 5].index)]

    return hotspot, n_components


def calculate_hotspots_with_hotspots_numbered(
    anndata_filtered: AnnData,
    significance_level: float = 0.05,
    score_column: str = 'scores',
    neighbours_param: int = 8,
    return_number_components: bool = False,
    hotspots_relative_to_batch: bool = True,
    add_hotspot_numbers: bool = False,
    permutation: int = 999,
    seed_number: int = 100
) -> Union['AnnData', Tuple[List[int], List[int], 'AnnData']]:

    n_components_high_list = []
    n_components_low_list = []
    anndata_filtered=anndata_filtered.copy()
    anndata_filtered.obs[score_column + "_hot"] = np.nan
    anndata_filtered.obs[score_column + "_cold"] = np.nan
    anndata_filtered.obs[score_column + "_hot_number"]=""
    anndata_filtered.obs[score_column + "_cold_number"]=""

    #convert score_column to float64 in anndata_filtered
    anndata_filtered.obs[score_column] = anndata_filtered.obs[score_column].astype('float64')



    #compute hotspots relative to batch
    if hotspots_relative_to_batch:
        for batch in anndata_filtered.obs['batch'].unique():
            score_df = anndata_filtered[anndata_filtered.obs['batch'] == str(batch)].obs
            score_df = score_df[~pd.isna(score_df[score_column])]
            
            pp = list(zip(score_df['array_row'], score_df['array_col']))
            kd = libpysal.cg.KDTree(np.array(pp))
            wnn2 = libpysal.weights.KNN(kd, neighbours_param)
            y = score_df[score_column]
            lg = G_Local(y, wnn2, seed=seed_number, permutations=permutation)
            #lg = G_Local(y, wnn2, seed=200)

            # strict fdr correction; removes majority of hotspots that visually are present
            #significance_level=fdr(lg.p_sim,significance_level)

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

                #only add numbers if there are "hotspot_label" columns
                if "hotspot_label" in high_hotspot.columns:
                    anndata_filtered.obs.loc[high_hotspot.index, score_column + "_hot_number"] = high_hotspot['hotspot_label']
                if "hotspot_label" in low_hotspot.columns:
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
        anndata_filtered.obs.loc[high_hotspot.index, score_column + "_hot_"+batch] = high_hotspot[score_column]
        anndata_filtered.obs.loc[low_hotspot.index, score_column + "_cold_"+batch] = low_hotspot[score_column]
        if return_number_components:
            n_components_low_list.append(n_components_low)
            n_components_high_list.append(n_components_high)
        
    if return_number_components:
        return n_components_low_list, n_components_high_list, anndata_filtered
    else:
        return anndata_filtered
    
def create_hotspots(anndata, column_name, filter_columns, filter_value, neighbours_parameters, p_value, number_components_return, relative_to_batch, number_hotspots, permutation, seed_number):
    anndata=anndata.copy()
    if filter_columns is not None:
        anndata_filtered = anndata[anndata.obs[filter_columns] == filter_value]
    else:
        anndata_filtered = anndata
    if any(anndata_filtered.obs[column_name] < 0):
        raise ValueError("Score values must be in the range 0 to 1. Ensure there are no negative values in the score column.")
    if number_hotspots or number_components_return:
        n_components_low, n_components_high, anndata_filtered = calculate_hotspots_with_hotspots_numbered(
            anndata_filtered,
            significance_level=p_value,
            score_column=column_name,
            neighbours_param=neighbours_parameters,
            return_number_components=True,
            hotspots_relative_to_batch=relative_to_batch,
            add_hotspot_numbers=number_hotspots,
            permutation=permutation,
            seed_number=seed_number
        )
        #fill with nans
        anndata.obs[column_name + "_hot"]=np.nan
        anndata.obs[column_name + "_cold"]=np.nan
        anndata.obs[column_name + "_hot_number"]=""
        anndata.obs[column_name + "_cold_number"]=""
        anndata.obs.loc[anndata_filtered.obs.index, column_name + "_hot"] = anndata_filtered.obs[column_name + "_hot"]
        anndata.obs.loc[anndata_filtered.obs.index, column_name + "_cold"] = anndata_filtered.obs[column_name + "_cold"]
        col_name=column_name + "_hot_number"
        if col_name in anndata_filtered.obs:
            anndata.obs.loc[anndata_filtered.obs.index, col_name] = anndata_filtered.obs[col_name]
            anndata.obs[col_name] = anndata.obs[col_name].astype(str)

            #add column_name to end of all values in anndata.obs[column_name + "_hot_number"] that are not nan
            anndata.obs[col_name] = anndata.obs[col_name].apply(lambda x: x + "_" + column_name if x != "nan" else np.nan)
        col_name=column_name + "_cold_number"
        if col_name in anndata_filtered.obs:
            anndata.obs.loc[anndata_filtered.obs.index, column_name + "_cold_number"] = anndata_filtered.obs[col_name]
            anndata.obs[col_name] = anndata.obs[col_name].astype(str)
            #add column_name to end of all values in anndata.obs[column_name + "_cold_number"] that are not nan
            anndata.obs[col_name] = anndata.obs[col_name].apply(lambda x: x + "_" + column_name if x != "nan" else np.nan)
        if number_components_return:
            return n_components_low, n_components_high, anndata
        else:
            return anndata
    
    else:
        anndata_filtered = calculate_hotspots_with_hotspots_numbered(
            anndata_filtered,
            significance_level=p_value,
            score_column=column_name,
            neighbours_param=neighbours_parameters,
            hotspots_relative_to_batch=relative_to_batch,
            add_hotspot_numbers=number_hotspots,
            permutation=permutation,
            seed_number=seed_number
        )
        anndata.obs.loc[anndata_filtered.obs.index, column_name + "_hot"] = anndata_filtered.obs[column_name + "_hot"]
        anndata.obs.loc[anndata_filtered.obs.index, column_name + "_cold"] = anndata_filtered.obs[column_name + "_cold"]

    return anndata

# note: here he filter out slides less than 30 spots, but this can be changed 
def calculateDistances(anndata, primary_variables, comparison_variables,split_by_slide_in_batch,empty_hotspot_default_to_max_distance,hotspot_number):
    if comparison_variables is None:
        comparison_variables = primary_variables
    #if comparison_variables is one string, convert to list
    if isinstance(comparison_variables, str):
        comparison_variables=[comparison_variables]
    distances_df_all = pd.DataFrame()
    # Loop through each unique batch
    for batch in anndata.obs['batch'].unique():
        batch_data = anndata[anndata.obs['batch'] == batch]
        if split_by_slide_in_batch:
            #get connected components for batch
            sq.gr.spatial_neighbors(batch_data, n_rings=1, coord_type="grid", n_neighs=6)
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
                #consider slide too small to calculate distances
                if batch_data_label.shape[0]<30:
                    continue
                distance_df=calculateDistancesHelper(batch_data_label, primary_variables, comparison_variables,batch,empty_hotspot_default_to_max_distance,hotspot_number_bool=hotspot_number)
                distances_df_all = pd.concat([distances_df_all, distance_df])
        else:
            distance_df=calculateDistancesHelper(batch_data, primary_variables, comparison_variables,batch,empty_hotspot_default_to_max_distance,hotspot_number_bool=hotspot_number)
            distances_df_all = pd.concat([distances_df_all, distance_df])

    return distances_df_all


def calculateDistancesHelper(batch_adata,primary_variables, comparison_variables,batch,empty_hotspot_default_to_max_distance,hotspot_number_bool):
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
        if hotspot_number_bool:
            #remove string after "_" in primary variable
            column_name = f"{primary_var}_number"
            if column_name not in batch_adata.obs.columns:
                raise ValueError(f"The column {column_name} does not exist in the data. Please run create_hotspots() for primary val with number_hotspots=True.")

            hotspot_labels=batch_adata[batch_adata.obs[primary_var].notnull()].obs[column_name]
            
        for comparison_var in comparison_variables:
            # Extract the points for the primary and comparison variables
            comparison_points = batch_adata[batch_adata.obs[comparison_var].notnull()].obs[['array_row', 'array_col']]


            # Calculate the distance matrix between the primary and comparison points
            dist_matrix = distance_matrix(primary_points, comparison_points)

            # Check if the distance matrix is empty
            if dist_matrix.size > 0:
                # Calculate the minimum distance for each primary point to the comparison points
                min_distances = np.amin(dist_matrix, axis=1)
                if hotspot_number_bool:
                    temp_df = pd.DataFrame({
                        'min_distance': min_distances,
                        'primary_variable': primary_var,
                        'comparison_variable': comparison_var,
                        'primary_index': primary_points.index,
                        'batch': batch,
                        'hotspot_number': hotspot_labels.values})
                else:
                    # Create a DataFrame to store the results
                    temp_df = pd.DataFrame({
                        'min_distance': min_distances,
                        'primary_variable': primary_var,
                        'comparison_variable': comparison_var,
                        'primary_index': primary_points.index,
                        'batch': batch})
                # Concatenate the results DataFrame with the overall DataFrame
                distances_per_batch = pd.concat([distances_per_batch, temp_df])
            else:
                # Print a warning message if the distance matrix is empty
                #if slide_distances.size > 0 then there are primary points, so add max_distance_of_slide to distances_per_batch
                if empty_hotspot_default_to_max_distance:
                    if slide_distances.size > 0:
                        #print(f"Warning: Empty distance matrix for batch = {batch}, primary_var = {primary_var}, comparison_var = {comparison_var}. Therefore, no hotspots calculated in slide for {comparison_var}")

                        if hotspot_number_bool:
                            temp_df = pd.DataFrame({'min_distance': [max_distance_of_slide] * len(primary_points),
                                                'primary_variable': [primary_var] * len(primary_points),
                                                'comparison_variable': [comparison_var] * len(primary_points),
                                                'primary_index': primary_points.index,
                                                'batch': [batch] * len(primary_points),
                                                'hotspot_number':hotspot_labels.values})
                        else:
                        #if primary variables exist, and no comparison value for that variable, make distance max length of slide                 
                            temp_df = pd.DataFrame({'min_distance': [max_distance_of_slide] * len(primary_points),
                                                    'primary_variable': [primary_var] * len(primary_points),
                                                    'comparison_variable': [comparison_var] * len(primary_points),
                                                    'primary_index': primary_points.index,
                                                    'batch': [batch] * len(primary_points)})
                        # Concatenate the results DataFrame with the overall DataFrame                
                        distances_per_batch = pd.concat([distances_per_batch, temp_df])
                     
    return distances_per_batch

def score_genes_by_batch(adata, gene_lists, gene_list_names):
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


def add_genes_to_obs(adata, gene_list):
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
