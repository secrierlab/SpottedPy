import scanpy as sc
import pandas as pd
import numpy as np
import squidpy as sq

def process_tumor_perimeter(data):
    unique_batches = data.obs['batch'].unique()
    for batch_id in unique_batches:
        # Filter the data for the specified batch
        adata_scores_filtered = data[data.obs['batch'] == batch_id]
        # Compute spatial neighbors
        sq.gr.spatial_neighbors(adata_scores_filtered, n_rings=1, coord_type="grid")
        # Filter to only include tumour cells
        tumour_filtered = adata_scores_filtered[adata_scores_filtered.obs['tumour_cells'] == 1]
        # Create connectivity matrix
        connectivity_matrix = pd.DataFrame.sparse.from_spmatrix(adata_scores_filtered.obsp['spatial_connectivities'])
        connectivity_matrix.index = adata_scores_filtered.obs.index
        connectivity_matrix.columns = adata_scores_filtered.obs.index
        # Filter connectivity matrix rows to tumour cells
        connectivity_matrix_tumour = connectivity_matrix.loc[tumour_filtered.obs.index, :]
        # Identify tumour perimeter cells
        tumour_perimeters = []
        for index, row in connectivity_matrix_tumour.iterrows():
            for col_name, value in row.items():
                if value == 1 and col_name not in connectivity_matrix_tumour.index:
                    tumour_perimeters.append(col_name)
        # Update the original data with perimeter information
        data.obs.loc[data.obs['batch'] == batch_id, 'tumour_perimeter'] = data.obs.loc[data.obs['batch'] == batch_id].index.isin(tumour_perimeters)
    # Replace True/False with 'Yes'/NaN for the entire dataset
    data.obs['tumour_perimeter'] = data.obs['tumour_perimeter'].replace({True: 'Yes', False: np.NaN})
    return data