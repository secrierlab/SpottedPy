from .main import create_hotspots, calculateDistances
from . import sp_plotting as spl
import pandas as pd
import numpy as np
import scanpy as sc



def process_hotspots(adata, variable_name, parameter_size, sensitivity_parameter):
    if sensitivity_parameter=="pvalue":
        adata_hotspots = create_hotspots(
            adata, column_name=variable_name,
            neighbours_parameters=10,
            p_value=parameter_size,
            number_components_return=False,
            number_hotspots=True
        )
    
    if sensitivity_parameter=="neighbourhood":
        adata_hotspots = create_hotspots(
            adata, column_name=variable_name,
            neighbours_parameters=parameter_size,
            p_value=0.05,
            number_components_return=False,
            number_hotspots=True
        )
    return adata_hotspots

#helper function
def calculate_distances(adata_hotspots, variables):
    return calculateDistances(adata_hotspots, variables,hotspot_number=True)

#helper function
def process_batches(spatial_anndata_obj, params):
    distances_df_sensitivity = pd.DataFrame()
    #add empty column to spatial_anndata_obj
    for col in [params['variable_comparison'], params['variable_one'], params['variable_two'],
            params['variable_comparison']+"_number", params['variable_one']+"_number", params['variable_two']+"_number"]:
        spatial_anndata_obj.obs[col] = np.nan
    for batch in spatial_anndata_obj.obs['batch'].unique():
        adata_batch = spatial_anndata_obj[spatial_anndata_obj.obs['batch'] == batch]
        tumour_filtered = adata_batch[adata_batch.obs['tumour_cells'] == 1]

        # Processing for variable_comparison
        adata_hotspots = process_hotspots(adata_batch if not params['variable_comparison_tumour'] else tumour_filtered, 
                                          params['variable_comparison'].rsplit('_', 1)[0], 
                                          params['parameter_comparison_variable_neighbourhood'],params['sensitivity_parameter'])
        spatial_anndata_obj=spl.add_hotspots_to_fullanndata(spatial_anndata_obj,params['variable_comparison'].rsplit('_', 1)[0],batch,adata_hotspots)

        # Processing for variable_one and variable_two
        for variable in [params['variable_one'], params['variable_two']]:
            adata_hotspots = process_hotspots(tumour_filtered if params['variable_one_two_is_tumour'] else adata_batch, 
                                              variable.rsplit('_', 1)[0], 
                                              params['parameter_variables_neighbourhood'],params['sensitivity_parameter'])
            spatial_anndata_obj=spl.add_hotspots_to_fullanndata(spatial_anndata_obj,variable.rsplit('_', 1)[0],batch,adata_hotspots)
        
        #select tumour_cell column in spatial_anndata 
        spatial_anndata_obj.obs.loc[
            (spatial_anndata_obj.obs['batch'] == batch) & (spatial_anndata_obj.obs['tumour_cells'] == 1), 
            'tumour_cells_number'
        ] = batch


        # Calculate distances
        #filter spatial_anndata to only include batch
        #spatial_anndata_batch=spatial_anndata[spatial_anndata.obs['batch']==batch]
        #sc.pl.spatial(spatial_anndata_batch,color=params['variable_one'],library_id=str(batch))
        #sc.pl.spatial(spatial_anndata_batch,color=params['variable_two'],library_id=str(batch))
        #sc.pl.spatial(spatial_anndata_batch,color=params['variable_three'],library_id=str(batch))
    distances_batch = calculate_distances(spatial_anndata_obj, [params['variable_one'], params['variable_two'],params['variable_comparison'],params['variable_three']])
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
        ].groupby(['batch','hotspot_number'])['min_distance'].median()
        distance_variable_two = distances_df_sensitivity[
            (distances_df_sensitivity['comparison_variable'] == params['variable_comparison']) & 
            (distances_df_sensitivity['primary_variable'] == params['variable_two'])
            ].groupby(['batch','hotspot_number'])['min_distance'].median()
        #reference value ie tumour cells 
        distance_variable_three = distances_df_sensitivity[
            (distances_df_sensitivity['comparison_variable'] == params['variable_comparison']) & 
            (distances_df_sensitivity['primary_variable'] == params['variable_three'])
        ].groupby(['batch','hotspot_number'])['min_distance'].median()  
        #here we want to look at MIN, MEDIAN, MEAN of EACH hotspot within variable of interest, group by hotspot ID in the functio above
        #paired_df = pd.DataFrame({'variable_one': distance_variable_one, 'variable_two': distance_variable_two,'variable_three':distance_variable_three }).dropna()
        results["distance_variable_one"].append(distance_variable_one.median())
        results["distance_variable_two"].append(distance_variable_two.median())
        results["distance_variable_three"].append(distance_variable_three.median())
    spl.plot_sensitivity(params['values_to_test'], results, params['variable_comparison'],
                     params['variable_one'], params['variable_two'], params['variable_three'], params['save_path'],params['sensitivity_parameter'])

    return results