import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squidpy as sq
import sp_plotting as spl
import joypy
from matplotlib import colors as mcolors
from anndata import AnnData
from typing import Optional
import scanpy as sc

def plot_distance_distributions_across_batches(df, comparison_variable,fig_size,fig_name=None):
    df = df[df['comparison_variable'] == comparison_variable]
    primary_variables = df['primary_variable'].unique()
    df_merged = pd.DataFrame()
    for variable in primary_variables:
        df_filtered = df[df['primary_variable'] == variable]
        for index, row in df_filtered.iterrows():
            if variable == primary_variables[0]: 
                new_row = {f'min_distances_{primary_variables[0]}_hot': row['min_distance'], 'batch': row['batch'], f'min_distances_{primary_variables[1]}_hot': np.nan}
            else:  
                new_row = {f'min_distances_{primary_variables[0]}_hot': np.nan, 'batch': row['batch'], f'min_distances_{primary_variables[1]}_hot': row['min_distance']}
            df_merged = pd.concat([df_merged, pd.DataFrame([new_row])], ignore_index=True)
    plt.figure(figsize=fig_size)
    
    joypy.joyplot(df_merged, by='batch', legend=True)
    if fig_name:
        plt.savefig(fig_name)
    plt.show()


def plot_distance_distributions_across_hotspots(df, comparison_variable,batch,fig_size=(3, 5),fig_name=None):  
    df=df[df['comparison_variable']==comparison_variable]
    df=df[df['batch']==batch]
    df['hotspot_suffix'] = df['hotspot_number'].apply(lambda x: x.split('_')[-1])

    # Sort DataFrame by the helper column and then by hotspot_number
    df = df.sort_values(by=['hotspot_suffix', 'hotspot_number'])# Get unique primary variables and create a color palette
    unique_primary_variable = df['primary_variable'].unique()
    palette = sns.color_palette("hsv", len(unique_primary_variable))
    color_dict = dict(zip(unique_primary_variable, palette))

    # Map colors to hotspot numbers based on primary variables
    hotspot_color_map = df.drop_duplicates('hotspot_number').set_index('hotspot_number')['primary_variable'].map(color_dict).to_dict()

    # Ensure the colors are mapped in the correct order after sorting
    colors = [hotspot_color_map[x] for x in df['hotspot_number'].unique()]

    # Create joyplot with colored distributions
    fig, axes = joypy.joyplot(
        df.groupby('hotspot_number',sort=False),
        column="min_distance",
        legend=True,
        color=colors
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(palette), norm=plt.Normalize(vmin=0, vmax=len(unique_primary_variable) - 1))
    colorbar_axes = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Adjust the position and size of the colorbar
    cbar = fig.colorbar(sm, cax=colorbar_axes)
    cbar.set_ticks(np.linspace(0, 1, len(unique_primary_variable)))
    cbar.ax.set_yticklabels(unique_primary_variable)
    cbar.ax.tick_params(labelsize=10)
    fig.set_size_inches(*fig_size)
    if fig_name:
        plt.savefig(fig_name)
    plt.show()


def plot_hotspots_by_number(
    anndata: AnnData,
    column_name: str,
    batch_single: Optional[str] = None,
    save_path: Optional[str] = None,
    color_for_spots: str = 'Reds_r'
) -> None:

    if batch_single is not None:
        data_subset = anndata[anndata.obs['batch'] == str(batch_single)].copy()
        sc.pl.spatial(data_subset, color=[column_name],  vmax='p0',color_map=color_for_spots, library_id=str(batch_single),save=f"_{save_path}",colorbar_loc=None,alpha_img= 0.5)
    else:
        for batch in anndata.obs['batch'].unique():
            data_subset = anndata[anndata.obs['batch'] == str(batch)].copy()
            sc.pl.spatial(data_subset, color=[column_name],  vmax='p0',color_map=color_for_spots, library_id=str(batch),save=f"_{str(batch)}_{save_path}",colorbar_loc=None,alpha_img= 0.5)
