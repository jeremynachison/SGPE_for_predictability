#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:09:37 2024

@author: jeremynachison
"""
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd
import ordpy

def time_embedding_helper(embedding, dt, taut, ds, tshifts, i):
    """
    Helper function when t_type="2d", gets right indices depending if taut is negative or positive
    """
    tau_index = tshifts[i]
    chunk_index = ds + i*ds
    # i.e. construct embeddings by looking into the future
    if taut < 0:
        embedding[:tau_index, chunk_index: chunk_index + ds] = embedding[-tau_index:, 0:ds]
    # i.e. construct embeddings by looking into the past
    if taut > 0:
        embedding[tau_index:, chunk_index: chunk_index + ds] = embedding[:-tau_index, 0:ds]
    return embedding
    
def make_embedding(poi, case_data, flow_data, ds, dt, taus, taut, 
                   asc=True, t_type ="1d", remove_na = True):
    """
    For the given poi, construct the embedding matrix used to calculate SGPE. 
    The embedding matrix has ds+dt-1 (if t_type=1d) or ds*dt (if t_type=2d) 
    columns and the same number of rows as case_data. Row i of the embedding 
    matrix holds the embedding vector for the poi at time point i. The 
    embedding vector here may include both temporal and spatial information.
    
    t_type (str): Either "1d" or "2d". 
        If "1d", then row i in the embedding matrix will contain the case data 
        at time i for the poi and ds-1 of its nearest neighbors according to 
        flow_data, and also 
        
        If "2d", then row i of embedding matrix contains the case data for the poi
        and ds-1 of its nearest neighbors at time i, as well as the case data 
        for the poi and its ds-1 nearest neighbors at time i+taut, i+2*taut,...,i+(dt-1)*taut
    
    remove_na (bool): indicates if NA values (which always occur when dt>1) 
        should be removed from the returned matrix
    """
    # for 1d t_type, embedding vectors have length ds+dt-1
    if t_type == "1d":
        embedding = np.full((len(case_data.index), ds+dt-1), np.nan)
    # for 2d t_type embedding vectors have length ds*dt 
    if t_type == "2d":
        embedding = np.full((len(case_data.index), ds*dt), np.nan)
    # the point of interest is the the only location when ds=dt=1
    embedding[:,0] = case_data[poi]
    # assemble additional spatial embedding (if ds >1)
    if ds>1:
        # get top ds*taus locations with highest flow into the poi location
        flow_sorted = (flow_data.sort_values(['origin', flow_data.columns[2]], ascending=asc).
                       groupby('origin').head(ds*taus))
        top_flows = flow_sorted.loc[flow_sorted['origin'] == poi, 'destination'].tolist()
        # extract only the top flows according to the spatial separation
        top_flows_tau = top_flows[taus-1::taus][:ds-1]
        embedding[:,1:ds] = np.array(case_data[top_flows_tau])
    # for sptio-temporal embeddings, can add temporal data from the future/past
    if (dt > 1 and t_type == "1d"):
        # for shift method (or np.roll), need opposite sign of taut (i.e. negative to see future, positive for past)
        taut *= -1 
        tshifts = np.arange(taut, dt*taut,taut)
        embedding[:,ds:] = np.column_stack([case_data[poi].shift(s) for s in tshifts])
    if (dt > 1 and t_type == "2d"):
        taut *= -1 
        tshifts = np.arange(taut, dt*taut,taut)
        for i in range(dt-1):
            embedding = time_embedding_helper(embedding, dt, taut, ds, tshifts, i)
    if remove_na:
        # Remove NAs introduced from temporal shifts
        embedding = embedding[~np.isnan(embedding).any(axis=1)]
    return embedding

def calculate(poi, case_data, flow_data, ds=1, dt=1, taus=1, taut=3, 
         normalize=True, embedding=False, t_type = "1d", remove_na=True):
    """
    Calculates the SGPE for a location with specified temporal and spatial 
    embedding vectors.
    
    poi (str): column name of case_data, a location to calculate SGPE over
    
    case_data (pd.DataFrame): a dataframe where each column is a time series 
        for a disease case counts in a given location. 
        
        FORMAT OF case_data
        - Columns: Each column corresponds to a specific location (identified 
            by FIPS code, name etc..) representing the area where the disease 
        outbreak is being monitored.
        - Rows: Each row corresponds to an evenly spaced time index,
            representing the progression of time across each column in the dataset.
        - Values (float): Each cell contains the recorded number of disease 
            cases at the specified location (column) and time index (row).
        
    flow_data (pd.DataFrame): A dataframe with 3 columns, each row holds a 
        unique pair of locations present in case_data, and the spatial ordering 
        metric beyween these locations (i.e distance, flow, etc.)
        
        FORMAT OF flow_data
        - Column 1: Contains the name (or identifying code) of each location in 
            case_data. If there are N locations in case_data, each location is 
            repeated N-1 times to hold the data for all of the respective 
            destinations in column 2 (-1 because do not include self loop)
        - Column 2: Contains the names (or identifying codes) of each location 
            in case_data, representing the destination locsation for each 
            respective origin locationin column 1.
        - Column 3 ("distance", "flow", etc.): Holds the spatial metric (like 
            distance) between the location in column 1 and the location in 
            column 2
    
    dt (int): temporal embedding dimension, gretaer than or equal to 1. If only 
    interested in spatial PE, set to 1.
    
    ds (int): spatial embedding dimension, greater than or equal to 1. If only 
    interested in temporal PE, set to 1.
    
    taut (int): temporal delay.
    
    taus (int): spatial delay
    
    normalize (bool): indicates if permutation entropy should be normalized to 
        be between 0 and 1
        
    embedding (bool): If true, indicates the passed poi is already an embedding 
        matrix. If false, the embedding matrix for the poi column in case_data 
        must be constructed.
        
    t_typr (str): either "1d" or "2d". 
        If "1d", then the temporal embedding will only use the next dt-1 data 
        points for poi in the embedding vector (no future data points for any 
        neighbors will be included). This is the "tripod" embedding pattern
        
        If "2d", then the next dt-1 data points for the poi and its ds-1 
        neighbors will also be included in the embedding vector. This is the 
        "cross" embedding pattern
    """
    if not embedding:
        embedding = make_embedding(poi, case_data, flow_data, ds, dt, taus, taut, 
                                   t_type = t_type, remove_na=remove_na)
    else:
        embedding = poi
    if t_type == "2d":
        embed_dim = ds*dt
    if t_type== "1d":
        embed_dim = ds+dt-1
    entropy = ordpy.permutation_entropy(embedding, dx=embed_dim, dy=1, normalized=normalize)
    return entropy

############################ Accessory Functions #############################

def get_window_entropy(arr3d, arg_dict, normalize):
    """
    Helper function for rolling_window. For a collection of 2d embeddings for 
    each rolling window (stored a s a 3d array), get the SGPE for each 2d embedding. 
    """
    
    flattened_arr = arr3d.reshape(arr3d.shape[0],-1)
    window_entropy = np.apply_along_axis(lambda x: calculate(x.reshape((arr3d.shape[1], arr3d.shape[2])), 
                                                             embedding=True, normalize=normalize, **arg_dict),
                                         1, flattened_arr)
    return window_entropy

def rolling_window(window_size, case_data, flow_data, ds=1,dt=1,taus=1,taut=1,
                   t_type="1d", normalize=True):
    """
    Get the SGPE for each location in case_data across rolling window of length window_size
    
    window_size (int): the length (in time) of each window. 
        i.e if window_size=5, the SGPE will be calculated for time points 1-5, then for timepoints 2-6, 3-7 etc...
    
    All other args are the same as calculate, and dictate how SGPE will be calculated for each window
    """
    arg_dict = {"case_data":case_data, "flow_data":flow_data, "ds":ds,"dt":dt, "taus":taus, 
                "taut":taut, "t_type":t_type}
    # make placeholder series for data
    embedding_series = pd.Series([np.nan]*case_data.shape[1], 
                                 index=np.arange(case_data.shape[1]), dtype=object)
    # 1d series holding 2d embedding matrices, ith entry holds the embedding for location i
    embedding_series = case_data.apply(lambda x: [make_embedding(x.name, remove_na=False, **arg_dict)])
    # series of 3d arrays, ith entry has shape (nrow(case_data) - window_size +1, window_size, embedding dimension)
    # 2d slices when first axis=i holds the embedding matrix for the ith moving window
    windows = embedding_series.apply(lambda embedding: [np.squeeze(sliding_window_view(embedding[0], 
                                             window_shape=(window_size, embedding[0].shape[1])))])
    # for each 3d array of embedding matrices, get the entropy for each window
    window_entropy = windows.apply(lambda x: get_window_entropy(x[0], arg_dict, normalize))
    # get same index as original data
    window_entropy.index = case_data.index[window_size-1:]
    return window_entropy


def shuffle(flow_data, alpha, sort = True):
    """
    Shuffles a subset of the flows (third column in flow_data) to disrupt 
    spatial alignment. 
    
    flow_data (pd.DataFrame): Flow dataframe, see arguments of calculate for details
    
    alpha (float): Between 0 and 1, specifies the percent of the data to shuffle. 
        When alpha=1, the entire dataset is shuffled
        
    sort (bool): Specifies if the returned dataset is sorted by fips code in ascending order or not
    """
    # since flow A -> B the same as flow B -> A, flow_data has duplicate info, make data with only unique A,B flows
    unique_flows = (flow_data[flow_data["origin"].astype(float) < flow_data["destination"].astype(float)]
                    .reset_index(drop=True))
    # get number of rows to shuffle
    num_shuffled = round(alpha*len(unique_flows))
    # randomly select num_shuffled number of indices from the unique_flows
    shuffled_indices = np.random.choice(unique_flows.index, size=num_shuffled, replace=False)
    # get subset to shuffle
    subset_df = unique_flows.loc[shuffled_indices]
    # shuffle the flows in the selected indices
    subset_df.iloc[:, 2] = np.random.permutation(subset_df.iloc[:, 2])
    # replace the flows with their shuffled values for the selected indices
    unique_flows.iloc[shuffled_indices,2] = subset_df.iloc[:, 2]
    # Now recover original dataset structure that includes duplicate flows
    swapped_flows = unique_flows.rename(columns={"origin":"destination", "destination":"origin"})
    original_but_shuffled = pd.concat([unique_flows, swapped_flows], ignore_index=True)
    # Note: rows will be in different order than flow_data, if data is sorted can recover with sorting
    if sort:
        original_but_shuffled = original_but_shuffled.sort_values(by=['origin', 'destination']).reset_index(drop=True)
    return original_but_shuffled
    

