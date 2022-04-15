from tqdm import tqdm

import numpy as np

def att_entropy(att):
    """
    Get entropy of a single attribute
    
    Parameters
    ----------
    att: 1-D series with datatype str
        attribute (column) of the data which entropy is needed
    
    Returns
    ----------
    entropy: float
        single scalar value entropy of the given attribute
    
    """
    
    n_case = len(att)
    
    unique_vals = att.unique()
    
    entropy = 0
    
    for unq in unique_vals:
        
        partition = (att == unq)
        
        P_part = partition.sum() / n_case
        
        entropy += P_part * np.log2(P_part)
    
    entropy = -entropy
    
    return entropy

def cond_entropy(att_x,att_y):
    """
    Get conditional entropy of att_x with respect to att_y
    
    Parameters
    ----------
    att_x: 1-D series with datatype str
        attribute (column) of the data which will be conditioned on
    
    att_y: 1-D series with datatype str
        attribute (column) of the data conditioning on
    
    Returns
    ----------
    cond_entropy: float
        single scalar value conditional entropy of att_x with respect to att_y
    
    """
    
    n_case = len(att_x)
    
    unique_vals_x = att_x.unique()
    unique_vals_y = att_y.unique()
    
    cond_entropy = 0
    
    for unq_y in unique_vals_y:
        partition_y = (att_y == unq_y)
        P_part_y = partition_y.sum() / n_case
        
        cond_P = 0
        
        for unq_x in unique_vals_x:
            
            partition_x = (att_x == unq_x)
            
            if (partition_x * partition_y).sum() != 0:
                x_insc_y = (partition_x * partition_y).sum()
                
                P_part_x_y = x_insc_y  /  partition_y.sum()
            
                cond_P += P_part_x_y * np.log2(P_part_x_y)
                
            else:
                cond_P += 0
            
        
        cond_entropy += P_part_y*cond_P
    
    cond_entropy = -cond_entropy
    
    return cond_entropy

def info_gain(att_x, att_y):
    """
    Get information gain of att_x with respect to att_y
    
    Parameters
    ----------
    att_x: 1-D series with datatype str
        attribute (column) of the data which will be conditioned on
    
    att_y: 1-D series with datatype str
        attribute (column) of the data conditioning on
    
    Returns
    ----------
    ig: float
        single scalar value information gain of att_x with respect to att_y
    
    """
    
    ig =  att_entropy(att_x) - cond_entropy(att_x,att_y)
    
    return ig

def gain_ratio(att_x,att_y):
    """
    Get gain ratio of att_x with respect to att_y
    
    Parameters
    ----------
    att_x: 1-D series with datatype str
        attribute (column) of the data which will be conditioned on
    
    att_y: 1-D series with datatype str
        attribute (column) of the data conditioning on
    
    Returns
    ----------
    gr: float
        single scalar value gain ratio of att_x with respect to att_y
    
    """
    
    gr = info_gain(att_x,att_y) / att_entropy(att_x)
    
    return gr

def mean_gain_ratio(data,att_index):
    """
    Get mean gain ratio of attribute given by attribute index
    
    Parameters
    ----------
    data: pandas DataFrame with datatype str
        data that need to be clustered
    
    att_index: integer
        index of the atribute that mean gain ratio should be calculated
    
    Returns
    ----------
    mgr: float
        mean gain ratio of the attribute
    
    """
    
    n_att = data.shape[1]
    
    att_i = data.iloc[:,att_index]
    
    att_list = list(range(n_att))
    
    att_list.remove(att_index)
    
    mgr = 0
    
    for index in att_list:
        mgr += gain_ratio(att_i,data.iloc[:,index])
    
    mgr = mgr / (n_att - 1)
    
    return mgr

def clust_entropy(cluster):
    """
    Get entropy of given cluster
    
    Parameters
    ----------
    cluster: pandas DataFrame with datatype str
        data from given cluster
    
    Returns
    ----------
    clust_entropy: float
        entropy of given cluster
    
    """
    
    n_att = cluster.shape[1]
    
    clust_entropy = 0
    
    for i in range(n_att):
        clust_entropy += att_entropy(cluster.iloc[:,i])
        
    return clust_entropy

import copy

def MGR(data,k):
    
    """
    Get cluster labels using MGR algorithm
    
    Parameters
    ----------
    data: pandas DataFrame with datatype str
        data that need to be clustered
    k: integer
        desired number of clusters
    
    Returns
    ----------
    clust_entropy: float
        entropy of given cluster
    
    """
    C = list(range(data.shape[0]))
    cnc = 1
    
    cluster_index_dict = dict()

    
    while ((cnc < k) and (C != [])):
        
        remaining_data = copy.deepcopy(data)
        
        remaining_data = remaining_data.iloc[C,:]
        
        att_n_unique = []
    
        for i in range(remaining_data.shape[1]):
            att_n_unique.append(len(remaining_data.iloc[:,i].unique()))
    
        rmv_att = (np.array(att_n_unique) == 1).nonzero()[0]
    
        remaining_data.drop(columns=remaining_data.columns[rmv_att], axis=1, inplace = True)
        
        mgr_list = []
        
        for i in tqdm(range(remaining_data.shape[1])):
            mgr_list.append(mean_gain_ratio(remaining_data,i))
        
        tmp = max(mgr_list)
        
        max_index = mgr_list.index(tmp)
        
        unique_vals = remaining_data.iloc[:,max_index].unique()
        
        partition_dict = dict()
        entropy_dict = dict()
        
        for unq in unique_vals:
            partition_dict[unq] = remaining_data[remaining_data.iloc[:,max_index] == unq]
            entropy_dict[unq] = clust_entropy(partition_dict[unq])
        
        min_key = min(entropy_dict, key=entropy_dict.get)
        
        min_partition = partition_dict[min_key]
            
        cluster_index = list(min_partition.index)
            
        cluster_index_dict[cnc] = cluster_index
        
        print(str(cnc) + 'th cluster assigned')
        
        C = list(set(C) - set(cluster_index))
        
        cnc += 1
        
        if ((cnc == k) and (C != [])):
            
            cluster_index_dict[cnc] = C
            
            break;
        
        
    
    return cluster_index_dict