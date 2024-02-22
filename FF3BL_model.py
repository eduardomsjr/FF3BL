import pandas as pd
import numpy as np

def get_P_q_omega_FF3(ticker, date, sigma, Er_train, tau, marketweight, size_pred, bm_pred, n_groups):
    """
    This function calculates the modified P, q, and omega for the Fama-French three-factor-based Black-Litterman optimization.

    Args:
        ticker (Numpy array): array of ticker names for portfolio construction.
        date (dict): A dictionary containing the 'train' and 'test' date lists. e.g., {'train':[19570101, ...],'test':[...]}
        sigma (Pandas DataFrame): The covariance matrix of asset returns.
        Er_train (Pandas DataFrame): The asset returns (derived from the Fama-French three-factor model to estimate r hat in paper).
        tau (float): The scaling factor for the uncertainty.
        marketweight (Pandas Series): The market capitalization weights of the assets.
        size (dict): The predicted size of each stock per each year.
            e.g., {'1992': {10065: 612124.2616737402, ... , 25953: 2361325.3544320907},
                   '1993': {10065: 599138.7934181925, ... , 12650: 1005343.9768143898},
                   ...}
        bm (dict): The predicted book-to-market ratio for each stock per each year.
        n_groups (int): the number of groups for the implications of the Fama-French three-factor model (e.g., 5 by 5 sorted matrix) 
        
    Returns:
        P (DataFrame): The modified view matrix. (1 by K)
        q (DataFrame): The modified view return. (1 by 1)
        omega (DataFrame): The modified uncertainty matrix. (1 by 1)
    """ 

    # place holder
    P = pd.DataFrame(np.zeros((sigma.shape[0], 1)), index=sigma.index)
    q = pd.DataFrame(np.zeros((1, 1)))
    omega = np.zeros((1,1))

    #sort by size
    y_pred = size_pred[str(date["test"][0])[0:4]]
    y_pred = dict(y_pred.item())
    y_pred = pd.DataFrame(y_pred, index=[0])
    y_pred = y_pred[ticker]
    sorted_size = y_pred.T.sort_values(by=[0], axis=0)
    
    #then, sort by bm
    y_pred = bm_pred[str(date["test"][0])[0:4]]
    y_pred = dict(y_pred.item())
    y_pred = pd.DataFrame(y_pred, index=[0])
    y_pred = y_pred[ticker]
    sorted_bm = y_pred.T.sort_values(by=[0], axis=0, ascending=False)
    
    def categorize_based_on_rank(series, n_groups):
        group_size = len(series) // n_groups
        groups = {ticker: None for ticker in series.index}
        for i, (ticker, _) in enumerate(series.items()):
            group_number = i // group_size
            if group_number >= n_groups:  # Ensure group number stays within range
                group_number = n_groups - 1
            groups[ticker] = group_number
        return groups
    
    def categorize_into_groups(sorted_size, sorted_bm, n_groups):
        size_groups = categorize_based_on_rank(sorted_size, n_groups)
        combined_groups = {}
        
        size_group = pd.DataFrame(size_groups, index=[0]).T
        all_group_n = list(set(size_group[0]))
        for group_number in all_group_n:
            corresponding_group_n_idx = size_group[size_group == group_number].dropna().index
            tickers_bm = sorted_bm.loc[corresponding_group_n_idx]
            tickers_bm_sorted = tickers_bm.sort_values(by=0, ascending=False)
#             print(tickers_bm_sorted)
            bm_groups = categorize_based_on_rank(tickers_bm_sorted[0], n_groups)
            for ticker, bm_group in bm_groups.items():
                combined_groups[ticker] = group_number * n_groups + bm_group
        return combined_groups


    # Create lists of indices for each of the 5 by 5 groups
    combined_groups = categorize_into_groups(sorted_size[0], sorted_bm, n_groups)
    
    group_indices = [[] for _ in range(n_groups * n_groups)]
    
    for ticker, group_number in combined_groups.items():
        group_indices[group_number].append(ticker)
    # Lists of indices for the 1st and 2nd groups
    good_group = group_indices[0]
    good_len = len(good_group)
    bad_group = group_indices[-1]
    bad_len = len(bad_group)
   
    bad_group_ = pd.Series(np.array([1/(bad_len+1e-20)]*bad_len), index = marketweight[bad_group].index)
    good_group_ = pd.Series(np.array([1/(good_len+1e-20)]*good_len), index = marketweight[good_group].index)
    P.loc[bad_group, 0] = -bad_group_
    P.loc[good_group, 0] = good_group_
       
    P = P.T
    q = pd.Series(np.dot(P, Er_train).flatten())
    omega = tau * np.dot(P @ sigma, P.T)

    matrix_P = np.vstack([P.values])
    matrix_q = np.array([q.values])
    
    return pd.DataFrame(matrix_P), pd.DataFrame(matrix_q), pd.DataFrame(omega)


# input parameters
n_groups = 5
ra = 3.07
tau = 0.1

# get view distribution P, q, and omega

P, q, omega = get_P_q_omega_FF3(ticker, date, sigma_is, Er_train, tau, w_mkt, size_pred, bm_pred, n_groups)

# Based on the calculated above P, q, and omega, we can get the Black-Litterman model weight

## w_BL = Black_Litterman_weight(P, q, ...)