import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import permutation_test_score, GridSearchCV, GroupKFold, StratifiedKFold

def str_to_list(x):
    """
    Convert string to list.
    """
    return [float(i) for i in x.strip('][').split(',')]

def add_indicator_columns(df, quantiles=[0.5, 0.75]):
    """
    Add indicator columns for each column in the DataFrame.
    - Add above median indicator column
    - Add above 75th percentile indicator column
    """
    new_df = df.copy()
    
    for col in df.columns:
        for quantile in quantiles:
            if quantile == 0.5:
                # Add above median indicator
                value = df[col].median()
                new_col_name = f"{col}_above_median"
            else:
                # Add above 75th percentile indicator
                value = df[col].quantile(quantile)
                new_col_name = f"{col}_above_{int(quantile*100)}thpercentile"
                
            new_df[new_col_name] = (df[col] > value).astype(int)
    
    return new_df

def load_data(also_load_reports=False):
    """
    Structure of the data:
    - [0] : 'id' - particiapnt id
    - [1:32] : GPT (ada-002) embeddings per prompt type (TAT, RIT, IAPS, Self-Report)
        - [1:10] : IAPS ('IAPS2276', 'IAPS2388', 'IAPS2389', 'IAPS3005.1', 'IAPS3230', 'IAPS3530', 'IAPS4000', 'IAPS5781', 'IAPS6561', 'IAPS7004')
        - [11:20] : RIT ('RIT1', 'RIT2', 'RIT3', 'RIT4', 'RIT5', 'RIT6', 'RIT7', 'RIT8', 'RIT9', 'RIT10')
        - [21:30] : TAT ('TAT1', 'TAT2', 'TAT3bm', 'TAT4', 'TAT6bm', 'TAT8bm', 'TAT9gf', 'TAT10', 'TAT13mf', 'TAT18gf')
        - [31] : 'Q313' - Self-Report
    - [32:110] : Questionnaire scores (ZSDS, GAD, BIS, OLIFE, AES, AUDIT, OCI, LSAS, EAT, YSQ, NEO, DSQ)
    - [110:188] : Questionnaire scores above median
    - [188:191] : Factors (ML1, ML2, ML3)
        - [188] : 'ML1' - Social Withdrawal
        - [189] : 'ML2' - Compulsive Behavior and Intrusive Thought
        - [190] : 'ML3' - Anxious-Depression
    - [191:197] : Factors above median and above 75th percentile (i.e. 'ML1_above_median', 'ML1_above_75thpercentile', etc.)
    - [197:200] : Factors top/bottom 25% (i.e. 'ML1_top_bottom_25', etc.)

    Optionally, you can load the verbal reports if you want to get new embeddings. 
    'id' is in the same order for data and reports.
    """
    df = pd.read_csv('dp_data.csv', index_col=0)
    if also_load_reports:
        reports = pd.read_csv('cleaned_reports_combined.csv', index_col=0)
        return df, reports
    return df

def load_embeddings(prompts: str) -> np.ndarray:
    """
    Load 3-large embeddings for a given prompt type and 'longways', i.e. 10240 embeddings per participant.
    """
    prompt_indices = {'IAPS': 0, 'RIT': 10, 'TAT': 20}
    prompt_index = prompt_indices[prompts]

    # Load the embeddings
    emarray = np.load("text_embeddings_3_large.npy")

    # Init empty array of shape 210 x 10240 (this is because 1024 * 10, since 10 images per prompt type)
    embed_array = np.zeros(shape=(emarray.shape[0],10240))
    for j in range(0, emarray.shape[0]): # for each participant
        temp_embeds = []
        for i in range(0,10): # for each prompt type
            temp_embeds = np.concatenate((temp_embeds, np.squeeze(emarray[j, prompt_index+i ,:])), axis=0) # concatenate the 1024 embeddings;; change index to 0 for IAPS, 10 for RIT and 20 for TAT
        embed_array[j,:] = temp_embeds

    return embed_array

def run_val_aro():
    """
    Run a logistic regression model to predict valence and arousal above median. It is currently set to run valence, see line 158.
    """
    # load dp_data cuz we'll need the id column
    dp_data = pd.read_csv('dp_data.csv', index_col=0)

    # load the 3-large embeddings and add id column
    large = np.load('text_embeddings_3_large.npy')
    df = pd.DataFrame(index=range(210), columns=range(31))
    for participant in range(210):
        for prompt_type in range(31):
            df.iloc[participant, prompt_type] = large[participant, prompt_type].tolist()

    # Change column names to match the original data
    df.columns = dp_data.columns[1:32].tolist()
    df['id'] = dp_data['id']

    meadows = pd.read_csv('meadows_valaro_w_emebds.csv', index_col=0)
    meadows.iloc[:,1:32] = meadows.iloc[:,1:32].applymap(str_to_list)
    meadows.iloc[:,-10:] = meadows.iloc[:,-10:].applymap(str_to_list)

    ### In df, keep only rows where id is in meadows['id']
    iaps_large = df[df['id'].isin(meadows['id'])].copy()
    iaps_large = iaps_large.merge(meadows[['id'] + ['2276', '2389', '2388', '3005.1', '3230', '3530', '4000', '5781',
        '6561', '7004']], on='id', how='left')

    # Get embeddings 'longways', then reshape into stacked format
    large_long = pd.concat([iaps_large[col].apply(pd.Series) for col in iaps_large.columns[:10]], axis=1)
    reports_per_participant = 10
    reshaped_large_long = []
    for i in range(reports_per_participant):
        start_col = i * 1024
        end_col = start_col + 1024
        temp_large_long = large_long.iloc[:, start_col:end_col].copy()
        
        temp_large_long.columns = [f'Emb{j}' for j in range(1024)]
        temp_large_long['id'] = iaps_large['id']

        prompt_ratings = iaps_large.columns[:10][i][4:]
        temp_large_long['valence'] = iaps_large[prompt_ratings].apply(lambda x: x[0])
        temp_large_long['arousal'] = iaps_large[prompt_ratings].apply(lambda x: x[1])
        reshaped_large_long.append(temp_large_long)
    large_long = pd.concat(reshaped_large_long).reset_index(drop=True)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectPercentile()),
        ('LogisticRegression', LogisticRegression(penalty='elasticnet',solver = 'saga'))
    ])

    param_grid = {
        'feature_selection__percentile': [10, 20, 30, 40, 50],
        'LogisticRegression__C': [0.1, 1, 10, 100],
        'LogisticRegression__l1_ratio': [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    }

    scorer = make_scorer(roc_auc_score)
    cv = GroupKFold(n_splits=10)
    groups = large_long['id'].values
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scorer, cv=cv)

    total_results = {}
    # for target in ['arousal_above_median', 'valence_above_median']:
        
    # Perform grid search
    print("Starting gridsearch")
    gridsearch.fit(X=large_long.iloc[:,~large_long.columns.isin(['id', 'valence', 'arousal'])], 
                        y=add_indicator_columns(large_long.iloc[:,-2:])['valence_above_median'],
                        groups=groups)
    print(f"***\nBest parameters: {gridsearch.best_params_},\nBest score: {gridsearch.best_score_}\n")
    best_model = gridsearch.best_estimator_

    # Perform permutation test
    score, permutation_scores, pvalue = permutation_test_score(
        best_model,
        X=large_long.iloc[:,~large_long.columns.isin(['id', 'valence', 'arousal'])],
        y=add_indicator_columns(large_long.iloc[:,-2:])['valence_above_median'],
        groups=groups,
        scoring=scorer,
        cv=cv,
        n_permutations=5000,
        random_state=42,
        n_jobs=32
    )
    print(f"Permuted score on arousal_above_median: {score}, p-value: {pvalue}")
    total_results['valence_above_median'] = {'best_model': best_model, 'best_params': gridsearch.best_params_, 'score': score, 'permuted_scores': permutation_scores, 'p_value': pvalue}
    # with open(f'VAL_results_jun21.pkl', 'wb') as f:
    #     pickle.dump(total_results, f)


def run_mh():
    """
    Run a logistic regression model to predict mental health factors. It is currently set to run ML3 with Self-Report, see line 186 and 192. 
    """
    df = load_data()
    features = np.load("text_embeddings_3_large.npy")[:,-1,:] # load_embeddings('TAT') # or use this for self-report: np.load("/home/eclips/Documents/UnCLIP/text_embeddings_3_large.npy")[:,-1,:]
    
    targets = ['ML1','ML2','ML3','ML1_above_median', 'ML2_above_median', 'ML3_above_median', 
                'ML1_top_bottom_25', 'ML2_top_bottom_25', 'ML3_top_bottom_25'] 

    ###################################
    pos_target = 2
    current_target_name = targets[pos_target]
    ###################################
    print(f"Running {current_target_name}")
    current_target_values = df[current_target_name]
    current_target_values = zscore(current_target_values)

    percentile_high = np.percentile(current_target_values, 15)
    percentile_low = np.percentile(current_target_values, 85)

    # Recode values based on percentiles
    current_target_values = current_target_values.apply(lambda x: 1 if x > percentile_low else (0 if x < percentile_high else np.nan))
    current_target_values = current_target_values.astype('float64')

    current_target_values = current_target_values.dropna().astype(int)
    features = features[current_target_values.index,:]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectPercentile()),
        ('LogisticRegression', LogisticRegression(penalty='elasticnet',solver = 'saga'))
        ])

    param_grid = {
        'feature_selection__percentile': [10, 20, 30, 40, 50],
        'LogisticRegression__C': [0.1, 1, 10, 100],
        'LogisticRegression__l1_ratio': [0.1, 0.2, 0.4, 0.6, 0.8, 1]
        }

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    total_results = {}

    # Perform grid search
    grid_search = GridSearchCV(estimator=clone(pipeline), param_grid=param_grid, scoring='roc_auc', cv=outer_cv, n_jobs=-1)
    grid_search.fit(X=features,
                    y=current_target_values
                    )
    best_model = grid_search.best_estimator_

    # Permute
    score, perm_scores, p_value = permutation_test_score(best_model, features, current_target_values, scoring='roc_auc', cv=outer_cv, n_permutations=5000, random_state=42,n_jobs = -1)
    # total_results[current_target_name] = {'best_model': best_model, 'best_params': grid_search.best_params_, 'score':score, 'permuted_scores': perm_scores, 'p_value': p_value}
    total_results[current_target_name] = {'best_model': best_model, 'best_params': grid_search.best_params_, 'score':score, 'permuted_scores': perm_scores, 'p_value': p_value, 'coef': grid_search.best_estimator_.named_steps.LogisticRegression.coef_}
    print(f"Best params: {grid_search.best_params_}")
    print(f"In-Fold Permuted score {current_target_name}: {score}, p-value: {p_value}")

    # with open(f'LogReg_Self_mh_'+str(pos_target+1)+'_perc_feat_results.pkl', 'wb') as f:
    #     pickle.dump(total_results, f)