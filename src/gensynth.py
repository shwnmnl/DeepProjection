import torch
import vec2text
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from diffusers import StableDiffusionXLPipeline

def predict(X, W, b) :
    return torch.matmul(X, W ) + b

def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

def rec_X(target, mean_features, W, b):
    dim = W.shape[0]
    #X = torch.tensor((1, dim), dtype=torch.float64, requires_grad=True)
    mean_features = np.reshape(mean_features, (1,dim))  
    X = torch.tensor(mean_features, requires_grad=True)
    
    W = np.transpose(W)
    W = np.reshape(W, (dim, 1))     
    W = torch.tensor(W)
    
    b = torch.tensor(b)
    
    #target = ((target+3)*10) / 2

    step_size = 0.1
    loss_list = []
    iter = 300
     
    for i in range (iter):
        # making predictions with forward pass
        Y_pred = predict(X, W, b)

        # calculating the loss between original and predicted data points
        loss = criterion(Y_pred, target)

        # storing the calculated loss in a list
        loss_list.append(loss.item())
        
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        
        # updateing the parameters after each iteration
        X.data = X.data - step_size * X.grad.data
        
        # zeroing gradients after each iteration
        X.grad.data.zero_()
        
        # priting the values for understanding
        print('{},\t{},\t{}'.format(i, loss.item(), Y_pred.item()))
     
    return X, Y_pred


def str_to_list(x):
    """
    Convert string to list.
    """
    return [float(i) for i in x.strip('][').split(',')]

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

def Get_embeddings(emb_option):

    if emb_option == 'ADA':
        # LOAD or EXTRACT embeddings + see 'if top_bottom in current_target_name:' below
        #features = ['TAT1','TAT2','TAT3bm','TAT4','TAT6bm','TAT8bm','TAT9gf','TAT10','TAT13mf','TAT18gf']
        #features = ['IAPS2276', 'IAPS2388', 'IAPS2389', 'IAPS3005.1', 'IAPS3230', 'IAPS3530', 'IAPS4000', 'IAPS5781', 'IAPS6561', 'IAPS7004','Q313']
        features = ['Q313']
        data = df[features]

        embed_array = np.empty((data.shape[0],10*1536))
        embed_array[:] = np.nan

        for part in range(0,data.shape[0]):
            for k in range(0,10):
                embed_array[part,(k*1536):((k+1)*1536)] = str_to_list(data.iloc[part,k])

    elif emb_option == '3-large':
        emarray = np.load("text_embeddings_3_large.npy")

        embed_array = np.zeros(shape=(emarray.shape[0],10240))
        for j in range(0,emarray.shape[0]):
            temp_embeds = []
            for i in range(0,10):
                temp_embeds = np.concatenate((temp_embeds, np.squeeze(emarray[j,20+i,:])), axis=0)
            embed_array[j,:] = temp_embeds

    return embed_array

df = load_data()
emb_option = 'ADA'
features = Get_embeddings(emb_option)
targets = ['ML1','ML2','ML3','ML1_above_median', 'ML2_above_median', 'ML3_above_median', 
            'ML1_top_bottom_25', 'ML2_top_bottom_25', 'ML3_top_bottom_25'] 

current_target_name = targets[1] # This example uses the first mental health factor
current_target_values = df[current_target_name]

#current_target_values = zscore(current_target_values)
percentile_high = np.percentile(current_target_values, 15)
percentile_low = np.percentile(current_target_values, 85)

# Recode values based on percentiles
Low_high = current_target_values.apply(lambda x: 1 if x > percentile_low else (0 if x < percentile_high else np.nan))
Low_high = Low_high.astype('float64')

Low_high = Low_high.dropna().astype(int)

#Those are the embeddings but with only the extreme values
im = 0
features_Lh = features[Low_high.index,(1536*im):(1536*(im+1))]
mean_features = np.mean(features[:, (1536*im):(1536*(im+1))], axis =0)


########################################
#   This is the evolution procedure
#           (can be skipped)
########################################

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lgReg', LogisticRegression())
    ])

clf = pipeline.fit(features_Lh, Low_high)

[embedding, y_pred] = rec_X(1, mean_features, np.absolute(clf['lgReg'].coef_[0]), clf['lgReg'].intercept_)
r_emb = embedding.to('cuda')

# Then do just mean group embeddings
mean_high = np.mean(features_Lh[Low_high == 1,:], axis = 0)
mean_low = np.mean(features_Lh[Low_high == 0,:], axis = 0)

########################################
#   Get a prompt from an embedding 
########################################
# https://github.com/jxmorris12/vec2text
corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")

# Start with the embedding of the evolution process
prompt = vec2text.invert_embeddings(
    embeddings=r_emb,
    corrector=corrector
)
prompt_evo = prompt

# For high anxiety people
mean_high = torch.from_numpy(mean_high).float()
mean_high = torch.reshape(mean_high, (1, 1536))
mean_high = mean_high.to("cuda")

prompt_mean_high = vec2text.invert_embeddings(
    embeddings=mean_high,
    corrector=corrector
)

# For Low anxiety people
mean_low = torch.from_numpy(mean_low).float()
mean_low = torch.reshape(mean_low, (1, 1536))
mean_low = mean_low.to("cuda")

prompt_mean_low = vec2text.invert_embeddings(
    embeddings=mean_low,
    corrector=corrector
)

########################################
#      Reconstruction process
########################################

pipe = StableDiffusionXLPipeline.from_pretrained("/home/eclips/Documents/UnCLIP/StableDiffusionXLPipeline/")
pipe = pipe.to("cuda")

[image,prompt_embeds,add_text_embeds] = pipe(prompt = prompt_evo,return_dict=False)
image[0].save("Factor_2_IAPS3230_2_prompt_evo_3.png")

[image,prompt_embeds,add_text_embeds] = pipe(prompt = prompt_mean_low,return_dict=False)
image[0].save("Factor_2_IAPS3230_2_prompt_mean_low.png")

[image,prompt_embeds,add_text_embeds] = pipe(prompt = prompt_mean_high,return_dict=False)
image[0].save("Factor_2_IAPS3230_2_prompt_mean_high.png")

[image,prompt_embeds,add_text_embeds] = pipe(prompt = prompt_mean_high,prompt_2 = prompt_mean_low,return_dict=False)
image[0].save("Factor_2_IAPS3230_2_prompt_mean_low_high_3.png")