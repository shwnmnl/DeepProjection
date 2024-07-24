import pandas as pd

def score_zsds(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Scoring function for the ZSDS questionnaire.

    - Reverse codes items [1,2,3,4] -> [4,3,2,1]: 2, 5, 6, 11, 12, 14, 16, 17, 18, 20
        (-1 when looping because 0 indexing)
    - Sums all items for total
    '''
    df = df.copy().astype(int)
    columns_to_switch = [1, 4, 5, 10, 11, 13, 15, 16, 17, 19]
    for i in columns_to_switch:
        df.iloc[:, i] = df.iloc[:, i].replace([1,2,3,4], [4,3,2,1])

    df['ZSDS_Total'] = df.sum(axis=1)
    return df

def score_gad(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Scoring function for the GAD questionnaire.

    - Removes attention check item 5
    - Renumbers columns
    - Subtracts one from each item in the df (coding mix up)
    - Sums all items for total
    '''
    df = df.copy().astype(int)
    df = df.drop(columns=['GAD_5'])
    # Rename
    df.columns = ['GAD_' + str(i) for i in range(1, df.shape[1]+1)]
    # Subtract 1 from each item becasue the GAD rating scale is from 0-3
    df = df - 1
    df['GAD_Total'] = df.sum(axis=1)
    return df

def score_olife(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for OLIFE questionnaire. 

    OLIFE has 4 short scales: Unusual Experiences, Cognitive Disorganization, Introvertive Anhedonia, Impulsive Nonconformity

         Subscale    |    n items    |           items (* = reverse coded)
    ---------------------------------------------------------------------------------
    1. Unusual Exp.  |      12       |  1, 6, 10, 15, 18, 23, 27, 29, 32, 33, 38, 40 
    ---------------------------------------------------------------------------------
    2. Cog. Disorg.  |      11       |  2, 9, 12, 16, 17, 22, 26, 36, 39, 41, 43
    ---------------------------------------------------------------------------------
    3. Introv. Anhe. |      10       |  3, 5*, 8, 14, 19*, 21, 25, 28*, 31*, 34*  
    ---------------------------------------------------------------------------------
    4. Imp. Nonconf. |      10       |  4, 7, 11, 13, 20, 24*, 30*, 35, 37, 42* 


    - Recodes 2s as 0s (coding mix up)
    - Reverse codes items (0 <-> 1): 5, 19, 24, 28, 30, 31, 34, 42
        (-1 when looping because 0 indexing)
    - Sum all for total
    - Sum subscales for total
    '''
    df = df.copy().astype(int)
    df = df.replace(2,0)
    columns_to_switch = [4, 18, 23, 27, 29, 30, 33, 41]
    for i in columns_to_switch:
        df.iloc[:, i] = df.iloc[:, i].replace([0,1], [1,0])

    # Total and subscale scores
    df['OLIFE_Total'] = df.sum(axis=1)
    df['OLIFE_UnExp'] = df.iloc[:, [0, 5, 9, 14, 17, 22, 26, 28, 31, 32, 37, 39]].sum(axis=1)
    df['OLIFE_CogDis'] = df.iloc[:, [1, 8, 11, 15, 16, 21, 25, 35, 38, 40, 42]].sum(axis=1)
    df['OLIFE_IntAnh'] = df.iloc[:, [2, 4, 7, 13, 18, 20, 24, 27, 30, 33]].sum(axis=1)
    df['OLIFE_ImpNon'] = df.iloc[:, [3, 6, 10, 12, 19, 23, 29, 34, 36, 41]].sum(axis=1)
    return df

def score_bis(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Scoring function for BIS questionnaire.

    BIS has 6 1st Order subscales: [Attentional, Cognitive Instability], [Motor, Perseverance], [Self-Control, Cognitive Complexity]
    BIS has 3 2nd Order subscales:          Attentional,                    Motor,                     Nonplanning

        2nd Order    |      1st Order      |    n items    |    items (* = reverse coded)
    -------------------------------------------------------------------------------------
    1. Attentional   |    1. Attention     |       5       |    5, 9*, 11, 20*, 28

                     |  2. Cog. Instab.    |       3       |    6, 24, 26
    -------------------------------------------------------------------------------------
    2. Motor         |     3. Motor        |       7       |    2, 3, 4, 17, 19, 22, 25

                     |  4. Perseverance    |       4       |    16, 21, 23, 30*
    -------------------------------------------------------------------------------------
    3. Nonplanning   |  5. Self-Control    |       6       |    1*, 7*, 8*, 12*, 13*, 14
    
                     | 6. Cog. Complexity  |       5       |    10*, 15*, 18, 27, 29*
    
                  
    - Reverse codes items (0 <-> 1): 1, 7, 8, 9, 10, 12, 13, 15, 20, 29, 30
        (-1 when looping because 0 indexing)
    - Sum all for total
    - Sum 1st and 2nd order subscales
    '''
    df = df.copy().astype(int)
    columns_to_switch = [0, 6, 7, 8, 9, 11, 12, 14, 19, 28, 29]
    for i in columns_to_switch:
        df.iloc[:, i] = df.iloc[:, i].replace([0,1], [1,0])
    
    # Total, 1st order and 2nd order scores
    df['BIS_Total'] = df.sum(axis=1)

    df['BIS_Att_Attention'] = df.iloc[:, [4, 8, 10, 19, 27]].sum(axis=1)
    df['BIS_Att_CogInstab'] = df.iloc[:, [5, 23, 25]].sum(axis=1)
    df['BIS_Mot_Motor'] = df.iloc[:, [1, 2, 3, 16, 18, 21, 24]].sum(axis=1)
    df['BIS_Mot_Perseverance'] = df.iloc[:, [15, 20, 22, 29]].sum(axis=1)
    df['BIS_Nonp_SelfControl'] = df.iloc[:, [0, 6, 7, 11, 12, 13]].sum(axis=1)
    df['BIS_Nonp_CogComplex'] = df.iloc[:, [9, 14, 17, 26, 28]].sum(axis=1)

    df['BIS_Att'] = df.iloc[:, [4, 5, 8, 10, 19, 23, 25, 27]].sum(axis=1)
    df['BIS_Mot'] = df.iloc[:, [1, 2, 3, 15, 16, 18, 20, 21, 22, 24, 29]].sum(axis=1)
    df['BIS_Nonp'] = df.iloc[:, [0, 6, 7, 9, 11, 12, 13, 14, 17, 26, 28]].sum(axis=1)
    return df
    
def score_aes(df: pd.DataFrame) -> pd.DataFrame:    
    '''
    Scoring function for AES questionnaire.

    - Removes attention check item 8 and renames columns
    - Reverse codes all items except 6, 10, 11 (so 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18)
        (-1 when looping because 0 indexing)
    - Sums all items for total
    '''
    df = df.copy().astype(int)
    df = df.drop(columns=['AES_8'])
    df.columns = ['AES_' + str(i) for i in range(1, df.shape[1]+1)]

    columns_to_switch = [0, 1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17]
    for i in columns_to_switch:
        df.iloc[:, i] = df.iloc[:, i].replace([1,2,3,4], [4,3,2,1])
        
    df['AES_Total'] = df.sum(axis=1)
    return df

def score_audit(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Scoring function for AUDIT questionnaire.

    - Recodes [1,2,3,4,5] -> [0,1,2,3,4], except for items 9-10 where [1,2,3] -> [0,2,4]
        (-1 when looping because 0 indexing)
    - Sums all items for total
    '''
    df = df.copy().astype(int)
    recode_some_cols = [8,9]
    recode_rest_cols = [0,1,2,3,4,5,6,7]

    for i in recode_some_cols:
        df.iloc[:, i] = df.iloc[:, i].replace([1,2,3], [0,2,4])

    for i in recode_rest_cols:
        df.iloc[:, i] = df.iloc[:, i].replace([1,2,3,4,5], [0,1,2,3,4])

    df['AUDIT_Total'] = df.sum(axis=1)
    return df

def score_oci(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for OCI questionnaire.

    OCI-R has 2 components: OCD and Hoarding
    OCI-R has 6 subscales: [Washing, Obsessing, Ordering, Checking, Neutralising], Hoarding

      Component     |      Subscale     |    n items    |     items
    -------------------------------------------------------------------
    1. OCD          |    1. Washing     |       3       |    5, 11, 17
    
                    |    2. Obsessing   |       3       |    6, 12, 18
   
                    |    3. Ordering    |       3       |    3, 9, 15
    
                    |    4. Checking    |       3       |    2, 8, 14

                    |  5. Neutralising  |       3       |    4, 10, 16
    -------------------------------------------------------------------
    2. Hoarding     |    6. Hoarding    |       3       |    1, 7, 13

    - Recodes [1,2,3,4,5] -> [0,1,2,3,4]
    - Sums all for total
    - Sums subscales for total
        (-1 when looping because 0 indexing)
    '''
    df = df.copy().astype(int)
    for col in df.columns:
        df[col] = df[col].replace([1,2,3,4,5], [0,1,2,3,4])

    # Total and subscale scores
    df['OCI_Total'] = df.sum(axis=1)

    df['Washing'] = df.iloc[:, [4, 10, 16]].sum(axis=1)
    df['Obsessing'] = df.iloc[:, [5, 11, 17]].sum(axis=1)
    df['Ordering'] = df.iloc[:, [2, 8, 14]].sum(axis=1)
    df['Checking'] = df.iloc[:, [1, 7, 13]].sum(axis=1)
    df['Neutralising'] = df.iloc[:, [3, 9, 15]].sum(axis=1)
    df['Hoarding'] = df.iloc[:, [0, 6, 12]].sum(axis=1) 
    return df

def score_lsas(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for the LSAS questionnaire.

    LSAS items measure two things: Fear and Avoidance. 
    Thus, the structure of the data currently is a floating point number. 
    The first digit is the Fear score and the second digit is the Avoidance score.
    There was a coding issue, so first digits need to be recoded [1,2,3,4] -> [0,1,2,3]
    and second digits need to be recoded [5,6,7,8] -> [0,1,2,3]
    
    LSAS has 2 subscales: Performance Anxiety and Social Situations.
    
           Subscale     |    n items     |                    items
    --------------------------------------------------------------------------------------
    1. Performance Anx. |       13       |    1, 2, 3, 4, 6, 8, 9, 13, 14, 16, 17, 20, 21
    --------------------------------------------------------------------------------------
    2. Social Sit.      |       11       |    5, 7, 10, 11, 12, 15, 18, 19, 22, 23, 24

    - Splits Fear and Avoidance scores
    - Recodes [1,2,3,4] -> [0,1,2,3] for Fear 
    - Recodes [5,6,7,8] -> [0,1,2,3] for Avoidance
    - Sums all for total
    - Sums Fear and Avoidance scores individually for total
        (-1 when looping because 0 indexing)
    - Sums Performance Anxiety and Social Situations subscales for total
        (-1 when looping because 0 indexing)
    '''
    df = df.copy()

    def convert_and_modify(x):
        '''
        Function to fix a small mix up where some participant answers only contain one value
        For values where the only available value is [1,2,3,4], we will add the same digit to the end (+4)
        For values where the only available value is [5,6,7,8], we will subtract the same digit from the end (-4)
        
        '''
        if ',' in x:
            # If it's a valid float with a comma, replace ',' with '.' and convert to float
            return float(x.replace(',', '.'))
        else:
            # If it's not a valid float (e.g., '3'), modify it
            if len(x) == 1:
                first_digit = int(x)
                if 1 <= first_digit <= 4:
                    return float(f"{first_digit}.{first_digit + 4}")
                elif 5 <= first_digit <= 8:
                    return float(f"{first_digit-4}.{first_digit}")
                return float(f"{first_digit}.{first_digit + 4}")
            else:
                return float(x)

    df = df.applymap(convert_and_modify)

    df_fear = df.copy()
    df_fear.columns = ['LSAS_Fear_' + str(i) for i in range(1, df_fear.shape[1]+1)]
    df_fear = df_fear.astype(str)
    df_fear = df_fear.apply(lambda x: x.str[0])
    df_fear = df_fear.replace(['1', '2', '3', '4'], [0,1,2,3])
    
    df_avoid = df.copy()
    df_avoid.columns = ['LSAS_Avoid_' + str(i) for i in range(1, df_avoid.shape[1]+1)]
    df_avoid = df_avoid.astype(str)
    df_avoid = df_avoid.apply(lambda x: x.str[2])
    df_avoid = df_avoid.replace(['5', '6', '7', '8'], [0,1,2,3])

    df_fear_avoid = pd.concat([df_fear, df_avoid], axis=1)

    # Total, Fear/Avoidane scores, and subscale scores
    df['LSAS_Total'] = df_fear_avoid.sum(axis=1)

    df['LSAS_Fear'] = df_fear.sum(axis=1)
    df['LSAS_Avoidance'] = df_avoid.sum(axis=1)

    df['LSAS_PerfAnx'] = df_fear_avoid.iloc[:, [0, 1, 2, 3, 5, 7, 8, 12, 13, 15, 16, 19, 20]].sum(axis=1)
    df['LSAS_SocSit'] = df_fear_avoid.iloc[:, [4, 6, 9, 10, 11, 14, 17, 18, 21, 22, 23]].sum(axis=1)

    # Add mean fear/avoidance for each item, this was done in Rouault's paper for FA
    for i in range(1, 25):
        df['LSAS_Mean' + str(i)] = df_fear_avoid['LSAS_Fear_' + str(i)].astype(int) + df_fear_avoid['LSAS_Avoid_' + str(i)].astype(int)/2

    return df

def score_eat(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for the EAT questionnaire.

    - Recodes [1,2,3,4,5] -> [3,2,1,0,0,0], except item 26 where [1,2,3,4,5] -> [0,0,0,1,2,3]
    - Sums all for total
    '''
    df = df.copy().astype(int)

    recode_some_cols = [25]
    recode_rest_cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                        15,16,17,18,19,20,21,22,23,24]
    
    for i in recode_some_cols:
        df.iloc[:, i] = df.iloc[:, i].replace([1,2,3,4,5,6], [0,0,0,1,2,3])
    
    for i in recode_rest_cols:
        df.iloc[:, i] = df.iloc[:, i].replace([1,2,3,4,5,6], [3,2,1,0,0,0])

    df['EAT_Total'] = df.sum(axis=1)
    return df

def score_ysq(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for the YSQ-S3 questionnaire.

    YSQ has 18 Early Maladaptive Schemas and no total score. 

           Schema     |    n items    |         items
    ----------------------------------------------------------
    1. Emo. Depriv.   |       5       |    1, 19, 37, 55, 73
    ----------------------------------------------------------
    2. Abandonment    |       5       |    2, 20, 38, 56, 74
    ----------------------------------------------------------
    3. Mistrust       |       5       |    3, 21, 39, 57, 75
    ----------------------------------------------------------
    4. Social Isol.   |       5       |    4, 22, 40, 58, 76
    ----------------------------------------------------------
    5. Unlovability   |       5       |    5, 23, 41, 59, 77
    ----------------------------------------------------------
    6. Failure        |       5       |    6, 24, 42, 60, 78
    ----------------------------------------------------------
    7. Dependence     |       5       |    7, 25, 43, 61, 79
    ----------------------------------------------------------
    8. Vulnerability  |       5       |    8, 26, 44, 62, 80
    ----------------------------------------------------------
    9. Enmeshment     |       5       |    9, 27, 45, 63, 81
    ----------------------------------------------------------
    10. Subjugation   |       5       |    10, 28, 46, 64, 82
    ----------------------------------------------------------
    11. Self-Sacr.    |       5       |    11, 29, 47, 65, 83
    ----------------------------------------------------------
    12. Emo. Inh.     |       5       |    12, 30, 48, 66, 84
    ----------------------------------------------------------
    13. Unrel. Stand. |       5       |    13, 31, 49, 67, 85
    ----------------------------------------------------------
    14. Entitlement   |       5       |    14, 32, 50, 68, 86
    ----------------------------------------------------------
    15. Insuf. Self-C.|       5       |    15, 33, 51, 69, 87
    ----------------------------------------------------------
    16. Recog.-Seek.  |       5       |    16, 34, 52, 70, 88
    ----------------------------------------------------------
    17. Pessimism     |       5       |    17, 35, 53, 71, 89
    ----------------------------------------------------------
    18. Self-Punit.   |       5       |    18, 36, 54, 72, 90

    - Calculates mean for each EMS
    '''
    df = df.copy().astype(int)

    df['YSQ_EmoDepriv'] = df.iloc[:, [0, 18, 36, 54, 72]].mean(axis=1)
    df['YSQ_Abandonment'] = df.iloc[:, [1, 19, 37, 55, 73]].mean(axis=1)
    df['YSQ_Mistrust'] = df.iloc[:, [2, 20, 38, 56, 74]].mean(axis=1)
    df['YSQ_SocIsol'] = df.iloc[:, [3, 21, 39, 57, 75]].mean(axis=1)
    df['YSQ_Unlovability'] = df.iloc[:, [4, 22, 40, 58, 76]].mean(axis=1)
    df['YSQ_Failure'] = df.iloc[:, [5, 23, 41, 59, 77]].mean(axis=1)
    df['YSQ_Dependence'] = df.iloc[:, [6, 24, 42, 60, 78]].mean(axis=1)
    df['YSQ_Vulnerability'] = df.iloc[:, [7, 25, 43, 61, 79]].mean(axis=1)
    df['YSQ_Enmeshment'] = df.iloc[:, [8, 26, 44, 62, 80]].mean(axis=1)
    df['YSQ_Subjugation'] = df.iloc[:, [9, 27, 45, 63, 81]].mean(axis=1)    
    df['YSQ_SelfSacrifice'] = df.iloc[:, [10, 28, 46, 64, 82]].mean(axis=1)
    df['YSQ_EmoInhibition'] = df.iloc[:, [11, 29, 47, 65, 83]].mean(axis=1)
    df['YSQ_UnrelStandards'] = df.iloc[:, [12, 30, 48, 66, 84]].mean(axis=1)
    df['YSQ_Entitlement'] = df.iloc[:, [13, 31, 49, 67, 85]].mean(axis=1)
    df['YSQ_InsufSelfControl'] = df.iloc[:, [14, 32, 50, 68, 86]].mean(axis=1)
    df['YSQ_RecogSeeking'] = df.iloc[:, [15, 33, 51, 69, 87]].mean(axis=1)
    df['YSQ_Pessimism'] = df.iloc[:, [16, 34, 52, 70, 88]].mean(axis=1)
    df['YSQ_SelfPunishment'] = df.iloc[:, [17, 35, 53, 71, 89]].mean(axis=1)
    return df

def score_neo(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for the NEO-FFI questionnaire.

    Neo has 5 factors, commonly known as the Big Five in personality research.

            Factor        |    n items     |                    items (* = reverse coded)
    ------------------------------------------------------------------------------------------------
    1. Openness           |       12       |   3, 8, 13, 18*, 23*, 28*, 33*, 38, 43, 48*, 53, 58
    ------------------------------------------------------------------------------------------------
    2. Conscientiousness  |       12       |   5, 10, 15*, 20, 25, 30*, 35, 40, 45*, 50, 55*, 60
    ------------------------------------------------------------------------------------------------
    3. Extraversion       |       12       |   2, 7, 12*, 17, 22, 27*, 32, 37, 42*, 47, 52, 57*
    ------------------------------------------------------------------------------------------------
    4. Agreeableness      |       12       |   4, 9*, 14*, 19*, 24*, 29, 34, 39*, 44*, 49, 54*, 59*
    ------------------------------------------------------------------------------------------------
    5. Neuroticism        |       12       |   1*, 6, 11, 16*, 21, 26, 31*, 36, 41, 46*, 51, 56

    - Recodes [1,2,3,4,5] -> [0-1-2-3-4]
    - Reverse codes items : 1,9,12,14,15,16,18,19,23,24,27,28,30,31,33,39,42,44,45,46,48,54,55,57,59
        (-1 when looping because 0 indexing)
    - Sums all for total
    - Sums factors for total 
    '''
    df = df.copy().astype(int)
    df = df.replace([1,2,3,4,5], [0,1,2,3,4])

    columns_to_switch = [0,8,11,13,14,15,17,18,22,23,26,27,29,30,32,38,41,43,44,45,47,53,54,56,58]
    for i in columns_to_switch:
        df.iloc[:, i] = df.iloc[:, i].replace([0,1,2,3,4], [4,3,2,1,0])

    df['NEO_Openness'] = df.iloc[:, [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57]].sum(axis=1)
    df['NEO_Conscientiousness'] = df.iloc[:, [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]].sum(axis=1)
    df['NEO_Extraversion'] = df.iloc[:, [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]].sum(axis=1)
    df['NEO_Agreeableness'] = df.iloc[:, [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58]].sum(axis=1)
    df['NEO_Neuroticism'] = df.iloc[:, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]].sum(axis=1)        
    return df

def score_dsq(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
    Scoring function for the DSQ-40 questionnaire.

    DSQ has 3 factors: Mature, Immature and Neurotic Defense Mechanisms
    DSQ measures 20 defense mechanisms, unevenly distributed among the 3 factors.

    The numbering is inherited from the original DSQ, which is why it exceeds 40.
    Since we numbered ours from 1-40, we provide the original number in parentheses.



        Factor           |    Defense Mech.          |    n items    |    items (og numbering)
    ------------------------------------------------------------------------------------------
    1. Mature            |    1. Sublimation         |       2       |   3, 38 (5, 84)
    
                         |    2. Humour              |       2       |   5, 26 (8, 61)

                         |    3. Suppression         |       2       |   2, 25 (3, 59)

                         |    4. Anticipation        |       2       |   30, 35 (68, 81)
    ------------------------------------------------------------------------------------------
    2. Neurotic          |    5. Undoing             |       2       |   32, 40 (71, 88)
      
                         |    6. Idealization        |       2       |   21, 24 (51, 58)

                         |    7. Pseudoaltruism      |       2       |   1, 39 (1, 86)

                         |    8. Reaction Formation  |       2       |   7, 28 (13, 63)
    ------------------------------------------------------------------------------------------
    3. Immature          |    9. Projection          |       2       |   6, 29 (12, 66)

                         |    10. Passive Aggression |       2       |   23, 36 (54, 82)

                         |    11. Acting Out         |       2       |   11, 20 (27, 46)

                         |    12. Isolation          |       2       |   34, 37 (76, 83)

                         |    13. Devaluation        |       2       |   10, 13 (24, 29)

                         |    14. Fantasy            |       2       |   14, 17 (31, 40)

                         |    15. Denial             |       2       |   8, 18 (16, 42)

                         |    16. Displacement       |       2       |   31, 33 (69, 73)

                         |    17. Dissociation       |       2       |   9, 15 (23, 37)

                         |    18. Splitting          |       2       |   19, 22 (43, 53)

                         |    19. Rationalization    |       2       |   4, 16 (6, 38)

                         |    20. Somatization       |       2       |   12, 27 (28, 62)

    - Averages both items for each defense mechanism to get mechanism score
    - Averages mechanism scores for each factor to get factor score    
    '''
    df = df.copy().astype(int)
    df = df.replace([1,2,3,10,11,12,13,14,15], [1,2,3,4,5,6,7,8,9])

    # Mechanism scores
    df['DSQ_Sublimation'] = df.iloc[:, [2, 37]].mean(axis=1)
    df['DSQ_Humour'] = df.iloc[:, [4, 25]].mean(axis=1)
    df['DSQ_Suppression'] = df.iloc[:, [1, 24]].mean(axis=1)
    df['DSQ_Anticipation'] = df.iloc[:, [29, 34]].mean(axis=1)
    df['DSQ_Undoing'] = df.iloc[:, [31, 39]].mean(axis=1)
    df['DSQ_Idealization'] = df.iloc[:, [20, 23]].mean(axis=1)
    df['DSQ_Pseudoaltruism'] = df.iloc[:, [0, 38]].mean(axis=1)
    df['DSQ_ReactionFormation'] = df.iloc[:, [6, 27]].mean(axis=1)
    df['DSQ_Projection'] = df.iloc[:, [5, 28]].mean(axis=1)
    df['DSQ_PassiveAggression'] = df.iloc[:, [22, 35]].mean(axis=1)
    df['DSQ_ActingOut'] = df.iloc[:, [10, 19]].mean(axis=1)
    df['DSQ_Isolation'] = df.iloc[:, [33, 36]].mean(axis=1)
    df['DSQ_Devaluation'] = df.iloc[:, [9, 12]].mean(axis=1)
    df['DSQ_Fantasy'] = df.iloc[:, [13, 16]].mean(axis=1)
    df['DSQ_Denial'] = df.iloc[:, [7, 17]].mean(axis=1)
    df['DSQ_Displacement'] = df.iloc[:, [30, 32]].mean(axis=1)
    df['DSQ_Dissociation'] = df.iloc[:, [8, 14]].mean(axis=1)
    df['DSQ_Splitting'] = df.iloc[:, [18, 21]].mean(axis=1)
    df['DSQ_Rationalization'] = df.iloc[:, [3, 15]].mean(axis=1)
    df['DSQ_Somatization'] = df.iloc[:, [11, 26]].mean(axis=1)
    
    # Get factors scores by averaging mechanism scores
    # For example, mature score is the average of sublimation, humour, suppression and anticipation scores
    df['DSQ_Mature'] = (df['DSQ_Sublimation'] + df['DSQ_Humour'] + df['DSQ_Suppression'] + df['DSQ_Anticipation']) / 4
    df['DSQ_Neurotic'] = (df['DSQ_Undoing'] + df['DSQ_Idealization'] + df['DSQ_Pseudoaltruism'] + df['DSQ_ReactionFormation']) / 4  
    df['DSQ_Immature'] = (df['DSQ_Projection'] + df['DSQ_PassiveAggression'] + df['DSQ_ActingOut'] + df['DSQ_Isolation'] + df['DSQ_Devaluation'] + df['DSQ_Fantasy'] + df['DSQ_Denial'] + df['DSQ_Displacement'] + df['DSQ_Dissociation'] + df['DSQ_Splitting'] + df['DSQ_Rationalization'] + df['DSQ_Somatization']) / 12
    
    return df

questionnaire_scoring_functions = {
    'ZSDS': score_zsds,
    'GAD': score_gad,
    'BIS': score_bis,
    'OLIFE': score_olife,
    'AES': score_aes,
    'AUDIT': score_audit,
    'OCI': score_oci,
    'LSAS': score_lsas,
    'EAT': score_eat,
    'YSQ': score_ysq,
    'NEO': score_neo,
    'DSQ': score_dsq
}

def score_all_questionnaires(questionnaires: dict, questionnaire_scoring_functions: dict = questionnaire_scoring_functions) -> pd.DataFrame:
    """
    Score all questionnaires and collect scored DataFrames
    
    Args:
    questionnaires: dict of DataFrames with questionnaire prefixes as keys and questionnaire DataFrames as values
    questionnaire_scoring_functions: dict of questionnaire scoring functions with questionnaire prefixes as keys and scoring functions as values

    Returns:
    pd.DataFrame: DataFrame with scored questionnaires
    """
    scored_questionnaires = {}
    for questionnaire_name, questionnaire_df in questionnaires.items():
        scoring_function = questionnaire_scoring_functions[questionnaire_name]
        scored_questionnaires[questionnaire_name] = scoring_function(questionnaire_df)
    final_scores_df = pd.concat(scored_questionnaires, axis=1)
    return final_scores_df

def extract_items(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract item columns from a DataFrame with MultiIndexed columns, i.e. the output of 'score_all_questionnaires'

    Args:
    scored_df: DataFrame with MultiIndexed columns

    Returns:
    pd.DataFrame: DataFrame with item columns
    """
    pattern = r'^[A-Z]+_?.*\d+$'
    # r'^[A-Z]+_.*\d+$' # Pattern to match item columns
    mask = [re.match(pattern, col[1]) is not None for col in scored_df.columns] # Create a boolean mask for columns where the second level matches the pattern
    item_columns = scored_df.loc[:, mask] # Apply the mask to filter columns based on the second level
    return item_columns