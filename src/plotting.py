import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import statsmodels.stats.multitest as smm
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

def plot_dimreduc_latentspaces():
    '''
    Plot the latent spaces of the mental health embeddings using PCA, UMAP, and t-SNE.
    '''
    large_embeds = np.load('large_embeds.npy')

    # Load your data and standardize
    X = StandardScaler().fit_transform(large_embeds.reshape(-1, large_embeds.shape[1]))

    # Create a pipeline for PCA, UMAP, t-SNE
    pca = PCA(n_components=3)
    umap = UMAP(n_components=3,
                n_neighbors=5,
                min_dist=0.9)
    tsne = TSNE(n_components=3, 
                perplexity=2)

    # Fit-transform the data
    X_pca = pca.fit_transform(X)
    X_umap = umap.fit_transform(X)
    X_tsne = tsne.fit_transform(X)

    # Create DataFrames
    df_pca = pd.DataFrame(X_pca, columns=['Component 1', 'Component 2', 'Component 3'])
    # df_pca['id'] = np.repeat(participant_ids, 31)
    df_umap = pd.DataFrame(X_umap, columns=['Component 1', 'Component 2', 'Component 3'])
    # df_umap['id'] = np.repeat(participant_ids, 31)      
    df_tsne = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2', 'Component 3'])
    # df_tsne['id'] = np.repeat(participant_ids, 31)

    # Prepare prompt labels and assign to DataFrames
    prompt_names = ['IAPS2276', 'IAPS2388', 'IAPS2389', 'IAPS3005.1', 'IAPS3230', 'IAPS3530', 'IAPS4000', 'IAPS5781', 'IAPS6561', 'IAPS7004', 
                    'RIT1', 'RIT2', 'RIT3', 'RIT4', 'RIT5', 'RIT6', 'RIT7', 'RIT8', 'RIT9', 'RIT10',
                    'TAT1', 'TAT2', 'TAT3bm', 'TAT4', 'TAT6bm', 'TAT8bm', 'TAT9gf', 'TAT10', 'TAT13mf', 'TAT18gf', 
                    'Self']
    num_samples = len(X) // len(prompt_names)
    repeated_image_names = np.tile(prompt_names, num_samples)[:len(X)]

    df_pca['Prompt'] = repeated_image_names
    df_umap['Prompt'] = repeated_image_names
    df_tsne['Prompt'] = repeated_image_names

    # Categories and their base colors
    categories = {
        'IAPS': 'Blues',
        'RIT': 'Greens',
        'TAT': 'Reds',
        'Self': 'Purples'
    }

    # Determine the number of prompts in each category to know how many shades we need
    prompt_counts = {
        'IAPS': sum(1 for p in prompt_names if p.startswith('IAPS')),
        'RIT': sum(1 for p in prompt_names if p.startswith('RIT')),
        'TAT': sum(1 for p in prompt_names if p.startswith('TAT')),
        'Self': 1
    }

    # Generate a color palette for each category
    color_palettes = {
        category: sns.color_palette(categories[category], n_colors=prompt_counts[category]).as_hex()
        for category in categories
    }

    # Assign colors to each prompt based on its category
    prompt_colors = {}
    for prompt in prompt_names:
        category = next(cat for cat in categories if prompt.startswith(cat) or prompt == cat)
        prompt_colors[prompt] = color_palettes[category].pop(0)

    # Create subplots
    fig = make_subplots(rows=1, cols=3, 
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=("<b>PCA</b>", "<b>UMAP</b>", "<b>t-SNE</b>"))

    # Add traces
    for df, col in zip([df_pca, df_umap, df_tsne], [1, 2, 3]):
        for prompt in prompt_names:
            fig.add_trace(
                go.Scatter3d(
                    x=df[df['Prompt'] == prompt]['Component 1'],
                    y=df[df['Prompt'] == prompt]['Component 2'],
                    z=df[df['Prompt'] == prompt]['Component 3'],
                    mode='markers',
                    marker=dict(color=prompt_colors[prompt], size=3),
                    name=prompt,
                    legendgroup=prompt,
                    showlegend=(col == 1)  # Only show legend in the first plot
                ),
                row=1, col=col
            )

    # This can be used to adjust the distance between titles and plots
    for i in fig['layout']['annotations']:
        i['y'] = 0.78
        i['font'] = dict(size=18)

    # Update layout for a better view
    fig.update_layout(
        width=1000,
        height=600,
        # title_text="Latent Mental Health Space Cartography",
        legend_title="<b>Prompt</b>",
        legend=dict(
            x=0.5,
            y=.20, # This can be used to adjust the distance between plots and legend
            xanchor='center',
            yanchor='top',
            orientation='h',
            font=dict(size=12),  # Adjust the font size of the legend to change the size of color markers
            itemsizing='constant' 
        ),
        font = dict(family="Helvetica"),
        margin=dict(t=50, b=50) # Reduce top and bottom margins if needed
    )

    # Remove axes 
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

    fig.show()

def load_results(file_path):
    '''
    Load results from a pickle file.
    '''
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results

# Function to extract relevant data from results
def extract_data(results, model_key):
    '''
    Extract relevant data from the results dictionary.
    '''
    score = results[model_key]['score']
    perm_scores = results[model_key]['permuted_scores']
    p_value = results[model_key]['p_value']
    return score, perm_scores, p_value

def bonf_correct_pvals(p_values):
    """
    Correct p-values using Bonferroni correction.
    
    Args:
    p_values (list): A list of p-values.
    
    Returns:
    list: An array of corrected p-values.
    """
    return smm.multipletests(p_values, method='bonferroni')[1]

# Function to create permutation histogram plots
def plot_permutation_histogram_plotly(perm_scores, model_score, p_value, title, color, fig, row, col, horiz_offset, show_yaxis_title, show_xaxis_title):
    '''
    Plot permutation histogram using Plotly.
    '''
    # Determine the annotation based on p-value
    if p_value < 0.016:
        p_value_annotation = '**'
        linecolor = 'red'
    elif p_value < 0.05:
        p_value_annotation = '*'
        linecolor = 'darkred'
    else:
        p_value_annotation = ''
        linecolor = 'darkslategrey'
    
    fig.add_trace(
        go.Histogram(
            x=perm_scores,
            nbinsx=45,
            marker_color=color,
            histnorm='probability density',
        ),
        row=row, col=col
    )
    fig.add_shape(
        type='line',
        x0=model_score, y0=0,
        x1=model_score, y1=6,
        line=dict(color=linecolor, width=3),
        row=row, col=col
    )
    fig.add_annotation(
        x=model_score + horiz_offset,
        y=5,  # Adjust as necessary
        text=f"AUC = {model_score:.2f}<br>p = {p_value:.3f}{p_value_annotation}",
        showarrow=False,
        row=row, col=col, font=dict(family="Helvetica", size=24)
    )
    fig.update_xaxes(title_text="ROC AUC score" if show_xaxis_title else "", row=row, col=col, range=[0.2, 0.9], dtick=0.1, title_font=dict(family="Helvetica", size=20), tickfont=dict(family="Helvetica", size=16))
    if show_yaxis_title:
        fig.update_yaxes(title_text="Probability density", row=row, col=col, title_font=dict(family="Helvetica", size=20), tickfont=dict(family="Helvetica", size=16))

# Function to create and show a 4x3 grid plot for all results, including the "Self" group
def create_combined_plot(model_names, display_names, file_groups, colors, group_titles, horiz_offsets, output_file, save=False):
    fig = make_subplots(rows=3, cols=4,
                        vertical_spacing=0.06)

    for col in range(4):  # Adjusted for 4 columns
        p_values = []
        results_data = []
        
        # Extract data and p-values for correction
        for row in range(3):
            file_path = file_groups[col][row]
            model_name = model_names[row]
            results = load_results(file_path)
            score, perm_scores, p_value = extract_data(results, model_name)
            p_values.append(p_value)
            results_data.append((score, perm_scores))

        # Correct p-values for multiple comparisons within each group
        corrected_p_values = bonf_correct_pvals(p_values)

        for row in range(3):
            score, perm_scores = results_data[row]
            p_value = corrected_p_values[row]
            color = colors[col][row]
            horiz_offset = horiz_offsets[col][row]
            show_yaxis_title = (col == 0)  # Only show y-axis title for the first column
            show_xaxis_title = (row == 2)
            plot_permutation_histogram_plotly(perm_scores, score, p_value,
                                              display_names[row], 
                                              color, fig, row + 1, col + 1, horiz_offset, show_yaxis_title, show_xaxis_title)
    
    # Add vertical labels for each group (TAT, RIT, IAPS, Self)
    for col, group_title in enumerate(group_titles):
        fig.add_annotation(
            x=(col + 0.27) / 3.6,  # Adjusted for 4 columns
            y=1.05,
            text=group_title,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=24, family="Helvetica", color="black"),
            textangle=0,
            
        )

    # Add row labels for each factor
    factor_labels = ['<b>Anxious-<br>Depression</b>', '<b>Compulsive Behavior<br> & Intrusive Thought</b>', '<b>Mood & Impulsivity</b>']
    row_positions = [0.93, 0.50, 0.01]
    for row, factor_label in enumerate(factor_labels):
        fig.add_annotation(
            x=-0.13,
            y=row_positions[row], #1 - (row + 0.14) / 2.2,
            text=factor_label,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=24, family="Helvetica", color="black"),
            textangle=-90
        )

    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=1200,
        height=900,  # Adjusted height for 4 rows
        plot_bgcolor='white',
        font=dict(family="Helvetica", size=12),
        margin=dict(l=135, r=50, b=50, t=50),
        title_font=dict(family="Helvetica", size=14)
    )

    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    if save:
        pio.write_image(fig, output_file)
    fig.show()

# Define file paths and model names 
tat_files = [
    'may29_All_LogReg_TAT_mh_1_30_perc_feat_results.pkl',
    'may29_All_LogReg_TAT_mh_2_30_perc_feat_results.pkl',
    'may29_All_LogReg_TAT_mh_3_30_perc_feat_results.pkl'
]

rit_files = [
    'may29_All_LogReg_RIT_mh_1_30_perc_feat_results.pkl',
    'may29_All_LogReg_RIT_mh_2_30_perc_feat_results.pkl',
    'may29_All_LogReg_RIT_mh_3_30_perc_feat_results.pkl'
]

iaps_files = [
    'may29_All_LogReg_IASP_mh_1_30_perc_feat_results.pkl',
    'may29_All_LogReg_IASP_mh_2_30_perc_feat_results.pkl',
    'may29_All_LogReg_IASP_mh_3_30_perc_feat_results.pkl',
]

self_files = [
    'jun6_LogReg_Self_mh_1_perc_feat_results.pkl',
    'jun6_LogReg_Self_mh_2_perc_feat_results.pkl',
    'jun6_LogReg_Self_mh_3_perc_feat_results.pkl',
]

file_groups = [tat_files, rit_files, iaps_files, self_files]  # Include the "Self" group
model_names = ['ML1', 'ML2', 'ML3']
display_names = ['Anxious-Depression', 'Compulsive Behavior & Intrusive Thought', 'Mood & Impulsivity']
group_titles = ["<b>TAT</b>", "<b>RIT</b>", "<b>IAPS</b>", "<b>Self</b>"]  # Include the "Self" group

# Define individual horizontal offsets for each plot, including the "Self" group
horiz_offsets = [
    [-0.22, -0.22, -0.22],  # TAT offsets
    [0.22, -0.22, -0.22],  # RIT offsets
    [0.22, -0.22, -0.22],  # IAPS offsets
    [-0.22, -0.22, -0.22]  # Self offsets
]

# Define individual colors for each plot, including the "Self" group
colors = [
    ['#fcb499', '#f7593f', '#940b13'],  # TAT colors
    ['#c1e6ba', '#62bb6d', '#006227'],  # RIT colors
    ['#bfd8ed', '#3f8fc5', '#08488e'],  # IAPS colors
    ['#d3bfff', '#9e9ac8', '#6a51a3']   # Self colors
]

output_file = "combined_permutation_histograms.png"

# Create and show the combined plot
create_combined_plot(model_names, display_names, file_groups, colors, group_titles, horiz_offsets, output_file, save=True)



def plot_permutation_histogram_plotly(perm_scores, model_score, p_value, title, color, fig, row, col, horiz_offset):
    fig.add_trace(
        go.Histogram(
            x=perm_scores,
            nbinsx=50,
            marker_color=color,
            name=title,
            histnorm='probability density',
        ),
        row=row, col=col
    )
    fig.add_shape(
        type='line',
        x0=model_score, y0=0,
        x1=model_score, y1=30,
        line=dict(color="Red", width=4),
        row=row, col=col
    )
    # Note: Adjusting y1 value to match the histogram height
    fig.add_annotation(
        x=model_score + horiz_offset,
        y=6,  # Adjust as necessary
        text=f"AUC = {model_score:.2f}<br>p = {p_value:.4f}**",
        showarrow=False,
        row=row, col=col,
        font=dict(family="Helvetica", size=18)
    )
    fig.update_xaxes(title_text="ROC AUC score", row=row, col=col, title_font=dict(family="Helvetica", size=20), tickfont=dict(family="Helvetica", size=18), range=[0.4, 0.85], dtick=0.1)
    fig.update_yaxes(title_text="Probability density", row=row, col=col, title_font=dict(family="Helvetica", size=20), tickfont=dict(family="Helvetica", size=18))
    # Update titles and layout within this function as necessary

# Create subplot figure with 2 columns
fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Valence</b>", "<b>Arousal</b>"))

# Valence plot
with open('VAL_results_jun21.pkl', 'rb') as f:
    valence_results = pickle.load(f)
valence_scores = valence_results['valence_above_median']['permuted_scores']
valence_model_score = valence_results['valence_above_median']['score']
valence_p_value = valence_results['valence_above_median']['p_value']

# Arousal plot
with open('ARO_results_jun21.pkl', 'rb') as f:
    arousal_results = pickle.load(f)
arousal_scores = arousal_results['arousal_above_median']['permuted_scores']
arousal_model_score = arousal_results['arousal_above_median']['score']
arousal_p_value = arousal_results['arousal_above_median']['p_value']

corrected_p_values = smm.multipletests([valence_p_value, arousal_p_value], alpha=0.05, method='bonferroni')[1]
valence_p_value = corrected_p_values[0]
arousal_p_value = corrected_p_values[1]

plot_permutation_histogram_plotly(valence_scores, valence_model_score, valence_p_value,
                                  "<b>Permuted ROC AUC - Valence</b>", '#7db8da', fig, 1, 1, -0.13)

plot_permutation_histogram_plotly(arousal_scores, arousal_model_score, arousal_p_value,
                                  "<b>Permuted ROC AUC - Arousal</b>", '#08488e', fig, 1, 2, 0.1)

fig.update_layout(
    # title_text='Permutation Histograms',
    showlegend=False,
    autosize=False,
    width=1000,
    height=500,
    font=dict(family="Helvetica")
)
fig.update_layout(plot_bgcolor='white')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', tickfont=dict(family="Helvetica"), )
fig.update_xaxes(dtick=0.1, row=1, col=1)
fig.update_xaxes(dtick=0.1, row=1, col=2)

for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(family="Helvetica", size=24) # A little wonky cuz size changes the AUC/p and the title size

fig.show()



def create_histograms(data, output_file, save=False):
    # Define the questionnaire scores and their corresponding colors
    questionnaire_scores = {
        'ZSDS_Total': '#B2DF8A',  # Depression
        'GAD_Total': '#FB9A99',  # Generalized Anxiety
        'BIS_Total': '#E31A1C',  # Impulsivity
        'OLIFE_Total': '#FF7F00',  # Schizotypy
        'AES_Total': '#1F78B4',  # Apathy
        'AUDIT_Total': '#A6CEE3',  # Alcoholism
        'OCI_Total': '#FDBF6F',  # OCD
        'LSAS_Total': '#CAB2D6',  # Social Anxiety
        'EAT_Total': '#33A02C',  # Eating Disorders
    }

    # Create a subplot figure with 3 columns
    fig = make_subplots(rows=3, cols=3, subplot_titles=[""]*9)  # Empty titles

    # Create histograms for each questionnaire score
    for i, (score, color) in enumerate(questionnaire_scores.items()):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.add_trace(
            go.Histogram(
                x=data[score],
                marker_color=color,
                name=score,
                histnorm='probability density'
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text=f"<b>{score}</b>", row=row, col=col, title_font=dict(family="Helvetica", size=18))

    # Update layout to remove individual y-axis titles and add a common y-axis title
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=1000,
        height=800,
        font=dict(family="Helvetica"),
        plot_bgcolor='white'
    )
    
    # Remove individual y-axis titles
    for i in range(2, 10):
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', row=(i-1)//3 + 1, col=(i-1)%3 + 1, title='')

    # Add centered y-axis label using annotations
    fig.update_layout(
        annotations=[
            dict(
                text="<b>Probability density</b>",
                x=-0.09,  # Adjust x to center the label
                xref='paper',
                y=0.5,
                yref='paper',
                showarrow=False,
                font=dict(size=18, family="Helvetica", color='black'),
                textangle=-90
            )
        ]
    )

    if save:
        pio.write_image(fig, output_file)

    fig.show()

d = pd.read_csv('dp_data.csv', index_col=0)
create_histograms(d, 'questionnaire_score_histograms.png')
