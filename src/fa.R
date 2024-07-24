library(polycor)
library(psych)
library(ggplot2)
library(tidyr)
library(RColorBrewer)
library(dplyr)

#####################
### Main Analysis ###
#####################
data <- read.csv("fa_items_only_March4.csv", row.names=1)
desired_order <- c('AUDIT', 'AES', 'ZSDS', 'EAT', 'GAD', 'BIS', 'OCI', 'OLIFE', 'LSAS_Mean')
quest_fullnames <- c("Alcoholism", "Apathy", "Depression", "Eating Disorders", "Generalized Anxiety", "Impulsivity", "OCD", "Schizotypy", "Social Anxiety")

fa <- fa(r=hetcor(data)$cor, nfactors=3, n.obs = nrow(data), rotate="oblimin", fm='ml', scores='regression') # Taken directly from Rouault et al.'s code

# Factor scores per participant
factorscores <- factor.scores(x=data, f=fa)
factorscores <- factorscores$score # Save factor scores : write.csv(factorscores, file='factorscores.csv')

# Factor loadings
loadings <- fa$loadings

# Factor loadings dataframe
specific_loadings <- data.frame(loadings[, c("ML1", "ML2", "ML3")])

# New column based on questionnaire prefix
specific_loadings$quest <- gsub("_[0-9]+|([A-Za-z])[0-9]+", "\\1", rownames(specific_loadings))

# Change order of questionnaires
specific_loadings$quest <- factor(specific_loadings$quest, levels = desired_order)
specific_loadings <- specific_loadings[order(specific_loadings$quest), ]
lookup_table <- setNames(quest_fullnames, desired_order)
specific_loadings$quest_full <- sapply(specific_loadings$quest, function(code) lookup_table[code])

# Add unique color for each questionnaire type 
unique_values <- unique(specific_loadings$quest)
custom_palette <- brewer.pal(n = length(unique_values), name = "Paired")
color_mapping <- setNames(custom_palette, unique_values)
specific_loadings$colors <- color_mapping[specific_loadings$quest]



########################################
### Scree plot for number of factors ###
########################################
# Create a data frame from factor analysis results
eigenvalues <- data.frame(Factor = seq_along(fa$values), Eigenvalue = fa$values)
eigenvalues$Color <- ifelse(
  eigenvalues$Factor == 1, "#FFE4E1",
  ifelse(eigenvalues$Factor == 2, "#FFB6C1",
         ifelse(eigenvalues$Factor == 3, "#DDA0DD", "gray"))
)

ggplot(eigenvalues, aes(x = Factor, y = Eigenvalue, fill = Color)) +
  geom_bar(stat = "identity") +
  scale_fill_identity() +  
  labs(title = "Scree Plot", x = "Factor number", y = "Eigenvalue") +
  scale_y_continuous(limits = c(0, 50)) +
  scale_x_continuous(breaks = seq(0, 60, by = 10)) +
  coord_cartesian(ylim = c(0, 30)) +
  theme_minimal() +
  theme(
    plot.title = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(family="Helvetica", face="bold", size=18, margin = margin(t = 5, r = 0, b = 0, l = 0)),
    axis.title.y = element_text(family="Helvetica", face="bold", size=18, margin = margin(t = 0, r = 5, b = 0, l = 0)),
    axis.text.x = element_text(family="Helvetica", margin = margin(t = 0, r = 0, b = 0, l = 0)),
    axis.text.y = element_text(family="Helvetica", margin = margin(t = 0, r = 0, b = 0, l = 0)),
    axis.line = element_line(color = "black"),
    panel.background = element_rect(fill = "#F5F5F5", color = NA),
    plot.background = element_rect(fill = "#F5F5F5", color = NA)
  )

#################################
### Factor loadings multiplot ###
#################################
# Save the default graphical parameters
default_par <- par(no.readonly = TRUE)
# Set up layout with space for the legend
layout(matrix(c(1, 2, 3, 4), ncol = 1), 
       heights = c(1, 1, 1, 0.4))
# Set margins for the individual plots
par(oma = c(4, 4, 2, 1)) # bottom, left, top, right
par(family='Helvetica', font.main=1, mar = c(2, 4, 4, 2) + 0.1)
# Plot 1: Anxiety
barplot(specific_loadings$ML1, main = "", col=specific_loadings$colors, ylim = c(-0.5, 0.5))
title(main="Anxious-Depression", font.main=2, col.main='#000000', cex.main=2)
# Plot 2: Compulsive Behavior and Intrusive Thought
barplot(specific_loadings$ML2, main = "", col=specific_loadings$colors, ylim = c(-0.5, 0.5))
title(main="Compulsive Behavior & Intrusive Thought", font.main=2, col.main='#000000',cex.main=2)
# Plot 3: Mood and Impulsivity
barplot(specific_loadings$ML3, main = "", col=specific_loadings$colors, ylim = c(-0.5, 0.5))
title(main="Mood & Impulsivity", font.main=2, col.main='#000000', cex.main=2)
# Add a common y-axis label
mtext("Respective Factor Loadings", side = 2, line =0.01, outer = TRUE, cex = 0.8, font=2)
mtext("Questionnaire Items", side = 1, line = 1, outer = TRUE, cex = 0.8, font = 2)
# Reset to single plot layout and allow plotting outside plot region
par(mar = c(0, 0, 0, 0), xpd = TRUE)
# Add the legend at the bottom
plot.new()
legend("center", 
       legend = unique(specific_loadings$quest_full), 
       fill = unique(specific_loadings$colors), 
       horiz = FALSE, cex = 1.3, text.width = 0.25, 
       x.intersp = 0.2, y.intersp = 1, 
       inset = c(0, 3), ncol=3)
# Restore the default graphical parameters
par(default_par)