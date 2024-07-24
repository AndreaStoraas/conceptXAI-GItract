import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#Create a dict with the concept and a list of 
#TCAV mean + std, and whether the concept was significant or not
#If not significant, the values are just set to 0
results_class0 = {
    'Polyp': [0, 0, True],
    'Instrument': [0, 0, False],
}

results_class1 = {
    'Polyp': [1, 0, True],
    'Instrument': [0, 0, False],
}

num_concepts = 2
bar_width = 0.35
# create location for each bar. scale by an appropriate factor to ensure 
# the final plot doesn't have any parts overlapping
index = np.arange(num_concepts) * bar_width


fig, ax = plt.subplots(1,2, figsize=(16,8))
#Values for healthy (class 0) images
plot_concepts0 = []
TCAV_means0 = []
TCAV_std0 = []
TCAV_significance0 = []
bar_x0 = []
for i, [concept_name, vals] in enumerate(results_class0.items()):
    #The TCAV mean is vals[0], the TCAV std is vals[1]
    TCAV_means0.append(vals[0])
    TCAV_std0.append(vals[1])
    bar_x0.append(i * bar_width)
    plot_concepts0.append(concept_name)
    TCAV_significance0.append(vals[2])
text_sequence0=[]
for j in TCAV_significance0:
    if j:
        text_sequence0.append('*')
    else:
        text_sequence0.append(' ')


plot_concepts1 = []
TCAV_means1 = []
TCAV_std1 = []
TCAV_significance1 = []
bar_x1 = []
for i, [concept_name, vals] in enumerate(results_class1.items()):
    #The TCAV mean is vals[0], the TCAV std is vals[1]
    TCAV_means1.append(vals[0])
    TCAV_std1.append(vals[1])
    bar_x1.append(i * bar_width)
    plot_concepts1.append(concept_name)
    TCAV_significance1.append(vals[2])
text_sequence1=[]
for j in TCAV_significance1:
    if j:
        text_sequence1.append('*')
    else:
        text_sequence1.append(' ')

#Now, to the subplots themselves:
#https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-demo-py
ax[0].bar(bar_x0, TCAV_means0, bar_width, yerr=TCAV_std0, label=plot_concepts0, 
    color=['#001765','#FF8100']) #Novo colors
    
ax[0].set_title('No disease',fontsize=32)
ax[0].set_xticks(bar_x0)
ax[0].set_xticklabels(plot_concepts0, rotation = 75,fontsize=32)
ax[0].set_ylim((0,1.13))
#Since all concepts are significant, I avoid the star:
for i in range(2):
    ax[0].text(bar_x0[i]-0.015,TCAV_means0[i]+0.01,text_sequence0[i],fontdict = {'weight': 'bold', 'size': 32})

#Disease images with polyps:
ax[1].bar(bar_x1, TCAV_means1, bar_width, yerr=TCAV_std1, label=plot_concepts1, 
    color=['#001765','#FF8100'])
ax[1].set_title('Disease',fontsize=32)
ax[1].set_xticks(bar_x1)
ax[1].set_xticklabels(plot_concepts1, rotation = 75,fontsize=32)
ax[1].set_ylim((0,1.13))
#Add star on top of bars for significant concepts:
for i in range(2):
    ax[1].text(bar_x1[i]-0.015,TCAV_means1[i]+0.01,text_sequence1[i],fontdict = {'weight': 'bold', 'size': 32})


ax[0].set_ylabel('TCAV score',fontsize=32)
ax[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=32)
# Hide x labels and tick labels for top plots and y ticks for right plots.
ax[1].label_outer()

#Shrink the space between the subplots:
plt.subplots_adjust(wspace=0.1)

plt.savefig('PlotTCAVscores_CroppedRepresentativetest_LastConvLayer.png', bbox_inches = 'tight')
