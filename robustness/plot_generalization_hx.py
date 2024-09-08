import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D


for dataset in ['complete', 'random']:
    for split in ['train', 'test']:
        # Read the CSV file
        df1 = pd.read_csv(dataset + '_0.01_seedvalue_2020_' + split + '.csv')
        df2 = pd.read_csv(dataset + '_0.01_seedvalue_2024_' + split + '.csv')
        df3 = pd.read_csv(dataset + '_0.01_seedvalue_2333_' + split + '.csv')
        df4 = pd.read_csv(dataset + '_0.01_seedvalue_1314_' + split + '.csv')
        df5 = pd.read_csv(dataset + '_0.01_seedvalue_512_' + split + '.csv')
        df =  pd.concat([df1,df2,df3,df4,df5],axis=1)

        labels = ['SGD', 'Adam', 'Ours']
        sgd,adam,ours = df.iloc[:,[0,3,6,9,12]].mean(axis=1)[49:], df.iloc[:,[1,4,7,10,13]].mean(axis=1)[49:], df.iloc[:,[2,5,8,11,14]].mean(axis=1)[49:]
        sgdstd,adamdtd,oursstd = df.iloc[:,[0,3,6,9,12]].std(axis=1)[49:], df.iloc[:,[1,4,7,10,13]].std(axis=1)[49:], df.iloc[:,[2,5,8,11,14]].std(axis=1)[49:]
        #sgd,adam,ours = df1['SGD'][49:], df1['Adam'][49:], df1['Ours'][49:]
        #print(sgd.head(10))
        #print(ours.head(10))

        # Calculate upper and lower bounds using standard deviation
        sgdupper,adamupper,oursupper = sgd+sgdstd,adam+adamdtd,ours+oursstd
        sgdlower,adamlower,ourslower = sgd-sgdstd,adam-adamdtd,ours-oursstd

        results = [sgd, adam, ours]
        resultsstd = [sgdstd,adamdtd,oursstd]
        resultsupper = [sgdupper,adamupper,oursupper]
        resultsslower = [sgdlower,adamlower,ourslower]
        #colors = ['lavenderblush','cornsilk','aliceblue']
        colors1 = ['blue','orange','green']
        colors = ['lightskyblue','lightsalmon','lightgreen']



        x = [i for i in range(50, 101)]
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, label in enumerate(labels):
            #ax.errorbar(x, results[i], yerr=resultsstd[i], label=label, fmt='-', capsize=5, elinewidth=2, markeredgewidth=2, errorevery=1)
            plt.plot(x, results[i], color=colors1[i],label=label)
            plt.fill_between(x, resultsslower[i], resultsupper[i], color=colors[i], alpha=0.5)
        if split=='test' and dataset=='random':
            plt.ylim(0.996,1.014)
        if split=='test' and dataset=='complete':
            plt.ylim(0.995,1.030)
        if split=='train' and dataset=='random':
            plt.ylim(0.84,1.02)
        if split=='train' and dataset=='complete':
            plt.ylim(0.3,1.1)


        ax.set_xlabel('Epoch', fontweight='bold', fontsize='x-large')
        ax.set_ylabel('Loss', fontweight='bold', fontsize='x-large')

        # Bold tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize('large')

        legend = ax.legend(bbox_to_anchor=(0.5, 0), loc='lower center', ncol=3)

        for text in legend.get_texts():
            text.set_fontweight('bold')

        # Bold plot edges
        for spine in ax.spines.values():
            spine.set_linewidth(2)


        plt.tight_layout()
        # Save the figure as a PDF
        pdf_path = './robustness/generalization_' + dataset + '_' + split + '.pdf'
        plt.savefig(pdf_path, format='pdf')

        plt.show()