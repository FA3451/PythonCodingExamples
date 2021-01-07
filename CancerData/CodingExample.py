#!/usr/bin/env python3

import collections
import subprocess
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import scipy
import pandas as pd
import sys
from scipy import stats
import math
import statistics
import statsmodels.stats.multitest

exitronsOriginDir = "exitronsOrigin.modfied"
#exitronsOriginDir = "exitrons.origin"

def logfunc(x, a, b):
    return a*np.log(x) + b


def make_summary(sampleDir, dataDir, genStandardOut): #gets summary of the files as standard output, that can be redirected
    sample_dic = {}
    ###loading all wanted data into a dic in a dic of each sample
    for sample in open(sampleDir):
        sample = sample.strip()
        sample_dic.update({sample: {
            "numOfSamples": (int(subprocess.Popen("ls -1 {}{}/psi/ | wc -l".format(dataDir, sample), stdout=subprocess.PIPE, shell=True).communicate()[0]) - 1) / 2,
            "numEI": len([1 for elem in open("{}{}/exitrons.bed".format(dataDir, sample))]),
            "devBy3": len([1 for elem in open( "{}{}/exitrons.bed".format(dataDir, sample)) if (int(elem.split()[2]) - int(elem.split()[1]) + 1) % 3 == 0]),
            "numGenes": len(set([elem.split()[3] for elem in open("{}{}/exitrons.info".format(dataDir, sample)) if not elem.startswith("#")])),
            "numMore10EI": len([1 for val in collections.Counter([elem.split()[3] for elem in open("{}{}/exitrons.info".format(dataDir, sample)) if not elem.startswith("#")]).values() if val >= 10])
                                    }})

    ###read gathered data and present it as standard output
    if genStandardOut:
        print("#name\tnumOfEI\tdevby3\tnumGenes\tCentitrons")
        for entry in sample_dic.keys():
            print('\t'.join([entry, str(sample_dic[entry]["numEI"]), str(sample_dic[entry]["devBy3"]),
                             str(sample_dic[entry]["numGenes"]), str(sample_dic[entry]["numMore10EI"]),
                             str(sample_dic[entry]["numOfSamples"])]))
    return sample_dic


def generate_EI_sample_plot(sampleDir, dataDir, outDir):
    sample_dic = make_summary(sampleDir, dataDir, False)

    #assigning X and y values for fits
    X = [sample_dic[elem]["numOfSamples"] for elem in sample_dic.keys()]
    y = [sample_dic[elem]["numEI"] for elem in sample_dic.keys()]

    #excluding STAD as an 'outlier', when plotting
    X_no_STAD = [sample_dic[elem]["numOfSamples"] for elem in sample_dic.keys() if
               elem != "TCGA-STAD"]  # excluding STAD for fit
    y_no_STAD = [sample_dic[elem]["numEI"] for elem in sample_dic.keys() if elem != "TCGA-STAD"]

    # linear regression model from sklearn, log fit from scipy
    reg = LinearRegression()
    lin_model = reg.fit(np.array(X_no_STAD).reshape((-1, 1)), np.array(y_no_STAD))
    poptExp, pcovExp = scipy.optimize.curve_fit(logfunc, X_no_STAD, y_no_STAD, p0=[1, 1])

    # plotting everything; if loop adds labels
    fig, ax = plt.subplots()
    ax.scatter(X, y, marker="o", s=[sample_dic[elem]["devBy3"] for elem in sample_dic.keys()])
    plt.ylabel("Number of exitrons")
    plt.xlabel("Number of samples")
    for k in sample_dic.keys():
        ax.annotate(''.join(list(k)[5:]), (sample_dic[k]["numOfSamples"], sample_dic[k]["numEI"]), textcoords="data")
    n = np.linspace(6, 50, 100)
    plt.plot(n, lin_model.coef_ * n + lin_model.intercept_, linestyle="solid", c="red")
    plt.plot(n, poptExp[0] * np.log(n) + poptExp[1], linestyle="dashed", c="green")
    plt.savefig("{}EIvsSampleNumber.png".format(outDir))
    plt.close()


def generate_PWM(dataSet, sampleFileDir, outDir, showPlot):
    ##identification of exclusive and non exclusive Exitrons:
    #naming of the directories according to default
    #TODO: add directories to arg paser
    set_of_all_EI_except_STAD = set(
        [EIline.split()[3] for samp in open(sampleFileDir)
         for EIline in open("/home/felix/Documents/CancerProject/DataSets/{}/exitrons.bed".format(samp.strip()))
         if samp.strip() != "TCGA-{}".format(dataSet)])

    list_exclusive_dataSet = [elem.split()[3] for elem in
                         open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/exitrons.bed".format(dataSet)) if
                         elem.split()[3] not in list(set_of_all_EI_except_STAD)]

    ##generating a dictionary with 8 primary keys: for each binding with both directions with exclusive or not
    dic_PWM = {fraction + spec: {"A": [0] * 9, "C": [0] * 9, "G": [0] * 9, "T": [0] * 9} for spec in
              ["_5p_notExcl", "_5p_Excl", "_3p_notExcl", "_3p_Excl"] for fraction in ["GT-AG_U2", "GC-AG_U2"]}

    # for fraction in ["GT-AG_U2", "GC-AG_U2"]: #fractions are not needed, as in the features file of the spice site independent from the splice site all exitrons are present and only scored according to the file label
    seen5p, seen3p = [], [] #p stands for prime
    for data_line in open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/features/GT-AG_U2.txt".format(dataSet)): #fixed directory. may add too argparser
        single_letter_site = data_line.split()[1][4]
        if data_line.split()[0].split(":")[2] == "+":
            identifier5p = data_line.split()[0].split(":")[0] + data_line.split()[0].split(":")[1].split("-")[0]
            identifier3p = data_line.split()[0].split(":")[0] + data_line.split()[0].split(":")[1].split("-")[1]
        else:
            identifier5p = data_line.split()[0].split(":")[0] + data_line.split()[0].split(":")[1].split("-")[1]
            identifier3p = data_line.split()[0].split(":")[0] + data_line.split()[0].split(":")[1].split("-")[0]

        #set exclusive identifier
        excl = "Excl" if data_line.split()[0] in list_exclusive_dataSet else "notExcl"

        #generate PWM dictionaries by pushing a list of already seen sites through to exclude doubles
        if identifier5p not in seen5p:
            for i, base in enumerate(data_line.split()[1]):
                dic_PWM["G{}-AG_U2_5p_{}".format(single_letter_site, excl)][base][i] += 1
            seen5p.append(identifier5p)
        if identifier3p not in seen3p:
            for i, base in enumerate(data_line.split()[2]):
                dic_PWM["G{}-AG_U2_3p_{}".format(single_letter_site, excl)][base][i] += 1
            seen3p.append(identifier3p)
    # by now the present weight matrix work, as it just runs through the program and checks all splice sites.
    # splice sites being used by two different exitrons are counted double.
    # Adjustment: implement a list, that is being pushed through together with the counts to see, whether this site has already been counted.
    # done

    # plotting, stacked bar plot
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    fig.suptitle("all PWM plotted (absolute)")
    spliceRange = range(0, 9, 1)
    for subset, ax in zip(dic_PWM.keys(), axes.flatten()):
        pltA = ax.bar(spliceRange, dic_PWM[subset]["A"], width=0.35)
        pltC = ax.bar(spliceRange, dic_PWM[subset]["C"], width=0.35, bottom=dic_PWM[subset]["A"])
        pltG = ax.bar(spliceRange, dic_PWM[subset]["G"], width=0.35,
                      bottom=[x + y for x, y in zip(dic_PWM[subset]["A"], dic_PWM[subset]["C"])])
        pltT = ax.bar(spliceRange, dic_PWM[subset]["T"], width=0.35,
                      bottom=[x + y + z for x, y, z in
                              zip(dic_PWM[subset]["A"], dic_PWM[subset]["C"], dic_PWM[subset]["G"])])
        ax.legend((pltA[0], pltC[0], pltG[0], pltT[0]), ('A', 'C', 'G', 'T'), loc="upper right")
        ax.set_title(subset.split("/")[-1])
    plt.savefig("{}PWMof{}.png".format(outDir, dataSet))
    if showPlot:
        plt.show()
    plt.close()


def origin_of_EI(dataSet, sampleFileDir, outDir, withNames, withoutNames, tissue, isExclusive):
    if tissue == "normaltumor":
        tissue_extracting = "normal;tumor"
    else:
        tissue_extracting = tissue
    ##identification of exclusive and non exclusive Exitrons:
    #naming of the directories according to default
    numberOfSamplesTissue = int(float(subprocess.Popen(
        "ls -l /home/felix/Documents/CancerProject/DataSets/TCGA-{}/psi/ | wc -l".format(dataSet),
        stdout=subprocess.PIPE, shell=True).communicate()[0].decode("utf-8").strip()) / 2-1)

    #manually modify, as it has special condition
    if dataSet == "LUAD":
        numberOfSamplesTissue -= 1

    #split up in lists depending on exclusivity
    if isExclusive:
        set_all_EI_except_dataSet = set([exitron_line.split()[3] for samp in open(sampleFileDir)
             for exitron_line in open("/home/felix/Documents/CancerProject/DataSets/{}/exitrons.bed".format(samp.strip()))
             if samp.strip() != "TCGA-{}".format(dataSet)])

        list_exclusive_dataSet = [elem.split()[3] for elem in
                             open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/exitrons.bed".format(dataSet)) if
                             elem.split()[3] not in list(set_all_EI_except_dataSet)]
        exclusivity = "Excl"

    else:
        #actually it is not exclusive in this case; naming is just according to default setting
        list_exclusive_dataSet = [exitron_line.split()[3] for exitron_line in open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/exitrons.bed".format(dataSet))]
        exclusivity = "NotExcl"

    #std out counts
    counts = collections.Counter([line_origin.split()[1] for line_origin in open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/{}".format(dataSet, exitronsOriginDir)) if line_origin.split()[0] in list_exclusive_dataSet])
    with open("{}numbersIn{}{}".format(outDir, dataSet, exclusivity), "w") as fout:
        [fout.write(key + ":\t" + str(counts[key]) + "\n") for key in counts.keys()]
        fout.close()

    # generate a dictioary with these three categories. and in each category a dic with the different patients
    dict_of_labels = {"normal": {}, "tumor": {}, "normal;tumor": {}}
    dict_EI_origin = {"normal": {}, "tumor": {}, "normal;tumor": {}}
    for line_origin in open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/{}".format(dataSet, exitronsOriginDir)):
        if line_origin.split()[0] in list_exclusive_dataSet:
            # this dict contains the positioning and the protein name
            dict_of_labels[line_origin.split()[1]].update({line_origin.split()[0]: subprocess.Popen("grep {} /home/felix/Documents/CancerProject/DataSets/TCGA-{}/exitrons.info | cut -f4".format(line_origin.split()[0], dataSet), stdout=subprocess.PIPE, shell=True).communicate()[0].decode("utf-8").strip()})
            #dependency on tissue
            if line_origin.split()[1] == "normal":
                for num, splicingNumber in enumerate(line_origin.split()[2:2+numberOfSamplesTissue]):
                    try:
                        dict_EI_origin[line_origin.split()[1]][num].append(splicingNumber)
                    except KeyError:
                        dict_EI_origin[line_origin.split()[1]].update({num: [splicingNumber]})
            elif line_origin.split()[1] == "normal;tumor":
                for num, splicingNumber in enumerate(line_origin.split()[2:]):
                    try:
                        dict_EI_origin[line_origin.split()[1]][num].append(splicingNumber)
                    except KeyError:
                        dict_EI_origin[line_origin.split()[1]].update({num: [splicingNumber]})
            else:   #tumor
                for num, splicingNumber in enumerate(line_origin.split()[2+numberOfSamplesTissue:]):
                    try:
                        dict_EI_origin[line_origin.split()[1]][num].append(splicingNumber)
                    except KeyError:
                        dict_EI_origin[line_origin.split()[1]].update({num: [splicingNumber]})


#write one file with names and one without
    if withNames:
        with open("{}{}In{}{}wNames.out".format(outDir, exclusivity, tissue, dataSet), "w") as fout:
            for protein_pos in dict_of_labels[tissue_extracting].keys():
                fout.write(protein_pos + "\t")
            fout.write("\n")
            for protein_name in dict_of_labels[tissue_extracting].values():
                fout.write(protein_name + "\t")
            fout.write("\n")
            for elements in dict_of_labels[tissue_extracting].values():
                fout.write("\t".join(elements) + "\n")
            fout.close()

    if withoutNames:
        with open("{}{}In{}{}noN.out".format(outDir, exclusivity, tissue, dataSet), "w") as fout:
            for elem in dict_of_labels[tissue_extracting].values():
                fout.write("\t".join(elem) + "\n")
            fout.close()


def normalize_data(dirMeta, NNDir, plusDir, tis):
    #normalization overwrites previous noName file
    #test normalizing data
    if tis == "normaltumor":
        tissue = ["normal", "tumor"]
    else:
        tissue = [tis]

    #normalization
    list_norm_length = []
    for sample_line in open(plusDir):
        if sample_line.split()[0] in tissue:
            line_res = subprocess.Popen("grep {} {}".format(sample_line.split()[1].split("/")[-2], dirMeta),
                                       stdout=subprocess.PIPE, shell=True).communicate()[0].decode(
                "utf-8").strip().split("\t")
            list_norm_length.append(int(line_res[7]) * int(line_res[9]))
    factor_for_norm = [elem / min(list_norm_length) for elem in list_norm_length]

    #write temporary file with the normalized values
    with open(NNDir+".temp", "w") as fout:
        for i, no_name_line in enumerate(open(NNDir)):
            for NNelem in no_name_line.split():
                fout.write(str(round(int(NNelem) / float(factor_for_norm[i]), 1)) + "\t")
            fout.write("\n")
        fout.close()

    #overwrite old NoName file, remove temp
    subprocess.call("rm {}".format(NNDir), shell=True)
    subprocess.call("mv {} {}".format(NNDir+".temp", NNDir), shell=True)


def investigation(inFilesDir, tissue, dataSet, outDir, showPlots, cutoffValue, numberOfClusters, isExclusive, minEI):

    # parameter preparation
    if tissue == "normal;tumor":
        tissue = "normaltumor"
    if isExclusive:
        exclusivity = "Excl"
    else:
        exclusivity = "NotExcl"

    #prepare data
    working_data = [[float(elem) for elem in ei_data_line.strip().split("\t")] for ei_data_line in open("{}{}In{}{}noN.out".format(inFilesDir,exclusivity, tissue, dataSet))]

    #check and exit if too less exitrons
    if len(working_data) == 0 or len(working_data[0]) <= minEI:
        print("dataset {} {} {} has less than {} EI".format(dataSet, tissue, exclusivity, minEI))
        sys.exit()


    #number of tissues needed later on
    number_of_samples_tissue = int(float(subprocess.Popen(
        "ls -l /home/felix/Documents/CancerProject/DataSets/TCGA-{}/psi/ | wc -l".format(dataSet),
        stdout=subprocess.PIPE, shell=True).communicate()[0].decode("utf-8").strip()) / 2 - 1)
    if dataSet == "LUAD":
        number_of_samples_tissue -= 1

    #diagnostic plot for percentage of described model variance (cumulative)
    resultsPCA = []
    with open("{}ComponensRepresentData{}{}{}".format(outDir, tissue, dataSet, exclusivity), "w") as fout:
        fout.write("#Compinents\tPercentageDescription")
        if number_of_samples_tissue > 25:   #max for PCA
            num_of_runs = 25
        else:
            num_of_runs = number_of_samples_tissue
        if len(working_data[0]) < num_of_runs:
            num_of_runs = len(working_data[0])
        for number_of_PCA_comp in range(1, num_of_runs):
            pca = PCA(n_components=number_of_PCA_comp)
            pca.fit_transform(X=working_data)
            fout.write(str(number_of_PCA_comp) + '\t' + str(round(np.sum(pca.explained_variance_ratio_)*100, 1)) + '\n')
            resultsPCA.append(round(np.sum(pca.explained_variance_ratio_), 5))
        fout.close()

    # generates the plot of how much data is being described by the PCA
    plt.plot(range(1, num_of_runs), resultsPCA)
    plt.xlabel("Number of principal components")
    plt.ylabel("Data description")
    plt.suptitle("PCA data desription")
    plt.savefig("{}dataDescriptionPCA{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    if showPlots:
        plt.show()
    plt.close()

    #scatter plot with two components indices. Indicates, which exitron contribute how much to the PC
    pca = PCA(n_components=2)
    pca.fit_transform(X=working_data)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle("Indices of Principle 2 components")
    for i, ax in zip(range(0, 2), axes.flatten()):
        ax.scatter(range(0, len(pca.components_[i])), pca.components_[i])
        ax.set_title("PC{}({}%)".format(str(i+1), str(round(pca.explained_variance_ratio_[i]*100, 1))))
        ax.set_xlabel("Exitron ID")
        ax.set_ylabel("Indices if PC" + str(i+1))
    plt.savefig("{}DistributionOf2Components{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    if showPlots:
        plt.show()
    plt.close()

    #scatter plot of the normal and tissue samples with PC1 and PC2 at the axes
    res_PCA_1, res_PCA_2 = [], []
    for patient_index, patient in enumerate(open("{}{}In{}{}noN.out".format(inFilesDir, exclusivity, tissue, dataSet))):
        temp_PCA_1, temp_PCA_2 = [], []
        for EI_count, PCA_elem_1, PCA_elem_2 in zip(patient.split(), pca.components_[0], pca.components_[1]):
            #dealing with nans
            if EI_count == "nan":
                EI_count = 0
            else:
                EI_count = float(EI_count)
            temp_PCA_1.append(EI_count * PCA_elem_1)
            temp_PCA_2.append(EI_count * PCA_elem_2)
        res_PCA_1.append(sum(temp_PCA_1))
        res_PCA_2.append(sum(temp_PCA_2))  # TODO: maybe rewrite as (2) list comprehension(s)

    plt.scatter(res_PCA_1, res_PCA_2)
    for i in range(0, len(res_PCA_1)):
        plt.annotate(i, (res_PCA_1[i], res_PCA_2[i]))
    plt.savefig("{}PCAplot2comp{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    plt.xlabel("PCA1({}%)".format(round(pca.explained_variance_ratio_[0] * 100, 1)))
    plt.ylabel("PCA2({}%)".format(round(pca.explained_variance_ratio_[1] * 100, 1)))
    plt.suptitle("Scatter plot of values in 2D")
    if showPlots:
        plt.show()
    plt.close()


    ### 3 principle components plot

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X=working_data)
    principalDf = pd.DataFrame(data=principalComponents)


    def pca_value(component, patient):  #helping function for task below
        return sum([0 if EI_count == "nan" else float(EI_count) * pca_element for EI_count, pca_element in zip(patient.split(), component)])

    #get results of the Principle Componenta Analysis for 3 components
    res_PCA_1, res_PCA_2, res_PCA_3 = [], [], []
    for patient_num, patient in enumerate(open("{}{}In{}{}noN.out".format(inFilesDir, exclusivity, tissue, dataSet))):
        res_PCA_1.append(pca_value(pca.components_[0], patient))
        res_PCA_2.append(pca_value(pca.components_[1], patient))
        res_PCA_3.append(pca_value(pca.components_[2], patient))


    # 3d plot of the first 3 principal components
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")
    for i in range(0, len(res_PCA_1)):
        ax.scatter3D(res_PCA_1[i], res_PCA_2[i], res_PCA_3[i], c="blue", s=50, marker="${}$".format(str(i + 1)))  # ["${}$".format(i) for i in range(0, len(resPCA1))])
    ax.set_xlabel("PC1({}%)".format(round(pca.explained_variance_ratio_[0] * 100, 1)))
    ax.set_ylabel("PC2({}%)".format(round(pca.explained_variance_ratio_[1] * 100, 1)))
    ax.set_zlabel("PC3({}%)".format(round(pca.explained_variance_ratio_[2] * 100, 1)))
    ax.set_title("Scatter plot values in 3D plot")
    for i in range(0, len(res_PCA_1)):
        plt.annotate(i, (res_PCA_1[i], res_PCA_2[i]))
    plt.savefig("{}Plot3PComponents{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    if showPlots:
        plt.show()
    plt.close()


    # this plot shows, that the principle component give rise to the importance of only a few genes.
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    for i, ax in zip(range(0, 3), axes.flatten()):
        ax.scatter(range(0, len(pca.components_[i])), pca.components_[i])
        ax.set_title("PC{}({}%)".format(i+1, round(pca.explained_variance_ratio_[i] * 100, 1)))
        ax.set_xlabel("Exitron ID")
        ax.set_ylabel("Index of PC")
    plt.savefig("{}DistributionOf3Components{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    if showPlots:
        plt.show()
    plt.close()


    # which ones those are, shall be identified in the following task
    # magic number can be changed // for mn=0.1 it is 13 values(genes)

    list_of_big_changes = [i for i, elem in enumerate(pca.components_[0]) if abs(elem) > cutoffValue]

    #get discriptive information of first lines
    first_line = open("{}{}In{}{}wNames.out".format(inFilesDir, exclusivity, tissue, dataSet)).readline()
    second_line = open("{}{}In{}{}wNames.out".format(inFilesDir, exclusivity, tissue, dataSet)).readlines()[1]

    #write a list of the genes, that heavily change
    with open("{}ResultsWithHighImpactPC1{}{}{}.txt".format(outDir, tissue, dataSet, exclusivity), "w") as fout:
        fout.write("num\tvalue\tposition\tproteinName\tCutoffValue " + str(cutoffValue) + "\n")
        for big_change in list_of_big_changes:
            fout.write("\t".join([str(big_change), str(round(pca.components_[0][big_change],3)), first_line.split()[big_change], second_line.split()[big_change]]) + "\n")
        fout.close()

    #wcss: within-cluster sums of squares
    if number_of_samples_tissue < 11:
        num_for_wvss = number_of_samples_tissue
    else:
        num_for_wvss = 11

    #k mean clustering, elbow method for choosing number of clusters
    wcss = []
    for i in range(1, num_for_wvss):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit([[elem1, elem2] for elem1, elem2 in zip(res_PCA_1, res_PCA_2)])
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, num_for_wvss), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("{}KMeansElbowMethodWCSS{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    if showPlots:
        plt.show()
    plt.close()


    # -> 3 clusters
    #plotting
    kmeans = KMeans(n_clusters=numberOfClusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict([[elem1, elem2] for elem1, elem2 in zip(res_PCA_1, res_PCA_2)])
    predGroup = kmeans.predict([[elem1, elem2] for elem1, elem2 in zip(res_PCA_1, res_PCA_2)])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='green')
    plt.scatter(res_PCA_1, res_PCA_2, c=predGroup, s=50, cmap="plasma")
    plt.xlabel("PCA1({}%)".format(round(pca.explained_variance_ratio_[0]*100), 1))
    plt.ylabel("PCA2({}%)".format(round(pca.explained_variance_ratio_[1]*100), 1))
    plt.suptitle("Kmean Clustering of {} {}".format(dataSet, tissue))
    for i in range(0, len(res_PCA_1)):
        plt.annotate(str(i + 1), (res_PCA_1[i], res_PCA_2[i]))
    plt.savefig("{}KMeansScatterCluster{}{}{}.png".format(outDir, tissue, dataSet, exclusivity))
    if showPlots:
        plt.show()
    plt.close()


def collect_PCs(sfDir, outDir, dataDir): #collect all principle components; sf: sample file
    with open(outDir+"PCcollection.out", "w") as fout:
        for sample in open(sfDir):
            sample = sample.strip().split("-")[1]

            #TODO: find expression for this
            #cycle through all commbinatorial possabilities between [normal, tumor, normaltumor] and [Excl, NotExcl]
            for tis, excl in [["normal", "Excl"], ["normal", "NotExcl"], ["normaltumor", "Excl"], ["normaltumor", "NotExcl"], ["tumor", "Excl"], ["tumor", "NotExcl"]]:
                #wdata = [line.split() for line in open(dataDir + sample + "/" + excl + "In" + tis + sample + "noN.out")]
                working_data = [[float(elem) for elem in dataline.strip().split("\t")] for dataline in open("{}{}/{}In{}{}noN.out".format(dataDir, sample, excl, tis, sample))]
                if len(working_data) == 0 or len(working_data[0]) < 10: #cutoff
                    continue
                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(X=working_data)
                principalDf = pd.DataFrame(data=principalComponents)
                fout.write("\t".join([sample, tis, excl,  str(round(pca.explained_variance_ratio_[0]*100, 1)), str(round(pca.explained_variance_ratio_[1]*100, 1))]) + "\n")


def translate(seq): #helper function
    # table for the translations of the nucleotide sequence to protein sequence
    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }
    protein = ""
    in_orf = False      #open reading frame
    print_next = ""
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        if len(codon) == 3:
            print_next = table[codon]
            if table[codon] == "M" and not in_orf:
                in_orf = True
                print_next = ">>>M"
            if table[codon] == "_" and in_orf:
                in_orf = False
                print_next = "<<<_"
            protein += print_next
    return protein


def generate_sequences(protName, protIDlist, dataSet, geneExonDir, geneFastaDir, outDir, reset):
    #only for one frame
    #if reset:
    #    subprocess.call("rm {}sequences.log".format(outDir), shell=True)
    with open(outDir+"sequences.log", "w") as fout:
        for prot_ID in protIDlist:
            # seq with exitron:
            protein_id = subprocess.Popen("grep {} /home/felix/Documents/CancerProject/DataSets/TCGA-{}/exitrons.info | cut -f2".format(prot_ID, dataSet), stdout=subprocess.PIPE, shell=True).communicate()[0].decode("UTF-8").strip()
            subprocess.call("grep {} {} > {}bed.temp".format(protein_id, geneExonDir, outDir), shell=True)
            subprocess.call("awk '{{OFS=\"\t\"; print $1, $2-1, $3, $4, $8, $12}}' {0}bed.temp > {0}modifbed.temp".format(outDir), shell=True)
            subprocess.call("bedtools getfasta -s -fi {0} -bed {1}modifbed.temp > {1}fasta.temp".format(geneFastaDir, outDir),
                            shell=True)

            #TODO: dry code
            dna_seq = "".join([fastaLine.strip() for fastaLine in open(outDir + "fasta.temp") if not fastaLine.startswith(">")])
            fout.write(">" + protName + "_" + prot_ID +"_DNA_includingEI\n" + dna_seq + "\n")
            fout.write(">" + protName + "_" + prot_ID + "_Translated_IncludingEI_Frame0\n" + translate(dna_seq) + "\n")
            fout.write(">" + protName + "_" + prot_ID + "_Translated_IncludingEI_Frame1\n" + translate(dna_seq[1:]) + "\n")
            fout.write(">" + protName + "_" + prot_ID + "_Translated_IncludingEI_Frame2\n" + translate(dna_seq[2:]) + "\n")
            #may also use .format

            # seq with exitron cut out:
            # -->relative position if the exitron on the mRNA

            start_ei = int(prot_ID.split(":")[1].split("-")[0])
            end_ei = int(prot_ID.split(":")[1].split("-")[1])
            current_pos = 0
            for bed_line in open(outDir + "bed.temp"):
                if not start_ei < int(bed_line.split()[2]):
                    current_pos += int(bed_line.split()[2]) - int(bed_line.split()[1]) + 1
                else:
                    rel_start_ei = current_pos + start_ei - int(bed_line.split()[1]) + 1
                    rel_end_ei = rel_start_ei + end_ei - start_ei + 1
                    break
            #print(rel_start_ei)
            #print(rel_end_ei)
            #print(rel_end_ei-rel_start_ei)

            #TODO: dry this
            fout.write(">" + protName + "_" + prot_ID +"_DNA_excludingEI\n" + dna_seq[:rel_start_ei] + dna_seq[rel_end_ei:] + "\n")
            fout.write(">" + protName + "_" + prot_ID + "_Translated_ExcludingEI_Frame0\n" + translate(dna_seq[:rel_start_ei] + dna_seq[rel_end_ei:]) + "\n")
            fout.write(">" + protName + "_" + prot_ID + "_Translated_ExcludingEI_Frame1\n" + translate(dna_seq[1:rel_start_ei] + dna_seq[rel_end_ei:]) + "\n")
            fout.write(">" + protName + "_" + prot_ID + "_Translated_ExcludingEI_Frame2\n" + translate(dna_seq[2:rel_start_ei] + dna_seq[rel_end_ei:]) + "\n")

            subprocess.call("rm {}*.temp".format(outDir), shell=True)
        fout.close()


def group_ttest(Group, GroupExclusion, DataSet, TestFor, NormalVsCancer, outDir):
    #initiation
    p_values = []
    rest = []
    protein_name_list = []
    in_group_spec, out_group_spec = [], []

    # for naming, needed later when writing into file
    for expression_line in open("/mnt/couch/EI_in_cancer/TCGA-{}/SF_transcript_expr.txt".format(DataSet)):
        if expression_line.startswith("Transcript"):
            continue
        if expression_line.split()[0].startswith("ENST"):
            protein_name_list.append(current_protein_name)
        else:
            current_protein_name = expression_line.split()[0]

    #write specifications of each gene, arranging data and running t-tests, oh whether the expression of splicing factors is enriched in the tumor sample
    with open(outDir, "w") as fout:
        fout.write("TranscriptID\tProteinName\tt\tp\tinGroup_min\tinGroup_mean\tinGroup_median\tinGroup_max\toutGroup_min\toutGroup_mean\toutGroup_median\toutGroup_max\n")  #
        for sf_line in open("/home/felix/Documents/CancerProject/DataSets/TCGA-{}/SF_transcript_expr_Fe.txt".format(DataSet)):  #external data of known splicing factors (sf)
            in_group, out_group = [], []
            identifier = sf_line.split()[0]
            if NormalVsCancer:
                in_group.extend([float(elem) for elem in sf_line.split()[1:int(math.floor(len(sf_line.split()) / 2) + 1)]])
                out_group.extend([float(elem) for elem in sf_line.split()[int(math.floor(len(sf_line.split()) / 2) + 1):]])
            #assigning data
            else:
                if TestFor == "normal":
                    work_data = sf_line.split()[1:int(math.floor(len(sf_line.split()) / 2) + 1)]
                if TestFor == "tumor":
                    work_data = sf_line.split()[int(math.floor(len(sf_line.split()) / 2) + 1):]
                if TestFor == "normaltumor":
                    work_data = sf_line.split()[1:]
                for i, tpm in enumerate(work_data):
                    if i + 1 in Group:
                        in_group.append(float(tpm))
                    elif i + 1 in GroupExclusion:
                        continue
                    else:
                        out_group.append(float(tpm))
            #ttests
            t, p = stats.ttest_ind(np.array(in_group), np.array(out_group))
            p_values.append(p)
            rest.append([identifier, str(round(t, 2))])
            #specifications
            in_group_spec.append("/".join([str(round(min(in_group), 2)), str(round(statistics.mean(in_group), 3)),
                                         str(round(statistics.median(in_group), 3)), str(round(max(in_group), 2))]))
            out_group_spec.append("/".join([str(round(min(out_group), 2)), str(round(statistics.mean(out_group), 3)),
                                          str(round(statistics.median(out_group), 3)), str(round(max(out_group), 2))]))

        #as multiple ttests are performed, a correction has to be performed
        bool_array, pvalues_corr, x, y = statsmodels.stats.multitest.multipletests(
            p_values)  # default is "hs" holm-sidac
        [fout.write("\t".join([r[0], n, r[1], str(round(p, 5)), "\t".join(i.split("/")), "\t".join(o.split("/"))]) + "\n") for r, p, n, i, o in
         zip(rest, pvalues_corr, protein_name_list, in_group_spec, out_group_spec)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_a = subparsers.add_parser("summary", help="Generating a short summary of all Datasets")
    parser_a.add_argument("--sampleFile", "-sf", help="directory of the sample file, should list all CancerSets, that should be investigated")
    parser_a.add_argument("--dirData", "-dirData", help="directory of the Data sets")

    parser_b = subparsers.add_parser("EIvsSamplePlot", help="generates the EI vs Sample plot excluding STAD data set")
    parser_b.add_argument("--sampleFile", "-sf", help="directory of the sample file, should list all CancerSets, that should be investigated")
    parser_b.add_argument("--dirData", "-dirData", help="directory of the Data sets")
    parser_b.add_argument("--outDir", "-o", help="set output directory, end with /")

    parser_c = subparsers.add_parser("generatePWM", help="generates a position weight matrix, program only works in my personal directories")
    parser_c.add_argument("--dataSet", default="STAD", help="input the dataset, that should be investigated, default STAD")
    parser_c.add_argument("--sampleFile", "-sf", default="/home/felix/Documents/CancerProject/DataSets/SampleFileCancer", help="directory of the sample file, should list all CancerSets, that should be investigated, default my diretory")
    parser_c.add_argument("--outputDir", "-o", default="/home/felix/Documents/CancerProject/DataSets/GeneratedOutput/", help="output directory, default my directory /GeneratedOutput/")
    parser_c.add_argument("--showPlot", "-show", action="store_true", help="flag, that can be thrown to show the plot directly")

    parser_d = subparsers.add_parser("originOfExitrons", help="origin of exitrony, only works with my directory, otherwise change program")
    parser_d.add_argument("--dataSet", default="STAD", help="input the dataset, that should be investigated, default STAD")
    parser_d.add_argument("--sampleFile", "-sf", default="/home/felix/Documents/CancerProject/DataSets/SampleFileCancer", help="directory of the sample file, should list all CancerSets, that should be investigated, default my diretory")
    parser_d.add_argument("--outputDir", "-o", default="/home/felix/Documents/CancerProject/DataSets/GeneratedOutput/", help="output directory, default my directory /GeneratedOutput/")
    parser_d.add_argument("--noFileWithNames", action="store_false", help="dont generate file with names")
    parser_d.add_argument("--noFileWithoutNames", action="store_false", help="do not generate a file without names")
    parser_d.add_argument("--fraction", choices=["normal", "tumor", "normaltumor"], default="normal", help="make the choice of which fraction should be investigated")
    parser_d.add_argument("--notExclusive", action="store_false", help="throw this flag to perform no extraction of the genes exclusive to this data set")

    parser_e = subparsers.add_parser("normingNoNamesDataSet", help="normes the data set according to the number of positions sequenced regarding the lowest number of reads")
    parser_e.add_argument("--MetaDataDir", required=True, help="directory of the metaData set")
    parser_e.add_argument("--NoNameDirectory", required=True, help="diretory of the NoName data set generated by 'originOfExitrons'")
    parser_e.add_argument("--samplesPlusDir", "-sp", required=True, help="diretory of the sample plus data set")
    parser_e.add_argument("--tissue", help="imput tissue")

    parser_f = subparsers.add_parser("PCAandKMC", help="perform principle component analysis and K-means Clustering")
    parser_f.add_argument("--dirFiles", required=True, help="input directory of the files being generated by originOfExitrons // equal to output of originOfExitrons")
    parser_f.add_argument("--fraction", default=None, choices=["normal", "tumor", "normaltumor"], help="Tissue imputted at file in, will be used at naming outputFiles; default None")
    parser_f.add_argument("--dataSet", default=None, help= "input data set at file in, will be used at naming; default None")
    parser_f.add_argument("--outputDir", "-o", default="/home/felix/Documents/CancerProject/DataSets/GeneratedOutput/", help="output directory, default my directory /GeneratedOutput/")
    parser_f.add_argument("--showPlots", action="store_true", help="flag can be thrown to output plots")
    parser_f.add_argument("--cutoffFirstComponent", "-cutoff", default=0.05, type=int, help="choose a cutoff vaule for the first principle component")
    parser_f.add_argument("--numberOfClusters", default=3, type=int, help="depending on the elbow plot, the number of clusters can be changed by rerunning the program")
    parser_f.add_argument("--notExclusive", action="store_false", help="throw this flag to perform no extraction of the genes exclusive to this data set; if flag thrown, 'originOfExitrons' had to be run with the same flag too")
    parser_f.add_argument("--minimalNumEI", default=10, type=int, help="choose the minimum number of Exitrons; used for Exclusive")

    parser_g = subparsers.add_parser("collectAllPCs", help="writes a summary file of the first two principle components' ratio split up by data set, exclusivity and tissuetype ")
    parser_g.add_argument("--sampleFile", "-sf", required=True, help="directory of the samplefile as it has been used before")
    parser_g.add_argument("--outDir", "-o", default="./", help="output directory; end with slash; default current directory")
    parser_g.add_argument("--dirData", "-data", default="./", help="directory of the data generated before")

    parser_h = subparsers.add_parser("sequences", help = "generate the sequence of the protein including and excluding the exitron")
    parser_h.add_argument("--proteinName", default="noName", help="enter protein name, just for the sake of naming the output")
    parser_h.add_argument("--proteinID", "-id", nargs="+", required=True, help="proteinID; id of the exitron, following format: 'chr:start-end:strandness'")
    parser_h.add_argument("--dataSet", required=True, help="name of the data set, exitrons.info and exitron-cointaining-exons need to be in according directory")
    parser_h.add_argument("--geneExonDir", default="/home/genomes/h_sapiens/hg38/gencode.v27.exons.bed", help="directory of the genomes exons")
    parser_h.add_argument("--geneFastaDir", default="/home/genomes/h_sapiens/hg38/GRCh38.primary_assembly.genome.fa", help="directory if the genomes fasta")
    parser_h.add_argument("--outDir", "-o", default="/home/felix/Documents/CancerProject/DataSets/GeneratedOutput/", help="output directory")
    parser_h.add_argument("--reset", action="store_true", help="throw this flag to reset the log file")

    parser_i = subparsers.add_parser("tTestForGroup", help="make a ttest for a selected group of samples (expressed as list of numbers/IDs)")
    parser_i.add_argument("--Group", "-g", nargs="+", type=int, help="give the patients numbers, starting from 1")
    parser_i.add_argument("--exclude", "-e", nargs="*", type=int, help="exclude some samples when grouping the samples")
    parser_i.add_argument("--dataset", default="STAD", help="write the name of the tumor sample; default 'STAD'")
    parser_i.add_argument("--testFor", "-test", choices=["normal", "tumor", "normaltumor"], default="normal", help="make a choice of which tissue shall be taken")
    parser_i.add_argument("--NormalVsTumor", action="store_true", help="Throw this flag to compare normal vs tumor; testFor will be obsolete then")
    parser_i.add_argument("--outDir", "-o", help="give the output directory including file name")


    args = parser.parse_args()


    if (args.command == "summary"):
        make_summary(args.sampleFile, args.dirData, True)
    elif (args.command == "EIvsSamplePlot"):
        generate_EI_sample_plot(args.sampleFile, args.dirData, args.outDir)
    elif (args.command == "generatePWM"):
        generate_PWM(args.dataSet, args.sampleFile, args.outputDir, args.showPlot)
    elif (args.command == "originOfExitrons"):
        origin_of_EI(args.dataSet, args.sampleFile, args.outputDir, args.noFileWithNames, args.noFileWithoutNames, args.fraction, args.notExclusive)
    elif (args.command == "PCAandKMC"):
        investigation(args.dirFiles, args.fraction, args.dataSet, args.outputDir, args.showPlots, args.cutoffFirstComponent, args.numberOfClusters, args.notExclusive, args.minimalNumEI)
    elif (args.command == "normingNoNamesDataSet"):
        normalize_data(args.MetaDataDir, args.NoNameDirectory, args.samplesPlusDir, args.tissue)
    elif (args.command == "collectAllPCs"):
        collect_PCs(args.sampleFile, args.outDir, args.dirData)
    elif (args.command == "sequences"):
        generate_sequences(args.proteinName, args.proteinID, args.dataSet, args.geneExonDir, args.geneFastaDir, args.outDir, args.reset)
    elif (args.command == "tTestForGroup"):
        group_ttest(args.Group, args.exclude, args.dataset, args.testFor, args.NormalVsTumor, args.outDir)
