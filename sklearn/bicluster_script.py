# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:52:40 2017

@author: binyang_ni
"""
import pandas as pd
import numpy as np
import os

#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster.bicluster import SpectralBiclustering
import argparse
import matplotlib
### The Checkerboard algorithm
def Checkerboard_structure(input_path,top_sd,n_clusters,output_path):

    ###input data
    input_dat=pd.read_csv(input_path,index_col=0,sep='\t',comment='#')
    ### get index and sample name
    get_index = input_dat.index.astype(str)+'_'+input_dat.ix[:,0].astype(str)+'_'+\
    input_dat.ix[:,1].astype(str)+'_'+input_dat.ix[:,2].astype(str)
    get_samp_name = input_dat.columns[3:]
    
    pro_dat = input_dat.fillna(0)
    pro_dat = pro_dat.ix[:,3:]
    pro_dat.index = get_index
    pro_dat.columns = get_samp_name
#    pro_dat = 2**pro_dat-1
    
    df_sd = pro_dat.apply(np.std,axis=1)
    df_sd_sort = df_sd.sort_values(ascending = False)
    df_sd_sort_top = df_sd_sort.ix[:int(len(df_sd_sort)*top_sd)]
    pro_dat = pro_dat.ix[df_sd_sort_top.index,:]
    
    sd_index = pro_dat.index
    sd_sample_names = pro_dat.columns
    
#    plt.matshow(common_data3, cmap=plt.cm.Blues)
#    plt.title("Original dataset")
    
    model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                                 random_state=0)
    model.fit(pro_dat)
    
    pro_dat = np.array(pro_dat)
    fit_data = pro_dat[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    
    ### output image
#    plt.figure(figsize=(12,12))
#    plt.matshow(fit_data, cmap=plt.cm.Blues)
#    plt.title("After biclustering; rearranged to show biclusters")
#    out_img_path = os.path.join(os.path.split(input_path)[0],'bicluster.png')
#    plt.savefig(out_img_path)
    
    ### output the model fitting data
    fit_data = pd.DataFrame(fit_data)
    fit_data.index = sd_index[np.argsort(model.row_labels_)]
    fit_data.columns = sd_sample_names[np.argsort(model.column_labels_)]
    
    out_fit_data_path = os.path.join(output_path,'fit_data.csv')
    fit_data.to_csv(out_fit_data_path)
    
    ### output image
    fig = plt.figure(figsize=(20,40))
    ax = fig.add_subplot(111)
    ax.matshow(fit_data, cmap=plt.cm.Blues)
    ax.set_title("After biclustering; rearranged to show biclusters")
    out_img_path = os.path.join(output_path,'bicluster.png')
    fig.savefig(out_img_path)
    
    ### output module
    a11 = pd.Series(model.row_labels_)
    b11 = pd.Series(model.column_labels_)
    c11 = a11.groupby(a11).size()
    c22 = b11.groupby(b11).size()
    d11 = pd.DataFrame(a11.sort_values().values,fit_data.index.values)
    d22 = pd.DataFrame(b11.sort_values().values,fit_data.columns.values)
    d11.columns = ['cpg_module']
    d22.columns = ['sample_module']
    
    out_module_path = os.path.join(output_path,'output.xlsx')
    writer = pd.ExcelWriter(out_module_path)
    d11.to_excel(writer,'Sheet1')
    d22.to_excel(writer,'Sheet2')
    writer.save()
   # 
    print("\n")
    print("cpg module:")
    print(c11)
    print("\n")
    print("sample module:")
    print(c22)

### The Block diagonal algorithm
def Block_diagonal(input_path,top_sd,n_clusters,output_path):
    
    ###input data
    input_dat=pd.read_csv(input_path,index_col=0,sep='\t',comment='#')
    
    ### get index and sample name
    get_index = input_dat.index.astype(str)+'_'+input_dat.ix[:,0].astype(str)+'_'+\
    input_dat.ix[:,1].astype(str)+'_'+input_dat.ix[:,2].astype(str)
    get_samp_name = input_dat.columns[3:]
    
    pro_dat = input_dat.fillna(0)
    pro_dat = pro_dat.ix[:,3:]
    pro_dat.index = get_index
    pro_dat.columns = get_samp_name
#    pro_dat = 2**pro_dat-1
    
    df_sd = pro_dat.apply(np.std,axis=1)
    df_sd_sort = df_sd.sort_values(ascending = False)
    df_sd_sort_top = df_sd_sort.ix[:int(len(df_sd_sort)*top_sd)]
    pro_dat = pro_dat.ix[df_sd_sort_top.index,:]
    
    sd_index = pro_dat.index
    sd_sample_names = pro_dat.columns
    
    #plt.matshow(pro_dat, cmap=plt.cm.Blues)
    #plt.title("Original dataset")
    
    ### model
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model.fit(pro_dat)
    
    pro_dat = np.array(pro_dat)
    fit_data = pro_dat[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    
    
    ### output the model fitting data
    fit_data = pd.DataFrame(fit_data)
    fit_data.index = sd_index[np.argsort(model.row_labels_)]
    fit_data.columns = sd_sample_names[np.argsort(model.column_labels_)]
    
    out_fit_data_path = os.path.join(output_path,'fit_data.csv')
    fit_data.to_csv(out_fit_data_path)
    ### output image
    fig = plt.figure(figsize=(20,40))
    ax = fig.add_subplot(111)
    ax.matshow(fit_data, cmap=plt.cm.Blues)
    #cax = ax.matshow(pro_dat, interpolation='nearest')
    #fig.colorbar(cax)
    ax.set_title("After biclustering; rearranged to show biclusters")
#    ax.set_xticklabels(fit_data.columns )
#    ax.set_yticklabels(fit_data.index)

    out_img_path = os.path.join(output_path,'bicluster.png')
    fig.savefig(out_img_path)
    
    ### output module
    a11 = pd.Series(model.row_labels_)
    b11 = pd.Series(model.column_labels_)
    c11 = a11.groupby(a11).size()
    c22 = b11.groupby(b11).size()
    d11 = pd.DataFrame(a11.sort_values().values,fit_data.index.values)
    d22 = pd.DataFrame(b11.sort_values().values,fit_data.columns.values)
    d11.columns = ['cpg_module']
    d22.columns = ['sample_module']
    
    out_module_path = os.path.join(output_path,'output.xlsx')
    writer = pd.ExcelWriter(out_module_path)
    d11.to_excel(writer,'Sheet1')
    d22.to_excel(writer,'Sheet2')
    writer.save()
   # 
    print("\n")
    print("cpg module:")
    print(c11)
    print("\n")
    print("sample module:")
    print(c22)

### main
def main():
    
    # Get command line arguments.
    parser = argparse.ArgumentParser(description='A bicluster to find module')
    
    parser.add_argument('-i', '-input', dest='input_path',
                        default = '/home/binyang_ni/bio_analysis/biclustering/py_script/WM.log2CPMplus1.170503.stat' ,
                        help='input file, the tab-deliminated data matrix file. default  /home/binyang_ni/bio_analysis/biclustering/py_script/WM.log2CPMplus1.170503.stat')
    parser.add_argument('-o', '-output', dest='output_path',
                        help='The output directory, default  output directory is the same as input directory')
    parser.add_argument('-sd','-SD', action='store',type = float,dest='top_sd',default = 0.05,
                        help='the top  percent of Standard deviation. default 0.05')
    parser.add_argument('-n','-n_clusters', action='store',type = int, dest='n_clusters',
                        default = 2,help='The number of biclusters to find on Block diagonal algorithm. default 2')
    parser.add_argument('-b', '-Block', dest = 'Block', 
                        help="Apply the Block diagonal algorithm. default  Block diagonal algorithm",
                        action="store_true")
    parser.add_argument('-c', '-Checkerboard',dest = 'Checkerboard', 
                        help="Apply the Checkerboard algorithm. ",
                        action="store_true")
    parser.add_argument('-nc', type = str, dest='collection',
                        help="The number of biclusters to find on Checkerboard algorithm. e.g. 'python3 bicluster_script.py -c -nc 10,3' . default 10,3")
    args = parser.parse_args()
    
    if args.top_sd<=0 or args.top_sd>1:
    #    print("The input number must be between 0 and 1")
        exit("ValueError:The '-sd' number must be between 0 and 1")
        
    if args.n_clusters<=1:
    #    print("The input number must be great than 1")
        exit("ValueError:The '-n' number must be great than 1")
    ###set the hyper-parameter
    input_path = args.input_path
    
    if args.output_path is None:
        if len(os.path.split(input_path)[0])==0:
            output_path = os.getcwd()
        else:
            output_path = os.path.split(input_path)[0]
    else:
        output_path = args.output_path
    if args.Block:
        if args.collection:
            exit("Argument Error:'-nc' is used for Checkerboard algorithm")
        n_clusters = args.n_clusters
        top_sd = args.top_sd
        ### run the Block diagonal algorithm
        Block_diagonal(input_path,top_sd,n_clusters,output_path)
    
    elif args.Checkerboard:
        if args.collection:
            n_clusters = args.collection.split(',') # ['1','2','3','4']
            n_clusters = tuple(map(int, n_clusters))
        else:
            n_clusters = (10,3)
#        print(n_clusters)
        top_sd = args.top_sd
        ### run The Checkerboard algorithm
        Checkerboard_structure(input_path,top_sd,n_clusters,output_path)
        
    else:
        if args.collection:
            exit("ValueError:'-nc' is used for Checkerboard algorithm")
        n_clusters = args.n_clusters
        top_sd = args.top_sd
        input_path = args.input_path
        
        ### run the Block diagonal algorithm
        Block_diagonal(input_path,top_sd,n_clusters,output_path) 

if __name__ == '__main__':
    main()
    print("\n")
    print("done")

#index_find = fit_data.index.get_loc('cg22944461')
#columns_find = fit_data.columns.get_loc('DPM-LC002-119-FE1')
#find_data = fit_data.ix[:a11.groupby(a11).size()[0],:b11.groupby(b11).size()[0]]
#
#find_data.to_csv('D:/python_workspace/biclustering/new_fit/co_cluster/find_2.csv')
