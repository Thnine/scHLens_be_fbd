from lib.utils import *
import scanpy as sc
import numpy as np
import pandas as pd
import umap
import time
import random
import os.path
import sc3s
import scanorama
from lib import openTSNEStab
from lib.RInterface import *
from lib.corr import DIcorrect
import openTSNE
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from qnorm import quantile_normalize
import cosg as cosg
import importlib
import gseapy as gp
import json
import seaborn as sns
from lib.vars import color_list


importlib.reload(cosg)

'''
Pipeline
'''


def globalPipeline(params):
    
    ## 读文件
    adata = readData(params['dataset'],params['JobId'])
    

    ## JobId
    JobId = params['JobId']
    ## ViewId
    ViewId = initView(JobId)
    ## 填入参数
    adata.uns['params'] = params
    adata.uns['JobId'] = JobId
    adata.uns['ViewId'] = ViewId
    adata.uns['ParentId'] = params['ParentId']
     
    
    ## 质量控制 对每个数据集进行过滤
    if 'QC' in adata.uns['params']:
        adata = QC(adata)
    
    ## 数据融合
    if 'DI' in adata.uns['params'] and len(adata.uns['params']['DI']['Datasets']) != 0:
        adata = DI(adata)
    
    ## 如果不自带label，那么给予统一的默认label id：c_0
    if 'label' not in adata.obs:
        adata.obs['label'] = pd.Series(['c_0' for i in range(len(adata.obs))],dtype='category',index=adata.obs.index)
    
    localPointAdata = adata.copy()
    
    ## 存入原始矩阵 TODO 和数据融合配合
    adata.uns['count'] = adata.copy()

    ## 变换 （FS发生在TS中）
    if 'TS' in adata.uns['params']:
        adata = TS(adata)

    ## neighbors
    if 'NB' in adata.uns['params']:
        adata = NB(adata)

    ## 降维
    if 'DR' in adata.uns['params']:
        adata = DR(adata)
    else:
        adata.obsm['embedding'] = np.zeros((len(adata.obs),2)) 

    ## 聚类
    if 'CL' in adata.uns['params']:
        adata = CL(adata)

    ## 保存聚类结果到localPoint
    ## 保存“供local pipeline进行划分时加载的缓存”
    localPointAdata.obs['label'] = adata.obs.label
    saveCache(localPointAdata, JobId, ViewId, 'localPoint')# uns为空，label只有
    

    ## 过滤单样本的聚类，防止影响后续
    adata = clearSimpleSizeCluster(adata)

    ## 轨迹推断
    if 'TI' in adata.uns['params']:
        adata = TI(adata)
    
    ## 细胞通讯
    if 'CC' in adata.uns['params']:
        adata = CC(adata)

    ## 计算标志基因
    if 'MK' in adata.uns['params']  and len(adata.obs['label'].cat.categories) > 1:
        adata = MK(adata)

    ## 保存 “查询用缓存”
    saveCache(adata,adata.uns['JobId'],adata.uns['ViewId'],'Query')

    ## 构建保存metaData
    metaData = generateMetaDataFromAdata(adata)
    saveViewMetaData(JobId, ViewId, metaData)

    ## 构建保存Tree
    saveToTree(adata)

    return adata


def localPipeline(params):
    ## JobId
    JobId = params['JobId']
    ## ParentId
    ParentId = params['ParentId']
    ## ViewId
    ViewId = initView(JobId)
    ## 读取文件    
    adata = readCache(JobId, ParentId, 'localPoint')

    ## 局部化
    adata = adata[params['type']['local']['chosenCells'],:].copy()

    
    ## 填入参数
    adata.uns['params'] = params
    adata.uns['JobId'] = JobId
    adata.uns['ViewId'] = ViewId
    adata.uns['ParentId'] = ParentId

    ## 如果不自带label，那么给予统一的默认label id：c_0
    if 'label' not in adata.obs:
        adata.obs['label'] = pd.Series(['c_0' for i in range(len(adata.obs))],dtype='category',index=adata.obs.index)

    localPointAdata = adata.copy()

    ## 存入原始矩阵 TODO 和数据融合配合
    adata.uns['count'] = adata.copy()


    ## 变换 （FS发生在TS中）
    if 'TS' in adata.uns['params']:
        adata = TS(adata)

    ## neighbors
    if 'NB' in adata.uns['params']:
        adata = NB(adata)

    ## 降维
    if 'DR' in adata.uns['params']:
        adata = DR(adata)
    else:
        adata.obsm['embedding'] = np.zeros((len(adata.obs),2))

    ## 聚类
    if 'CL' in adata.uns['params']:
        adata = CL(adata)

    ## 保存聚类结果到localPoint
    ## 保存“供local pipeline进行划分时加载的缓存”
    localPointAdata.obs['label'] = adata.obs.label
    saveCache(localPointAdata, JobId, ViewId, 'localPoint')# uns为空，label只有


    ## 过滤单样本的聚类，防止影响后续
    adata = clearSimpleSizeCluster(adata)

    ## 轨迹推断
    if 'TI' in adata.uns['params']:
        adata = TI(adata)

    ## 细胞通讯
    if 'CC' in adata.uns['params']:
        adata = CC(adata)

    ## 计算标志基因
    if 'MK' in adata.uns['params'] and len(adata.obs['label'].cat.categories) > 1:
        adata = MK(adata)

    ## 保存 “查询用缓存”
    saveCache(adata,adata.uns['JobId'],adata.uns['ViewId'],'Query')

    ## 构建保存metaData
    metaData = generateMetaDataFromAdata(adata)
    saveViewMetaData(JobId, ViewId, metaData)

    ## 构建保存Tree
    saveToTree(adata)


    return adata


def mergePipeline(params):
    adata = None

    return adata


'''
分析步骤
'''

## 质量控制  quality control
def QC(adata):
    '''
    adata: Anndata
    '''
    ## 过滤掉异常细胞和异常基因，这里异常被定义为表达量低
    if 'filterCells' in adata.uns['params']['QC']:
        sc.pp.filter_cells(adata, **adata.uns['params']['QC']['filterCells'])
    if 'filterGenes' in adata.uns['params']['QC']:
        sc.pp.filter_genes(adata, **adata.uns['params']['QC']['filterGenes'])
    ## 过滤掉高线粒体基因以及相关细胞
    if 'qcMetrics' in adata.uns['params']['QC']:
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        if 'geneCounts' in adata.uns['params']['QC']['qcMetrics']:
            adata = adata[adata.obs.n_genes_by_counts < adata.uns['params']['QC']['qcMetrics']['geneCounts'], :]
        if 'pctCounts' in adata.uns['params']['QC']['qcMetrics']:
            adata = adata[adata.obs.pct_counts_mt < adata.uns['params']['QC']['qcMetrics']['pctCounts'], :]
    return adata

## 数据集融合 Data Integration
def DI(adata):
    params = adata.uns['params']
    ### 编号batch
    batch = []
    batch.append(params['dataset']['name'])
    for item in adata.uns['params']['DI']['Datasets']:
        batch.append(item['Dataset']['name'])
    ### 取数据，并且进行简单的质量控制
    adatas = []
    for item in adata.uns['params']['DI']['Datasets']:
        tempData = readData(item['Dataset'],adata.uns['params']['JobId'])
        tempData.uns['params'] = {}
        tempData.uns['params']['QC'] = item['qcParams']
        adatas.append(QC(tempData))
    ### 求基因的交集
    var_names = adata.var_names
    for item in adatas:
        var_names = var_names.intersection(item.var_names)
    adata = adata[:,var_names]
    adatas = list(map(lambda e:e[:,var_names],adatas))

    ## 融合
    ### ingest方法
    if 'Ingest' in adata.uns['params']['DI']['Method']:
        ### 处理ref数据集
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        if adata.uns['params']['DI']['Method']['Ingest']['embeddingMethod'] == 'UMAP':
            sc.tl.umap(adata)
        ### 处理query数据集
        if adata.uns['params']['DI']['Method']['Ingest']['embeddingMethod'] == 'UMAP':
            for i in range(len(adatas)):
                sc.tl.ingest(adatas[i],adata,embedding_method='umap')
        elif adata.uns['params']['DI']['Method']['Ingest']['embeddingMethod'] == 'PCA':
            for i in range(len(adatas)):
                sc.tl.ingest(adatas[i],adata,embedding_method='pca')
        ### 合并
        adata_all = adata.concatenate(*adatas,batch_categories=batch)
        if adata.uns['params']['DI']['Method']['Ingest']['embeddingMethod'] == 'UMAP':
            adata_all.obsm['di_prj'] = adata_all.obsm['X_umap']
        elif adata.uns['params']['DI']['Method']['Ingest']['embeddingMethod'] == 'PCA':
            adata_all.obsm['di_prj'] = adata_all.obsm['X_pca']
        adata_all.var['vst.variable'] = pd.Series(True,adata_all.var.index)
        ### 纠正
        adata_all = DIcorrect(adata_all)

    ### Scanorama方法
    elif 'Scanorama' in adata.uns['params']['DI']['Method']:
        corrected = scanorama.correct_scanpy([adata,*adatas],return_dimred=True)
        adata_all = corrected[0].concatenate(*corrected[1:],batch_categories=batch)

    ### Harmony方法
    elif 'Harmony' in adata.uns['params']['DI']['Method']:
        adata_all = adata.concatenate(*adatas,batch_categories=batch)
        sc.tl.pca(adata_all)
        sc.external.pp.harmony_integrate(adata_all,'batch')
        adata_all.obsm['di_prj'] = adata_all.obsm['X_pca_harmony']
        adata_all.var['vst.variable'] = pd.Series(True,adata_all.var.index)
        ### 纠正
        adata_all = DIcorrect(adata_all)

    adata_all.uns['params'] = params
    return adata_all

## 变换 transform
def TS(adata): 
    '''
    adata: Anndata
    '''
    if 'normalize' in adata.uns['params']['TS']:
        # if 'qnorm' in adata.uns['params']['TS']['normalize']:
        #     0
        # elif 'total' in adata.uns['params']['TS']['normalize']:
        #     0
        sc.pp.normalize_total(adata, target_sum=1e4)
    if 'log1p' in adata.uns['params']['TS']:
        sc.pp.log1p(adata)

    if 'CC' in adata.uns['params']:
        saveCache(adata,adata.uns['JobId'],adata.uns['ViewId'],'CellChat')

    ## 特征选择
    if 'FS' in adata.uns['params']['TS']:
        adata = FS(adata)

    if 'local' not in adata.uns['params']['type']:
        if 'regressOut' in adata.uns['params']['TS']:        
            sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt']) #TODO 这里是否和qc_metrics有强相关性？
        if 'scale' in adata.uns['params']['TS']:
            sc.pp.scale(adata, max_value=10)

    return adata

## 特征选择 feature selection
def FS(adata):
    '''
    adata: Anndata
    '''
    if 'highlyVariableGenes' in adata.uns['params']['TS']['FS']:
        HVParams = {}   
        if 'topGenes' in adata.uns['params']['TS']['FS']['highlyVariableGenes']:
            HVParams['n_top_genes'] = adata.uns['params']['TS']['FS']['highlyVariableGenes']['topGenes']
        if 'minMean' in adata.uns['params']['TS']['FS']['highlyVariableGenes']:
            HVParams['min_mean'] = adata.uns['params']['TS']['FS']['highlyVariableGenes']['minMean']
        if 'maxMean' in adata.uns['params']['TS']['FS']['highlyVariableGenes']:
            HVParams['max_mean'] = adata.uns['params']['TS']['FS']['highlyVariableGenes']['maxMean']
        if 'minDisp' in adata.uns['params']['TS']['FS']['highlyVariableGenes']:
            HVParams['min_disp'] = adata.uns['params']['TS']['FS']['highlyVariableGenes']['minDisp']
        if 'maxDisp' in adata.uns['params']['TS']['FS']['highlyVariableGenes']:
            HVParams['max_disp'] = adata.uns['params']['TS']['FS']['highlyVariableGenes']['maxDisp'],
        sc.pp.highly_variable_genes(adata, **HVParams)
        adata = adata[:, adata.var.highly_variable]
    elif 'scry' in adata.uns['params']['TS']['FS']:
        nTopGenes = adata.uns['params']['TS']['FS']['scry']['topGenes']
        if hasattr(adata.uns['count'].X,'A'):
            adata = adata[:,scry(adata.uns['count'].X.A, adata.var.index, adata.obs.index,nTopGenes)]
        else:
            adata = adata[:,scry(adata.uns['count'].X, adata.var.index, adata.obs.index,nTopGenes)]
    elif 'SCTransform' in adata.uns['params']['TS']['FS']:
        nTopGenes = adata.uns['params']['TS']['FS']['SCTransform']['topGenes']
        if hasattr(adata.uns['count'].X,'A'):
            adata = adata[:,SCTransform(adata.uns['count'].X.A, adata.var.index, adata.obs.index,nTopGenes)]
        else:
            adata = adata[:,SCTransform(adata.uns['count'].X, adata.var.index, adata.obs.index,nTopGenes)]
    elif 'marker' in adata.uns['params']['TS']['FS']:## only local mode
        if 'local' in adata.uns['params']['type']:
            ## Parent
            ParentId = adata.uns['params']['ParentId']
            JobId = adata.uns['params']['JobId']
            parentAdata = readCache(JobId, ParentId, 'Query')
            
            chosenCells = adata.obs.index.tolist()
            chosenLabels = parentAdata[chosenCells].obs.label.cat.categories.tolist()
            parentLabels = parentAdata.obs.label.cat.categories.tolist()
            parentMarkerNames = parentAdata.uns['raw_marker']['names']
            marker_list = []
            for label in chosenLabels:
                i = parentLabels.index(label)
                for j in range(len(parentMarkerNames)):
                    marker_list.append(parentMarkerNames[j][i])
            ## important: marker_list去重
            marker_list = list(set(marker_list))
            adata = adata[:,marker_list]
    elif 'combinedMarker' in  adata.uns['params']['TS']['FS']:## only local mode
        if 'local' in adata.uns['params']['type']:
            ## Parent
            ParentId = adata.uns['params']['ParentId']
            JobId = adata.uns['params']['JobId']
            parentAdata = readCache(JobId, ParentId, 'Query')
            
            chosenCells = adata.obs.index.tolist()
            chosenLabels = parentAdata[chosenCells].obs.label.cat.categories.tolist()
            parentLabels = parentAdata.obs.label.cat.categories.tolist()
            parentMarkerNames = parentAdata.uns['raw_marker']['names']
            marker_list = []
            for label in chosenLabels:
                i = parentLabels.index(label)
                for j in range(len(parentMarkerNames)):
                    marker_list.append(parentMarkerNames[j][i])
            ## important: marker_list去重
            marker_list = list(set(marker_list))



    return adata

## 邻近图 neighbors
def NB(adata):
    '''
    adata: Anndata
    '''
    NBParams = {}
    sc.tl.pca(adata, svd_solver='arpack')
    if 'nNeighbors' in adata.uns['params']['NB']:
        NBParams['n_neighbors'] = adata.uns['params']['NB']['nNeighbors']
    if 'nPcs' in adata.uns['params']['NB']:
        NBParams['n_pcs'] = adata.uns['params']['NB']['nPcs']
    sc.pp.neighbors(adata, **NBParams)
    return adata

## 降维 dimension reduction
def DR(adata):  
    
    '''
    adata: Anndata
    '''
    
    ## 计算并保存TD
    TD = calculateTD(adata.X)
    adata.uns['TD'] = TD

    ## 执行降维合并算法
    if 'UMAP' in adata.uns['params']['DR']:
        if 'minDist' in adata.uns['params']['DR']['UMAP']:
             embedding = umap.UMAP(metric="precomputed",min_dist = adata.uns['params']['DR']['UMAP']['minDist'],random_state= 0).fit_transform(TD)
             adata.obsm['embedding'] = embedding
    elif 'T-SNE' in adata.uns['params']['DR']:
        if 'perplexity' in adata.uns['params']['DR']['T-SNE']:
            embedding = openTSNE.TSNE(metric="precomputed", negative_gradient_method='fft',random_state= 0,initialization='random',perplexity=adata.uns['params']['DR']['T-SNE']['perplexity']).fit(TD)
            embedding = np.array(embedding)
            adata.obsm['embedding'] = embedding
    elif 'PCA' in adata.uns['params']['DR']:
            sc.tl.pca(adata,n_comps=2,svd_solver='arpack')
            adata.obsm['embedding'] = adata.obsm['X_pca']

    sc.tl.pca(adata,n_comps=2,svd_solver='arpack') ## TODO to_delete

    
    return adata

## 聚类 cluster
def CL(adata):
    '''
    adata: Anndata
    '''
    if 'leiden' in adata.uns['params']['CL']:
        if 'resolution' in adata.uns['params']['CL']['leiden']:
            sc.tl.leiden(adata, resolution=adata.uns['params']['CL']['leiden']['resolution'])
            adata.obs['label'] = adata.obs['leiden']
    elif 'louvain' in adata.uns['params']['CL']:
        if 'resolution' in adata.uns['params']['CL']['louvain']:
            sc.tl.louvain(adata, resolution=adata.uns['params']['CL']['louvain']['resolution'])
            adata.obs['label'] = adata.obs['louvain']
    elif 'sc3s' in adata.uns['params']['CL']:
        if 'n_clusters' in adata.uns['params']['CL']['sc3s']:
            sc3s.tl.consensus(adata, n_clusters=adata.uns['params']['CL']['sc3s']['n_clusters'])
            adata.obs['label'] = adata.obs['sc3s_' + str(adata.uns['params']['CL']['sc3s']['n_clusters'])]
            adata.uns.pop('sc3s_trials')
    
    ## refine the label
    new_cate = []
    for item in adata.obs['label'].cat.categories:
        new_cate.append('c_' + str(item))
    adata.obs['label'].cat.rename_categories(new_cate,inplace=True)

    ## 聚类分层
    # sc.tl.dendrogram(adata,groupby='label',key_added='dendrogram')


    return adata

## 计算标志基因 marker gene
def MK(adata):
    '''
    adata: Anndata
    '''
    MarkerParams = {}
    if 'markerMethod' in adata.uns['params']['MK']:
        MarkerParams['method'] = adata.uns['params']['MK']['markerMethod']
    if 'nGenes' in adata.uns['params']['MK']:
        MarkerParams['n_genes'] = adata.uns['params']['MK']['nGenes']

    if MarkerParams['method'] == 'wilcoxon-test': ## wilcoxon-test
        sc.tl.rank_genes_groups(adata,
                        groupby='label',
                        method = 'wilcoxon',
                        n_genes = MarkerParams['n_genes'],
                        tie_correct=False,
                        key_added='raw_marker')
    elif MarkerParams['method'] == 'wilcoxon-test(TLE)': ## wilcoxon-test(TLE)
        sc.tl.rank_genes_groups(adata,
                        groupby='label',
                        method = ' ilcoxon',
                        n_genes = MarkerParams['n_genes'],
                        tie_correct=True,
                        key_added='raw_marker')
    elif MarkerParams['method'] == 'logreg': ## logreg
        sc.tl.rank_genes_groups(adata,
                        groupby='label',
                        method = 'logreg',
                        n_genes = MarkerParams['n_genes'],
                        key_added='raw_marker',
                        pts=True)
    elif MarkerParams['method'] == 'COSG': ## COSG
        cosg.cosg(adata,key_added='raw_marker',
                        mu=1,
                        n_genes_user=MarkerParams['n_genes'],
                        groupby='label')

    return adata

## 轨迹推断 Trajectory inference
def TI(adata):
    '''
    adata: Anndata
    '''
    if 'paga' in adata.uns['params']['TI']:
        sc.tl.paga(adata,groups='label')
        sc.pl.paga(adata,show=False,add_pos=True)
        sc.tl.draw_graph(adata,init_pos='paga')
    elif 'slingshot' in adata.uns['params']['TI']:
        rd = adata.obsm['embedding']
        cl = adata.obs['label'].to_list()
        result = slingshot(rd,cl)
        adata.uns['slingshot'] = result
    return adata

## 细胞通讯 Cell Chat
def CC(adata):
    '''
    adata:Anndata
    '''
    ## read data
    cc_data = readCache(adata.uns['JobId'],adata.uns['ViewId'],'CellChat')
    ## run
    if 'CellChat' in adata.uns['params']['CC']:
        result = CellChat(cc_data.X,cc_data.obs.index.tolist(),cc_data.var.index.tolist(),adata.obs['label'].tolist(),adata.uns['params']['CC']['CellChat']['DatabaseType'])
        adata.uns['CC'] = result
    elif 'NicheNet' in adata.uns['params']['CC']:
        print('run NicheNet')
    
    return adata


## 为adata生成新的metaData
def generateMetaDataFromAdata(adata):

    ## JobId
    JobId = adata.uns['JobId']
    ## ViewId
    ViewId = adata.uns['ViewId']
    ## ParentId
    ParentId = adata.uns['ParentId']


    ## generate
    metaData = {}
    group_ids = adata.obs.label.cat.categories
    ### color
    # colors = sns.color_palette("colorblind",n_colors=len(group_ids)).as_hex()
    colors = color_list[:len(group_ids)]
    metaData['group_color'] = {}
    for i,id in enumerate(group_ids):
        metaData['group_color'][id] = colors[i]
    ### name
    metaData['group_name'] = {}
    for i,id in enumerate(group_ids):
        metaData['group_name'][id] = id
    ### raw_embedding_range
    minRawEmbedding = np.min(adata.obsm['embedding'],axis=0)
    maxRawEmbedding = np.max(adata.obsm['embedding'],axis=0)
    metaData['raw_embedding_range'] = {
        'x':[float(minRawEmbedding[0]),float(maxRawEmbedding[0])],
        'y':[float(minRawEmbedding[1]),float(maxRawEmbedding[1])],
    }
    metaData['history_group_num'] = len(group_ids)

    return metaData

## 删除adata中单样本聚类
def clearSimpleSizeCluster(adata):
    labelIndex = {}
    for key,value in adata.obs.label.items():
        labelIndex[key] = (adata.obs['label'].value_counts() > 1)[value]
    labelIndex = pd.Series(labelIndex)
    return adata[labelIndex,:]