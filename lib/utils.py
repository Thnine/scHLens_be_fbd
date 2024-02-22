import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
import random
import scanpy as sc
import numpy as np
import os
import time
from pathlib import Path
import math
import matplotlib
import seaborn as sns
import json
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix,csc_matrix
import re
import mousipy


#
# 后台管理
#

## 程序启动的初始化工作，每次启动程序时运行
def initApp():
    ### 初始化文件夹
    if not os.path.exists('job'):
        os.makedirs('job')
    if not os.path.exists('sample'):
        os.makedirs('sample')
    if not os.path.exists('job_attachment'):
        os.makedirs('job_attachment')
    if not os.path.exists('message'):
        os.makedirs('message')


## 初始化Job
def initJob():
    if not os.path.exists('./job'):
        os.makedirs('job')
    ##8位随机ID
    ID = None
    IDLen = 8
    characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    while True:
        candidateID = ''.join(random.choice(characterList) for _ in range(IDLen))
        if not os.path.exists('job/' + candidateID):
            ID = candidateID
            break

    ##创建job的文件夹
    os.makedirs('job/' + ID)
    os.makedirs('job/' + ID + '/dataset') #存放数据集
    os.makedirs('job/' + ID + '/view') #存放视图数据
    os.makedirs('job/' + ID + '/meta') #存放元数据

    ##创建Tree文件
    with open('job/' + ID + '/meta/' + 'tree.json','w') as f:
        jsonTree = json.dumps({})
        f.write(jsonTree)
    ##创建deleteCells文件
    with open('job/' + ID + '/meta/' + 'deleteCells.json','w') as f:
        deleteCells = json.dumps([])
        f.write(deleteCells)
    ##在job_attachment中创建对应的目录
    if not os.path.exists('./job_attachment'):
        os.makedirs('job_attachment')
    os.makedirs('job_attachment/' + ID)

    return ID

## 分配视图的ID，并且创建新的视图
def initView(JobId,IDLen = 8):
    if not os.path.exists('job/' + JobId + '/view'):
        os.makedirs('job/' + JobId + '/view')
    ### 分配视图ID
    ID = None
    characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    while True:
        candidateID = ''.join(random.choice(characterList) for _ in range(IDLen))
        if not os.path.exists('job/' + JobId + '/view/' + candidateID):
            ID = candidateID
            break
    ### 创建新的视图文件夹
    os.makedirs('job/' + JobId + '/view/' + candidateID)
    os.makedirs('job/' + JobId + '/view/' + candidateID + '/cache')
    os.makedirs('job/' + JobId + '/view/' + candidateID + '/export')
    return ID

## 保存anndata为缓存
def saveCache(adata,JobId,ViewId,name):
    '''
    adata : anndata数据
    name : 缓存的种类，重复存在的缓存会被覆盖
    '''
    path = 'job/' + JobId + '/view/' + ViewId +  '/cache/' + name + '.h5ad'
    adata.write(path)

    return adata

## 读取缓存
def readCache(JobId,ViewId,name):
    path = 'job/' + JobId + '/view/' + ViewId +  '/cache/' + name + '.h5ad'
    adata = sc.read(path)

    return adata

## 读数据集为anndata
def readData(datasetConfig,JobId):
    
    adata = None

    ##构建文件路径
    path = ''
    if datasetConfig['from'] == 'sample':
        path = './sample/' + datasetConfig['name']
    elif datasetConfig['from'] == 'user':
        path = './job/' + JobId + '/dataset/' + datasetConfig['name']
    
    ##读取文件
    if datasetConfig['type'] == '10x-mtx':
        adata = sc.read_10x_mtx(path, var_names='gene_symbols')
    elif datasetConfig['type'] == 'csv':
        adata = sc.read(path + '/' + 'expression.csv')
        if os.path.exists(path + '/' + 'anno.txt'):
            label = pd.read_csv(path + '/' + 'anno.txt',delimiter='\t')
            adata.obs['label'] = label['cellAnno'].values
    elif datasetConfig['type'] == 'h5ad':
        filename = None
        for fn in os.listdir(path):
            if re.match('.*\.h5ad',fn) is not None:
                filename = fn
                break
        adata = sc.read_h5ad(path + '/' + filename)
    return adata

## 检测View元数据是否存在
def isViewMetaDataExist(JobId,ViewId):
    return os.path.exists('job/' + JobId + '/meta/' + ViewId + '.json')
## 保存View元数据
def saveViewMetaData(JobId,ViewId,metaData):
    # if not os.path.exists('job/' + JobId + '/meta/' + ViewId):
    #     os.makedirs('job/' + JobId + '/meta/' + ViewId)
    meta_json = json.dumps(metaData)
    tempFile = open('job/' + JobId + '/meta/' + ViewId + '.json', 'w')
    tempFile.write(meta_json)
    tempFile.close()
## 读取View元数据
def getViewMetaData(JobId,ViewId):
    with open('job/' + JobId + '/meta/' + ViewId + '.json','r',encoding = 'utf-8') as f:
        result = json.load(f)
    return result

## 读取Tree信息
def getTree(JobId):
    with open('job/' + JobId + '/meta/' + 'tree.json','r',encoding = 'utf-8') as f:
        result = json.load(f)
    return result

## 将该adata保存到tree.json中
def saveToTree(adata):
    JobId = adata.uns['JobId']
    ViewId = adata.uns['ViewId']
    ParentId = adata.uns['ParentId']

    tree = getTree(JobId)

    def findNode(root,nodeId):
        if(root['ViewId'] == nodeId):
            return root
        elif len(root['children']) != 0:
            for child in root['children']:
                result = findNode(child, nodeId)
                if result is not None:
                    return result
        return None

    if ParentId == 'root': ##根节点
        tree = {
            'ViewId':ViewId,
            'children':[],
        }
    else: ##子节点
        ### 首先找到父节点
        parent = findNode(tree, ParentId)
        parent['children'].append({
            'ViewId':ViewId,
            'children':[],
        })    
    ## 保存修改后的Tree
    jsonTree = json.dumps(tree)
    tempFile = open('job/' + JobId + '/meta/' + 'tree.json', 'w')
    tempFile.write(jsonTree)
    tempFile.close()

## 获取deleteCells信息
def getDeleteCells(JobId):
    with open('job/' + JobId + '/meta/' + 'deleteCells.json','r',encoding = 'utf-8') as f:
        result = json.load(f)
    return result
    
## 设置deleteCells信息
def setDeleteCells(JobId,deleteCells):
    meta_json = json.dumps(deleteCells)
    tempFile = open('job/' + JobId + '/meta/' + 'deleteCells.json', 'w',encoding = 'utf-8')
    tempFile.write(meta_json)
    tempFile.close()
    return True



#
# 基因推荐
#


## 获取gene sets信息（不包含实际的内容）
def getGeneSetsInfo():
    with open('resource/gene_sets/meta.json','r',encoding = 'utf-8') as f:
        gene_set_info = json.load(f)
    return gene_set_info
## 获取指定物种的gene set（包含实际的内容）
def getGeneSet(organism,gene_set_name):
    gene_set_info = getGeneSetsInfo()
    
    gene_set_index = [item['name'] for item in gene_set_info[organism]].index(gene_set_name)

    with open('resource/gene_sets/' + organism + '/' + gene_set_info[organism][gene_set_index]['file'],'r',encoding = 'utf-8') as f:
        gene_sets = json.load(f)
    return gene_sets



#
# 工具函数
#

## 创建一个空文件
def createFile(path):
    parentPath = path[0:path.rfind("/")]
    if not os.path.isdir(parentPath):
        os.mkdir(parentPath)
    if not os.path.isfile(path):
        fd = open(path, mode="w", encoding="utf-8")
        fd.close()
        return True # 文件不存在，并且创建文件成功  
    else:
        return False # 文件已经存在、创建失败//TODO

##从adata计算距离矩阵TD
def calculateTD(X):
    _X = X
    if hasattr(_X,'A'):
        _X = _X.A
    TD1d = pdist(_X, 'euclidean')
    TD = squareform(TD1d, force='no', checks=True)

    return TD

def arr_to_json(data, group):
    jsonData = []
    data = data.tolist()
    for index, item in enumerate(data):
        jsonData.append({'id': group[index][0], 'data': item, 'group': group[index][1]})
    return jsonData

def filter_data(data, group, filter):
    groupID = group[:, 0]
    filter = np.array(filter)
    group = group[np.isin(groupID, filter)]
    data = data[np.isin(groupID, filter)]
    return data, group

def sort_data(data, group, gene):
    pcaDR = PCA(n_components=1)
    embedding = pcaDR.fit_transform(data).flatten()
    sortIndex = np.argsort(-embedding)
    data = data[sortIndex]
    group = group[sortIndex]
    gene = gene[sortIndex]
    return data, group, gene

def calculateSilhouetteScore(posArr,groupArr):

    return float(silhouette_score(posArr,groupArr.tolist(),metric='euclidean'))

def calculateARI(adata):
    ## TODO 找到有标签的数据后再写
    return

def calculateNMI(adata):
    ## TODO 找到有标签的数据后再写
    return

def calculateLocalScores(posArr,groupArr):
    result = {}
    posArr = np.array(posArr)
    groups = np.unique(groupArr).tolist()

    ## SSE
    for group in groups:
        filterPosArr = posArr[groupArr == group]
        mean_pos = np.mean(filterPosArr,0)
        SSE = 0
        for pos in filterPosArr:
            SSE += np.dot((mean_pos - pos),(mean_pos - pos))
        result[group] = float(SSE/len(filterPosArr))
    return result


#
# 功能函数
#
def translateMouseGeneToHumanGene(gene_list,value_list):## 小鼠基因转换为人类基因
    '''
    params:
        gene_list: a list of gene name
        value_list: a list of gene score(z-score in marker)
    return:
        gene_list: a list of gene name after translation
        value_list: a list of gene score(z-score in marker) translation

    '''
    check_result = mousipy.check_orthologs(gene_list)
    '''
    对于check-orthologs的结果，有以下四种情况，对应四种不同的基因处理方法：
        1. direct：一对一的同源 -> 直接转换
        2. multiple：一对多的同源
        3. no_hit：能在biomart中找到，但是没有同源
        4. no_index：在biomart中找不到 -> 除了一些在不可能转换的基因，其他直接转大写
        这其中还隐含了多对一的关系，在z-score的层面上，我们把他们转换基因合并取平均
    '''
    direct = check_result[0]
    multiple = check_result[1]
    no_hit = check_result[2]
    no_index = check_result[3]

    raw_gene_value_dict = dict(zip(gene_list,value_list)) ##原始小鼠基因查询dict
    new_gene_list = []
    new_value_list = []
    
    ## direct
    for gene in direct:
        new_gene_list.append(direct[gene])
        new_value_list.append(raw_gene_value_dict[gene])
    ## multiple
    for gene in multiple:
        new_gene_list.extend(multiple[gene])
        new_value_list.extend([raw_gene_value_dict[gene] for i in range(0,len(multiple[gene]))])
    ## no index
    for gene in no_index:
        if gene[:2] == 'Gm':
            continue
        if 'Rik' in gene:
            continue
        if gene[:2] == 'RP':
            continue
        if 'Hist' in gene:
            continue
        if 'Olfr' in gene:
            continue
        if '.' in gene:
            continue
        new_gene_list.append(gene.upper())
        new_value_list.append(raw_gene_value_dict[gene])
    ## collapse duplicate genes
    collapse_duplicate_mean = pd.Series(new_value_list,index=new_gene_list).groupby(by=new_gene_list,sort=False).mean()
    new_gene_list = collapse_duplicate_mean.index.tolist()
    new_value_list = collapse_duplicate_mean.values.tolist()
    return new_gene_list,new_value_list



# 
# 关键函数
# 

## 将adata转换为所需的数据，并以json的形式返回
def getResponseFromAdata(adata):

    ## JobId
    JobId = adata.uns['JobId']
    ## ViewId
    ViewId = adata.uns['ViewId']
    ## ParentId
    ParentId = adata.uns['ParentId']

    ## 读取元数据
    metaData = getViewMetaData(JobId,ViewId)

    ## 考虑deleteCells进行细胞过滤
    raw_adata = adata
    remain = list(set(adata.obs.index) - set(getDeleteCells(JobId)))
    if len(remain) == 0: ## 删除到无细胞剩余，返回空数据
        return {"cellData": [],
                "groups": [], 
                "genes":[],
                "chosenData": [],
                'globalScores':None,
                'localScores':None,
                'TI':{},
                'CC':{},
                "MK": [],
                'ViewId':ViewId,
                'ParentId':ParentId,
                'dendrogram':[],#adata.uns['dendrogram']['linkage'].tolist()},
                'paramsObj':adata.uns['params'],
                'raw_embedding_range':metaData['raw_embedding_range'],
            }
    else:
        adata = adata[remain]

    ## cells
    cells_embedding = adata.obsm['embedding']
    cells_id = adata.obs.index
    cells_label = adata.obs.label
    cells = [{'id': cells_id[index], 
              'pos': cells_embedding[index].tolist(), 
              'group': cells_label[index]} for index, pos in enumerate(cells_embedding)]


    ## groups
    groups_id = adata.obs['label'].cat.categories
    ### 计算聚类中心点
    groupFrame = pd.DataFrame(adata.obsm['embedding'],columns=['X','Y'])
    groupFrame['group'] = cells_label.tolist()
    group_X_mean = groupFrame.groupby('group')['X'].agg(np.mean)
    group_Y_mean = groupFrame.groupby('group')['Y'].agg(np.mean)
    ### 打包groups数据
    groups = [{'id':id,
               'name':metaData['group_name'][id],
               'color':metaData['group_color'][id],
               'size':cells_label.tolist().count(id),
               'centerX':np.float(group_X_mean[id]),
               'centerY':np.float(group_Y_mean[id])} for id in groups_id]
    
    ## 获取过滤后的group与原有group的匹配布尔矩阵
    raw_groups_id = raw_adata.obs['label'].cat.categories
    remain_group_boolIndex = [raw in groups_id for raw in raw_groups_id]

    #标记基因数据
    markerData = []
    if 'MK' in adata.uns['params'] and len(adata.obs['label'].cat.categories) > 1:
        marker_list = []

        rawMarker = adata.uns['raw_marker']['names']
        for i in range(0,len(remain_group_boolIndex)):
            if remain_group_boolIndex[i]:
                for j in range(0,len(rawMarker)):
                    marker_list.append(rawMarker[j][i] + "@" + raw_groups_id[i])

        df = None
        if hasattr(adata.X,'A'):
            df = pd.DataFrame(adata.X.A,index=adata.obs.index,columns=adata.var.index)
        else:
            df = pd.DataFrame(adata.X,index=adata.obs.index,columns=adata.var.index)
        
        df['__group'] = adata.obs['label']
        dfg = df.groupby('__group')

        raw = adata.uns['count'][np.isin(adata.uns['count'].obs.index,adata.obs.index),np.isin(adata.uns['count'].var.index,adata.var.index)]

        raw_df = None
 
        if hasattr(raw.X,'A'):
            raw_df = pd.DataFrame(raw.X.A,index=raw.obs.index,columns=raw.var.index)
        else:
            raw_df = pd.DataFrame(raw.X,index=raw.obs.index,columns=raw.var.index)
        raw_df['__group'] = adata.obs['label']
        raw_dfg = raw_df.groupby('__group')
        
        for gene in marker_list:
            real_gene = gene[:gene.find("@")] ##TODO 也许在搜索字符串的时候，加一个参数使其从后往前搜索可以提高准确度？
            ## 统计在每个group中的mean
            means = dfg[real_gene].agg(np.mean)
            ## 统计在每个group中的fraction
            fraction = raw_dfg[real_gene].agg(lambda x : 1.0 * x[x!=0].shape[0] / x.shape[0])
                        
            ## 改list
            mean_list = []
            fraction_list = []
            for group in groups_id:
                mean_list.append(float(means[group]))
                fraction_list.append(float(fraction[group]))

            markerData.append({
                    'name':gene,
                    'means':mean_list,
                    'fraction':fraction_list,
            })

    ## 轨迹推断数据
    TI = {}
    if 'TI' in adata.uns['params']:
        if 'paga' in adata.uns['params']['TI']:
            ### 连接数据 
            TI['connectivities'] = adata.uns['paga']['connectivities'].A.tolist() ##注意，这里的index与groups的顺序对应，而不是与索引绑定
            ### 散点数据
            TI['scatter'] = []
            for scatter_pos,scatter_group in zip(adata.obsm['X_draw_graph_fa'].tolist(),groupArr.tolist()):  ##注意，这里与groupArr的绑定主要通过obs中index的顺序，而不是与索引绑定
                TI['scatter'].append({'pos':scatter_pos,'group':scatter_group})
            ### 散点平均数据
            TI['mean'] = []
            group_TIpos = pd.DataFrame(list(map(lambda item:{'X':item['pos'][0],'Y':item['pos'][1],'group':item['group']},TI['scatter']))).groupby('group')
            TI_X_mean = group_TIpos['X'].agg(np.mean)
            TI_Y_mean = group_TIpos['Y'].agg(np.mean)
            for x,y in zip(TI_X_mean.items(),TI_Y_mean.items()):
                TI['mean'].append({'X':x[1],'Y':y[1],'group':x[0]})
        elif 'slingshot' in adata.uns['params']['TI']:
            TI = adata.uns['slingshot']



    ## 细胞通讯数据
    CC = {}
    if 'CC' in adata.uns['params']:
        CC = adata.uns['CC']

    # ## 生成颜色数据

    # # for i in range(0,len(groups)):
    # #     hue = (1.0 * i / len(groups) + 0.136) % 1
    # #     startuation = 0.8
    # #     lightness = 0.6
    # #     rgb = matplotlib.colors.hsv_to_rgb([hue,startuation,lightness]).tolist()
    # #     r = hex(int(256*rgb[0]))
    # #     g = hex(int(256*rgb[1]))
    # #     b = hex(int(256*rgb[2]))

    # #     # ## 转10进制
    # #     # r = int(r,16)
    # #     # g = int(g,16)
    # #     # b = int(b,16)

    # #     # ## 调整颜色
    # #     # alpha = 0.5 #偏移系数 1表示不偏移
    # #     # baseColor = 255 # 想要变暗设置为0，想要变亮设置为255
    # #     # r = int(alpha * r + (1 - alpha) * baseColor)
    # #     # g = int(alpha * g + (1 - alpha) * baseColor)
    # #     # b = int(alpha * b + (1 - alpha) * baseColor)

    # #     # ## 转16进制
    # #     # r = hex(r)
    # #     # g = hex(g)
    # #     # b = hex(b)

    # #     groups[i]['color'] = '#' + str(r)[2:] + str(g)[2:] + str(b)[2:]

    # # colors = sns.color_palette("colorblind",n_colors=len(groups)).as_hex()
    # # colors = sns.color_palette("Paired",n_colors=len(groups)).as_hex()

    # colors = sns.color_palette("colorblind",n_colors=len(groups)).as_hex()
    # for i in range(0,len(groups)):
    #     # ## 变暗
    #     # alpha = 0.5 #偏移系数 1表示不偏移
    #     # baseColor = 0 # 想要变暗设置为0，想要变亮设置为255
    #     # changedColor = [int(alpha * x * 255 + (1 - alpha) * baseColor) for x in colors[i]]
        
    #     # r = hex(changedColor[0])
    #     # g = hex(changedColor[1])
    #     # b = hex(changedColor[2])
        
    #     # groups[i]['color'] = '#' + str(r)[2:] + str(g)[2:] + str(b)[2:]
    #     groups[i]['color'] = colors[i]


    ## 计算全局聚类得分
    
    globalScores = {}
    if len(adata.obs['label'].cat.categories) > 1:
        globalScores['Silhouette Score'] = calculateSilhouetteScore(cells_embedding.tolist(),adata.obs['label'])
    else:
        globalScores['Silhouette Score'] = 'No score' ## 这里改了前端也要改

    ## 计算局部聚类得分
    localScores = calculateLocalScores(cells_embedding.tolist(),adata.obs['label'])    

    

    ## 装填结果
    result = {}
    ### 装入视图ID
    result['ViewId'] = ViewId
    ### 装入父亲视图ID
    result['ParentId'] = ParentId
    ### 装入参数数据
    result['paramsObj'] = adata.uns['params']
    ### 装入细胞表达信息
    result['cellData'] = cells
    ### 装入投影原始范围数据
    result['raw_embedding_range'] = getViewMetaData(JobId,ViewId)['raw_embedding_range']
    ### 装入默认基因数据
    result['defaultGene'] = adata.var.index.tolist()[0]
    ### 装入group信息
    result['groups'] = groups
    ### 装入基因信息
    result['genes'] = adata.var.index.tolist()
    ### 装入全局投影分数
    result['globalScores'] = globalScores
    ### 装入局部投影分数
    result['localScores'] = localScores
    ### 装入标志基因数据
    result['MK'] = markerData
    ### 装入轨迹推断数据
    result['TI'] = TI
    ### 装入细胞通讯数据
    result['CC'] = CC
    ### 装入group分层数据
    result['dendrogram'] = [] #adata.uns['dendrogram']['linkage'].tolist()},
    ### 装入细胞选择数据
    result['chosenData'] = []


    return result