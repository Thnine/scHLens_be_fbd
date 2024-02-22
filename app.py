from cProfile import label
import enum
from unittest import result
from flask import Flask, request, jsonify,send_file,make_response
import json
from flask.json import JSONEncoder
from zmq import DRAFT_API
from lib.pipline import *
from lib.utils import *
from lib.vars import color_list
import scanpy as sc
import numpy as np
import pandas as pd
import umap
import time
import os
from lib import openTSNEStab
import config
import shutil
import traceback
import datetime
app = Flask(__name__)

# 打开debug模式
app.debug = True
app.config.update(DEBUG=True)
app.config.from_object(config)
app.config.from_pyfile('config.cnf', silent=True)

## 设置json编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16,np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
app.json_encoder = NumpyEncoder


@app.route('/scHLens/api/',methods=['POST','GET'])
def hello_world(): 
    return 'Hello World!'


'''
关键
'''
## 流水线计算（global | local）
@app.route('/scHLens/api/runPipeline', methods=['POST'])
def runPipeline():
    reqParams = json.loads(request.get_data())

    try:
        if 'global' in reqParams['type']: ## 全局
            adata = globalPipeline(reqParams)
        elif 'local' in reqParams['type']: ## 局部
            adata = localPipeline(reqParams)
    except Exception:
        traceback.print_exc()
        return {
            'JobId':reqParams['JobId'],
            'ViewId':'',
            'status':0, ## 失败
            'message':'error',
        }
    else:
        return {
            'JobId':adata.uns['JobId'],
            'ViewId':adata.uns['ViewId'],
            'status':1, ##成功
            'message':'success'
        }


## 从视图的adata中获取视图数据给前端
@app.route('/scHLens/api/fetchViewData', methods=['POST'])
def fetchViewData():
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']

    adata = readCache(JobId,ViewId,'Query')

    dict_result = getResponseFromAdata(adata)



    return dict_result


## 合并视图
@app.route('/scHLens/api/mergeViews',methods=['POST'])
def mergeViews():

    ## 读取数据
    reqParams = json.loads(request.get_data())

    try:

        JobId = reqParams['JobId']
        globalViewId = reqParams['globalViewId']
        localViewIdList = reqParams['localViewIdList']
        globalAdata = readCache(JobId, globalViewId, 'Query')
        globalAdata.uns['log1p']['base'] = None ## 算是存取的一个bug


        '''
        embedding merge
        '''
        globalTD = globalAdata.uns['TD']
        ReplaceIndex = set([])
        ## 距离矩阵替换
        for localViewId in localViewIdList:
            localAdata = readCache(JobId, localViewId, 'Query')
            localTD = localAdata.uns['TD']
            indexArr = []
            for index,item in enumerate(globalAdata.obs.index):
                if item in localAdata.obs.index:
                    indexArr.append(index)
                    ReplaceIndex.add(index)
            i = 0
            for index,item in enumerate(globalTD):
                if index in indexArr:
                    globalTD[index,indexArr] = localTD[i]
                    i+=1
        
        globalAdata.uns['TD'] = globalTD

        
        ## TODO remain的计算方法还要再斟酌
        remain = np.array(list(set(range(len(globalAdata.obs.index))) - ReplaceIndex))


        ## 稳定性降维
        embedding = None
        if 'T-SNE' in globalAdata.uns['params']['DR']:
            embedding = openTSNEStab.TSNE(metric="precomputed",random_state= 0,initialization='random').fit(X=globalTD,remain=remain)
            embedding = np.array(embedding)
        elif 'UMAP' in globalAdata.uns['params']['DR']:
            embedding = umap.UMAP(min_dist = 0.5, metric="precomputed", random_state= 0).fit_transform(globalTD)
            embedding = np.array(embedding)
        elif 'PCA' in globalAdata.uns['params']['DR']:
            ## TODO 请补充PCA降维合并
            embedding = None
        globalAdata.obsm['embedding'] = embedding
        

        '''
        label merge
        '''

        ## cluster标签更新
        globalLabel = globalAdata.obs.label.astype(object,copy=True) ## 拷贝一份
        globalMetaData = getViewMetaData(JobId,globalViewId)
        AnnoMap = {} # 局部标签新id -> anno
        colorMap = {} # 局部标签新id -> 颜色
        name_count = 0 
        for localViewId in localViewIdList:
            localAdata = readCache(JobId, localViewId, 'Query')
            localLabel = localAdata.obs.label.copy()
            localMetaData = getViewMetaData(JobId,localViewId)
            IdMap = {} # cluster -> new id
            ## 给local的group分配临时新id
            for cluster in localLabel.cat.categories:
                new_id = 'n_' + str(name_count)
                name_count+=1
                IdMap[cluster] = new_id
                AnnoMap[new_id] = localMetaData['group_name'][cluster]
            localLabel = localLabel.cat.rename_categories(IdMap)
            globalLabel[localLabel.index] = localLabel


        globalLabel = globalLabel.astype('category')
 
        ## modify metaData
        globalLabelCategories = globalLabel.cat.categories.tolist()
        modify_id_map = {} ## id调整的映射
        ### remove unexist group id
        meta_ids = list(globalMetaData['group_name'].keys())
        for id in meta_ids: 
            if id not in globalLabel.cat.categories:
                del globalMetaData['group_name'][id]
        meta_ids = list(globalMetaData['group_color'].keys())
        for id in meta_ids: 
            if id not in globalLabel.cat.categories:
                del globalMetaData['group_color'][id]
        ### modify id
        for id in globalLabelCategories:
            if id not in globalMetaData['group_name']:
                modify_id = 'c_' + str(globalMetaData['history_group_num'])
                globalMetaData['history_group_num'] += 1
                modify_id_map[id] = modify_id
                globalLabel = globalLabel.cat.rename_categories({id:modify_id})
        ### add new annotation
        for id in globalLabelCategories:
            if id not in globalMetaData['group_name']:
                modify_id = modify_id_map[id]
                globalMetaData['group_name'][modify_id] = AnnoMap[id]
        ### modify new group color
        for id in globalLabelCategories:
            if id not in globalMetaData['group_color']:
                modify_id = modify_id_map[id]
                exist_colors = globalMetaData['group_color'].values()
                choose_color = 'black'
                for candidate_color in color_list:
                    if candidate_color not in exist_colors:
                        choose_color = candidate_color
                        break
                globalMetaData['group_color'][modify_id] = choose_color
                
            
        globalAdata.obs.label = globalLabel

        ## TODO 重做marker
        if 'MK' in globalAdata.uns['params'] and len(globalAdata.obs['label'].cat.categories) > 1:
            globalAdata = MK(globalAdata)

        ## save
        saveViewMetaData(JobId,globalViewId,globalMetaData)
        saveCache(globalAdata,JobId,globalViewId,'Query')

    except Exception:
        traceback.print_exc()
        result = {
            'JobId':JobId,
            'ViewId':globalViewId,
            'status':0, ##报错
            'message':'error'

        }
    else:
        result = {
            'JobId':JobId,
            'ViewId':globalViewId,
            'status':1, ##成功
            'message':'success'
        }

    return result


'''
Job
'''

## 创建Job
@app.route('/scHLens/api/createNewJob',methods=['POST'])
def createJob():
    JobId = initJob()
    return JobId

## 读取Job
@app.route('/scHLens/api/loadExistJob',methods=['POST'])
def loadJob():
    # 检索JobId是否存在
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']
    if os.path.exists('./job/'+ JobId):##JobId存在
        ##读取Tree信息
        tree = getTree(JobId)

        if not tree:
            return json.dumps({})

        def attachDataToTree(root):
            curView = root['ViewId']
            curData = readCache(JobId, curView, 'Query')
            curResult = getResponseFromAdata(curData)
            root['data'] = curResult
            if len(root['children']) != 0:
                for child in root['children']:
                    attachDataToTree(child)

        attachDataToTree(tree)
    
        ## 解决numpy数据无法进行json编码的问题
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32,
                                    np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_,)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        return json.dumps(tree,cls=NpEncoder)
    else:##JobId不存在
        return 'unexist'

## 导出job
@app.route('/scHLens/api/exportJob',methods=['POST'])
def exportJob():
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']

    ##压缩文件
    if not os.path.exists('./job_attachment/' + JobId):
        os.makedirs('job_attachment/' + JobId)
    shutil.make_archive(base_name='job_attachment/' + JobId + '/' + JobId, 
                        format='zip',
                        root_dir='./job',
                        base_dir='./' + JobId)


    path = './job_attachment/' + JobId + '/' + JobId +  '.zip'

    response = send_file(path,as_attachment=True)

    return response

## 上传job
@app.route('/scHLens/api/uploadJob',methods=['POST'])
def uploadJob():

    _file = request.files.get('file')
    JobId = request.form.get('JobId')

    ## 保存压缩包
    archivePath = './job_attachment/' + JobId + '/' + JobId +  '.zip'
    if not os.path.exists('./job_attachment/' + JobId):
        os.makedirs('job_attachment/' + JobId)
    _file.save(archivePath)

    ## 解压压缩包
    shutil.unpack_archive(archivePath,'./job/')

    return 'success'


'''
数据集
'''

## 获取数据集信息
@app.route('/scHLens/api/fetchDatasets',methods=['POST'])
def fetchDatasets():
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']

    datasetConfigs = []

    ##导入sample数据集
    for sample in os.listdir('./sample'):
        if os.path.exists('./sample/' + sample + '/config.json'):
            with open('./sample/' + sample + '/config.json','r',encoding = 'utf-8') as f:
                sample_config = json.load(f)
                datasetConfigs.append(sample_config)
    ##导入user上传的数据集
    for dataset in os.listdir('./job/' + JobId + '/dataset'):
        if os.path.exists('./job/' + JobId + '/dataset/' + dataset + '/config.json'):
            with open('./job/' + JobId + '/dataset/' + dataset + '/config.json','r',encoding = 'utf-8') as f:
                dataset_config = json.load(f)
                datasetConfigs.append(dataset_config)

    return jsonify(datasetConfigs)

## 上传数据集
@app.route('/scHLens/api/upload',methods=['POST'])
def upload():
    upfile = request.files['file']
    dataset_name = request.form['name']
    dataset_type = request.form['type']
    JobId = request.form['JobId']
    filename = upfile.filename
    
    # 构造数据集配置文件
    
    config = {}
    config['name'] = dataset_name
    config['type'] = dataset_type
    config['from'] = 'user'

    # 初始化文件夹
    if not os.path.exists('job/' + JobId + '/dataset/' + dataset_name):
        os.mkdir('job/' + JobId + '/dataset/' + dataset_name)

    # 装入文件
    upfile.save('./job/' + JobId + '/dataset/' + dataset_name + '/' + filename)
    if not os.path.exists('job/' + JobId + '/dataset/'  + dataset_name + '/config.json'):
        config_json = json.dumps(config)
        tempFile = open('job/' + JobId + '/dataset/' + dataset_name + '/config.json', 'w')
        tempFile.write(config_json)
        tempFile.close()

    return 'success'

## 上传数据集
@app.route('/api/upload',methods=['POST'])
def upload1():
    upfile = request.files['file']
    dataset_name = request.form['name']
    dataset_type = request.form['type']
    JobId = request.form['JobId']
    filename = upfile.filename
    
    # 构造数据集配置文件
    
    config = {}
    config['name'] = dataset_name
    config['type'] = dataset_type
    config['from'] = 'user'

    # 初始化文件夹
    if not os.path.exists('job/' + JobId + '/dataset/' + dataset_name):
        os.mkdir('job/' + JobId + '/dataset/' + dataset_name)

    # 装入文件
    upfile.save('./job/' + JobId + '/dataset/' + dataset_name + '/' + filename)
    if not os.path.exists('job/' + JobId + '/dataset/'  + dataset_name + '/config.json'):
        config_json = json.dumps(config)
        tempFile = open('job/' + JobId + '/dataset/' + dataset_name + '/config.json', 'w')
        tempFile.write(config_json)
        tempFile.close()

    return 'success'

## 保存子数据集
@app.route('/scHLens/api/saveLocalDataset',methods=['POST'])
def saveLocalDataset(): #TODO 没有考虑数据融合的情况
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    chosenData = reqParams['chosenData']

    dataset = readData(readCache(JobId, ViewId, 'Query').uns['params']['dataset'], JobId)
    
    ##过滤数据
    dataset = dataset[chosenData,:]

    ##导出为独立的数据集
    path = 'job/' + JobId + '/view/' + ViewId +  '/export/' + 'export' + '.h5ad' #TODO 这里的命名重复问题
    dataset.write(path)

    response = send_file(path,as_attachment=True,attachment_filename='export.h5ad')

    return response


'''
Group
'''

## 更改组名
@app.route('/scHLens/api/updateGroupName',methods=['POST'])
def updateGroupName():
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    newGroupNames = reqParams['group_name']
    metaData = getViewMetaData(JobId, ViewId)
    metaData['group_name'] = newGroupNames
    saveViewMetaData(JobId, ViewId, metaData)

    return 'success'

## 更改颜色
@app.route('/scHLens/api/updateGroupColor',methods=['POST'])
def updateGroupColor():
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    newGroupColors = reqParams['group_color']
    metaData = getViewMetaData(JobId, ViewId)
    metaData['group_color'] = newGroupColors
    saveViewMetaData(JobId, ViewId, metaData)

    return 'success'



'''
查询
'''

# 根据字符串查询符合匹配条件的基因数组
@app.route('/scHLens/api/queryCandidateGeneList', methods=['POST'])
def queryCandidateGeneList():
    reqParams = json.loads(request.get_data())
    ## read cache
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')
    geneList = adata.var.index[adata.var.index.str.contains(reqParams['geneMatch'], case=False)].tolist()
    return jsonify(geneList)

# 根据基因名查询该基因的表达值范围
@app.route('/scHLens/api/queryGeneValueRange', methods=['POST'])
def queryGeneValueRange():
    reqParams = json.loads(request.get_data())

    ## read cache
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')
    
    #adata = qualityControl(adata,reqParams)
    geneValueArr = None
    if hasattr(adata[:, [reqParams['geneName']]].X,'A'):
        geneValueArr = adata[:, [reqParams['geneName']]].X.A.flatten()
    else: 
        geneValueArr = adata[:, [reqParams['geneName']]].X.flatten()
    return jsonify([float(geneValueArr.min()), float(geneValueArr.max())])

# 根据基因名查询每个细胞中的表达值数组
@app.route('/scHLens/api/queryGeneValueList', methods=['POST'])
def queryGeneValueList():
    reqParams = json.loads(request.get_data())

    ## read cache
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')

    #adata = qualityControl(adata,reqParams)
    keys = adata.obs_names.tolist()
    geneValueArr = None
    if hasattr(adata[:, [reqParams['geneName']]].X,'A'):
        geneValueArr = adata[:, [reqParams['geneName']]].X.A.flatten().tolist()
    else: 
        geneValueArr = adata[:, [reqParams['geneName']]].X.flatten().tolist()
    return jsonify(dict(zip(keys, geneValueArr)))

# 根据过滤器中的条件查询过滤细胞
@app.route('/scHLens/api/queryFilteredCellList', methods=['POST'])
def requestFilteredCellList():
    reqParams = json.loads(request.get_data())

    ## read cache
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')

    #adata = qualityControl(adata,reqParams)
    adata = adata[:, reqParams['geneName']]
    for index, item in enumerate(reqParams['geneRange']):
        adata = adata[(adata.X.A[:, index] >= item[0]) & (adata.X.A[:, index] <= item[1])]
    return jsonify(adata.obs_names.tolist())




'''
细胞类型推荐
'''
## 查询基因集信息
@app.route('/scHLens/api/queryGeneSets', methods=['POST'])
def queryGeneSets():
    reqParams = json.loads(request.get_data())

    gene_sets_info = getGeneSetsInfo()
    for org in gene_sets_info.keys():
        gene_sets_info[org] = list(map(lambda x:x['name'],gene_sets_info[org]))

    return jsonify(gene_sets_info)

## 查询gsea推荐结果(Prerank)
@app.route('/scHLens/api/queryGSEA', methods=['POST'])
def queryGSEA():
    reqParams = json.loads(request.get_data())

    ## read gene sets
    organism = reqParams['organism']
    gene_set_name = reqParams['gene_set_name']
    gene_sets = getGeneSet(organism,gene_set_name)

    ## read cache
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')

    ## 判断合法性
    if 'raw_marker' not in adata.uns: ## 是否已经进行marker
        return jsonify([])
    

    ## make rank
    cluster_id = reqParams['cluster_id']
    gene_list = adata.uns['raw_marker']['names'][cluster_id].tolist()
    value_list = adata.uns['raw_marker']['scores'][cluster_id].tolist()
    if 'logfoldchanges' in adata.uns['raw_marker']: ## 按照logfoldchange>0进行过滤
        foldchange_list = adata.uns['raw_marker']['logfoldchanges'][cluster_id].tolist()
        new_gene_list = []
        new_value_list = []
        for i in range(0,len(gene_list)):
            if foldchange_list[i] > 0:
                new_gene_list.append(gene_list[i])
                new_value_list.append(value_list[i])
        gene_list = new_gene_list
        value_list = new_value_list
    if organism == 'Mouse':## 鼠基因转换
        gene_list,value_list = translateMouseGeneToHumanGene(gene_list,value_list)
    rnk = pd.Series(value_list,index=gene_list)

    ## run gsea preRank
    gsea_result = gp.prerank(rnk=rnk,gene_sets=gene_sets,min_size=1,max_size=len(gene_list)).res2d

    ## filter
    q_threshold = reqParams['q_threshold']
    p_threshold = reqParams['p_threshold']
    top = reqParams['top']
    gsea_result = gsea_result[gsea_result['ES'] > 0]
    if q_threshold != 'all':
        gsea_result = gsea_result[gsea_result['FDR q-val'] < q_threshold]
    if p_threshold != 'all':
        gsea_result = gsea_result[gsea_result['FWER p-val'] < p_threshold]
    if top != 'all':
        gsea_result = gsea_result.head(top)

    ## sort
    gsea_result.sort_values(by=['NES'],ascending=False,inplace=True)

    return jsonify(gsea_result.to_dict(orient='records'))

## 查询enricher推荐结果
@app.route('/scHLens/api/queryEnricher', methods=['POST'])
def queryEnricher():
    reqParams = json.loads(request.get_data())

    ## read gene sets
    organism = reqParams['organism']
    gene_set_name = reqParams['gene_set_name']
    gene_sets = getGeneSet(organism,gene_set_name)

    ## read cache
    JobId = reqParams['JobId'] 
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')

    ## 判断合法性
    if 'raw_marker' not in adata.uns: ## 是否已经进行marker
        return jsonify([])
    
    ## get gene_list（为了省事，这里连着value_list一起计算了）
    cluster_id = reqParams['cluster_id']
    gene_list = adata.uns['raw_marker']['names'][cluster_id].tolist()
    value_list = adata.uns['raw_marker']['scores'][cluster_id].tolist()
    if 'logfoldchanges' in adata.uns['raw_marker']: ## 按照logfoldchange>0进行过滤
        foldchange_list = adata.uns['raw_marker']['logfoldchanges'][cluster_id].tolist()
        new_gene_list = []
        new_value_list = []
        for i in range(0,len(gene_list)):
            if foldchange_list[i] > 0:
                new_gene_list.append(gene_list[i])
                new_value_list.append(value_list[i])
        gene_list = new_gene_list
        value_list = new_value_list
    if organism == 'Mouse':## 鼠基因转换
        gene_list,value_list = translateMouseGeneToHumanGene(gene_list,value_list)

    ## run enricher
    enrichr_result = gp.enrichr(gene_list,
                        gene_sets=gene_sets,
                        outdir=None).res2d

    ## filter
    p_threshold = reqParams['p_threshold']
    top = reqParams['top']

    if p_threshold != 'all':
        enrichr_result = enrichr_result[enrichr_result['Adjusted P-value'] < p_threshold]
    if top != 'all':
        enrichr_result = enrichr_result.head(top)

    ## sort
    enrichr_result.sort_values(by=['Combined Score'],ascending=False,inplace=True)

    return jsonify(enrichr_result.to_dict(orient='records'))




'''
其他
'''
# 基因推荐（marker）
@app.route('/scHLens/api/recommendGene',methods=['POST'])
def recommendGene():
    reqParams =  json.loads(request.get_data())

    ## read cache
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    adata = readCache(JobId, ViewId, 'Query')

    adata.uns['log1p']['base'] = None ## 算是存取的一个bug
    if reqParams['mode'] == 'HighlyVariable':
        return jsonify(adata[:,adata.var.highly_variable].var.index.tolist())
    elif reqParams['mode'] == 'Marker':
        np.unique(adata.obs['label']).tolist()
        return jsonify([])

## 用户打开的实例页面被关闭的事件触发函数
@app.route('/scHLens/api/InstanceClose', methods=['POST'])
def InstanceClose():
    reqParams = json.loads(request.get_data())
    
    return jsonify('close')

## 删除选择的节点
@app.route('/scHLens/api/updateDeleteCells', methods=['POST'])
def updateDeleteCells():
    reqParams = json.loads(request.get_data())
    JobId = reqParams['JobId']
    ViewId = reqParams['ViewId']
    chosenData = reqParams['chosenData']

    ## 更新deleteCells
    deleteCells = getDeleteCells(JobId)
    newDeleteCells = list(set(deleteCells) | set(chosenData))
    setDeleteCells(JobId,newDeleteCells)

    ## 视图更新Tree
    EmbeddingTree = getTree(JobId)
    def addEmbeddingToTree(cur):
        adata = readCache(JobId,cur['ViewId'],'Query')
        newResult = getResponseFromAdata(adata) ##由于deleteCells更新，所以从adata提取的数据也要取最新的
        cur['updateData'] = {
                'cellData':newResult['cellData'],
                'groups':newResult['groups'],
                'MK':newResult['MK'],
                'globalScores':newResult['globalScores'],
                'localScores':newResult['localScores'],
            }
        for child in cur['children']:
            addEmbeddingToTree(child)
    addEmbeddingToTree(EmbeddingTree)
    

    return jsonify(EmbeddingTree)


## 提交意见
@app.route('/scHLens/api/sendMessage', methods=['POST'])
def sendMessage():
    form = json.loads(request.get_data())
    first_name = form['first_name']
    last_name = form['last_name']
    email = form['email']
    message = form['message']
    
    if not os.path.exists('message'):
        os.makedirs('message')
    
    now_time = datetime.datetime.now()
    ## 四位随机ID
    characterList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ID = ''.join(random.choice(characterList) for _ in range(4))
    filename = now_time.strftime("%Y%m%d%H%M%S") + '-' + ID + '.txt'
    
    store_str = \
        'First name: ' + first_name + '\n' + \
        'Last name: ' + last_name + '\n' + \
        'Email: ' + email + '\n' + \
        'Time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n' + \
        'Message: ' + message
    
    with open('message/' + filename,'w',encoding='utf-8') as f:
        f.write(store_str)
    return jsonify(1)
        


if __name__ == '__main__':
    ## 初始化
    initApp()

    app.run()


