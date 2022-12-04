import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from torch_geometric.datasets import Planetoid,TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
import datetime
import matplotlib.pyplot as plt
import scipy


def read_pickle(path):
    objects = None
    if path=='':
        return object
    with (open(path, "rb")) as openfile:
        try:
            objects=pickle.load(openfile)
        except EOFError:
            print("Error in reading pickle file, check path and pickle file")
    return objects

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection

    if union == 0:
        return float(1)
        
    return float(intersection) / union


def get_jaccard(expl_dict,expl,index):
    assert(len(expl)!=0)
    n_methods=len(expl)
    jacard = np.zeros((n_methods, n_methods))
    node_indices=expl_dict[index]
    count = 0

    for k in node_indices:

        count += 1

        # jacard[0, 1] += jaccard(imp_nodes_ig[k], imp_nodes_gnn[k])
        # jacard[0, 2] += jaccard(imp_nodes_ig[k], imp_nodes_pge[k])
        # jacard[0, 3] += jaccard(imp_nodes_ig[k], imp_nodes_cam[k])
        # jacard[0, 4] += jaccard(imp_nodes_ig[k], imp_nodes_pgm[k])

        # jacard[1, 2] += jaccard(imp_nodes_gnn[k], imp_nodes_pge[k])
        # jacard[1, 3] += jaccard(imp_nodes_gnn[k], imp_nodes_cam[k])
        # jacard[1, 4] += jaccard(imp_nodes_gnn[k], imp_nodes_pgm[k])


        # jacard[2, 3] += jaccard(imp_nodes_pge[k], imp_nodes_cam[k])
        # jacard[2, 4] += jaccard(imp_nodes_pge[k], imp_nodes_pgm[k])

        # jacard[3, 4] += jaccard(imp_nodes_cam[k], imp_nodes_pgm[k])

        for i in range(0,n_methods-1):
            for j in range(i+1,n_methods):
                imp_nodes_for_expl_i=expl_dict[expl[i]][k]
                imp_nodes_for_expl_j=expl_dict[expl[j]][k]
                jacard[i,j]+=jaccard(imp_nodes_for_expl_i,imp_nodes_for_expl_j)


    jacard = (jacard + jacard.T)/(count)

    for i in range(n_methods):

        jacard[i,i] = 1

    return jacard,expl


def plot_jacard(jacard,labels,title,path=''):
    # labels = ["IG", "GNN", "PGE", "CAM", "PGM"]
    plt.clf()
    jacard_df = pd.DataFrame(jacard, index = labels, columns = labels)
    heatmap_fig=sns.heatmap(jacard_df, annot=True)
    heatmap_fig.set(title=title)
    heatmap_fig.set(xlabel="", ylabel="")
    heatmap_fig.xaxis.tick_top()
    if path !='':
        print("Saving {} file at {}".format(title,path))
        plt.savefig(path)

    return 

def obtain_hits_GC(dataset,expl_dict,expl=[]):
    assert(len(expl)!=0)
    
    expl_dict_auth={}
    expl_dict_hubsc={}

    expl_dict_hubsc['graph_indices']=expl_dict['graph_indices']
    expl_dict_auth['graph_indices']=expl_dict['graph_indices']

    for graph_idx in expl_dict['graph_indices']:
        data=dataset[graph_idx]
        # print(graph_idx)
        G = to_networkx(data)
        hubs = nx.hits(G)[0]
        authorities = nx.hits(G)[1]
        # print("size hubs:",len(hubs))
        # print("size auth:",len(authorities))
        # mapping each imp_list for each node_idx to the corresponding hub and authority list
        for expl_type in expl:
            # print(expl_type)
            if expl_type not in expl_dict_auth:
                expl_dict_auth[expl_type]={}
            if expl_type not in expl_dict_hubsc:
                expl_dict_hubsc[expl_type]={}
            # print(graph_idx," ",expl_type," ",expl_dict[expl_type][graph_idx])
            expl_dict_auth[expl_type][graph_idx]=[authorities[x] for x in expl_dict[expl_type][graph_idx]]
            expl_dict_hubsc[expl_type][graph_idx]=[hubs[x] for x in expl_dict[expl_type][graph_idx]]

    
    return expl_dict_hubsc,expl_dict_auth

def obtain_hits_NC(data,expl_dict,expl=[]):
    assert(len(expl)!=0)
    G = to_networkx(data)
    hubs = nx.hits(G)[0]
    authorities = nx.hits(G)[1]
    
    expl_dict_auth={}
    expl_dict_hubsc={}

    expl_dict_hubsc['node_indices']=expl_dict['node_indices']
    expl_dict_auth['node_indices']=expl_dict['node_indices']

    for node_idx in expl_dict['node_indices']:

        # imp_node_ig = imp_nodes_ig[node_idx]
        # hubs_imp_nodes_ig.append([hubs[x] for x in imp_node_ig])
        # authorities_imp_nodes_ig.append([authorities[x] for x in imp_node_ig])

        # imp_node_gnn = imp_nodes_gnn[node_idx]
        # hubs_imp_nodes_gnn.append([hubs[x] for x in imp_node_gnn])
        # authorities_imp_nodes_gnn.append([authorities[x] for x in imp_node_gnn])

        # imp_node_pge = imp_nodes_pge[node_idx]
        # hubs_imp_nodes_pge.append([hubs[x] for x in imp_node_pge])
        # authorities_imp_nodes_pge.append([authorities[x] for x in imp_node_pge])

        # imp_node_cam = imp_nodes_cam[node_idx]
        # hubs_imp_nodes_cam.append([hubs[x] for x in imp_node_cam])
        # authorities_imp_nodes_cam.append([authorities[x] for x in imp_node_cam])

        # imp_node_pgm = imp_nodes_pgm[node_idx]
        # hubs_imp_nodes_pgm.append([hubs[x] for x in imp_node_pgm])
        # authorities_imp_nodes_pgm.append([authorities[x] for x in imp_node_pgm])
        
        # mapping each imp_list for each node_idx to the corresponding hub and authority list
        for expl_type in expl:
            if expl_type not in expl_dict_auth:
                expl_dict_auth[expl_type]={}
            if expl_type not in expl_dict_hubsc:
                expl_dict_hubsc[expl_type]={}
            expl_dict_auth[expl_type][node_idx]=[authorities[x] for x in expl_dict[expl_type][node_idx]]
            expl_dict_hubsc[expl_type][node_idx]=[hubs[x] for x in expl_dict[expl_type][node_idx]]

    
    return expl_dict_hubsc,expl_dict_auth

def obtain_nx_scores(dataset,expl_dict,expl,index,score_name="degree_centrality"):
    assert(len(expl)!=0)
    
    expl_dict_score={}

    expl_dict_score[index]=expl_dict[index]
    
    data=dataset[0]

    if index=='graph_indices':
        for idx in expl_dict[index]:
            data=dataset[idx]
            # print(graph_idx)
            G = to_networkx(data)
            nx_score_list = []
            if score_name=='degree_centrality':
                nx_score_list=nx.degree_centrality(G)
            else :
                return {}
            # mapping each imp_list for each node_idx to the corresponding hub and authority list
            for expl_type in expl:
                # print(expl_type)
                if expl_type not in expl_dict_score:
                    expl_dict_score[expl_type]={}
                    
                expl_dict_score[expl_type][idx]=[nx_score_list[x] for x in expl_dict[expl_type][idx]]
    elif index=='node_indices':
        data=dataset[0]
        nx_score_list=[]
        G = to_networkx(data)

        if score_name=='degree_centrality':
            nx_score_list=nx.degree_centrality(G)
        else:
            return {}
        for idx in expl_dict[index]:
            # print("size hubs:",len(hubs))
            # print("size auth:",len(authorities))
            # mapping each imp_list for each node_idx to the corresponding hub and authority list
            for expl_type in expl:
                # print(expl_type)
                if expl_type not in expl_dict_score:
                    expl_dict_score[expl_type]={}
                    
                expl_dict_score[expl_type][idx]=[nx_score_list[x] for x in expl_dict[expl_type][idx]]

    
    return expl_dict_score

def get_valid_expl(expl_dict,index):
    # it returns the list of valid explanations available 
    # it checks if for any explanation there is a none entry

    available_expl=list(expl_dict.keys())[1:] # first one is always 'node_indices'
    valid_expl=[]
    node_indices=expl_dict[index]
    is_valid=True
    for expl in available_expl:
        if 'ranking' in expl.split('_'): # does not include any ranking entries eg. ig_ranking,cam_ranking, gcam_ranking
            continue
        is_valid=True
        for node_idx in node_indices:
            if(expl_dict[expl][node_idx]==None):
                is_valid=False # found a none entry
                break
        
        if is_valid:
            valid_expl.append(expl)
    
    return valid_expl

def pearson_corr(list1,list2):
    corr,_= scipy.stats.spearmanr(list1,list2)
    return corr

def calc_agg_score(expl_dict,expl_list,index):
    '''
        calculate the aggregate score for scores for each node_idx/graph_idx for each explainer
    '''
    node_indices=expl_dict[index]
    agg_score={}
    agg_score[index]=node_indices
    for node_idx in node_indices:
        for exp_type in expl_list:
            if exp_type not in agg_score:
                agg_score[exp_type]=[]
            if(len(expl_dict[exp_type][node_idx])!=0):
                agg_score[exp_type].append(sum(expl_dict[exp_type][node_idx])/len(expl_dict[exp_type][node_idx]))
            else:
                agg_score[exp_type].append(0)
    
    return agg_score

def plot_score(agg_score,expl_list,index,xlabel,ylabel,title,path='',type='normal'):
    '''
        Pointwise difference in the agg_score quantities for a random set of indices
    '''
    np.random.seed(12)
    plt.clf()
    x=agg_score[index]
    random_nodes_index=np.random.choice(list(range(0,len(x))), size=10)

    if isinstance(x[0], int):
        random_nodes=[x[i] for i in random_nodes_index]
    else:
        random_nodes=[x[i].item() for i in random_nodes_index]

    xaxis=list(range(0,len(random_nodes)))
    for exp_type in expl_list:
        if type=='log':
            agg_score_for_rand=[np.log(agg_score[exp_type][node]) for node in random_nodes_index]
        else:
            agg_score_for_rand=[agg_score[exp_type][node] for node in random_nodes_index]
        # print(agg_score_for_rand)
        plt.scatter(xaxis,agg_score_for_rand,label=exp_type)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xaxis,labels=random_nodes)
    fig = plt.gcf()
    # fig.set_size_inches(18.5, 10.5)
    if path!='':
        plt.savefig(path)

def compute_cosine(list1,list2):
    '''
        compute cosine distance of two lists/vectors
    '''
    return scipy.spatial.distance.cosine(list1,list2)

def compute_heatmap(expl_dict,expl_list):
    '''
        Function plots heatmap for explainers by computing the cosine distance 
        For each explainer, we have a single vector of a metric (eg aggregate hub score) indexed by the node idx 

    '''
    n=len(expl_list)
    matrix=np.zeros((n,n))
    for i in range(0,n):
        for j in range(i,n):
            matrix[i][j]=compute_cosine(expl_dict[expl_list[i]],expl_dict[expl_list[j]])
            matrix[j][i]=matrix[i][j]

    return matrix

def get_top3(expl_dict,expl_dict_rank,expl_list,index,out='nodes'):
    '''
        Function to obtain the top3 nodes of importance for a particular nodes using the rankings
    '''
    top3={}
    top3[index]=expl_dict[index]
    for idx in expl_dict[index]:
        for expl in expl_list:
            if expl not in top3:
                top3[expl]={}

            ranking_=expl_dict_rank[expl][idx]
            nodes_=expl_dict[expl][idx]
            if(out=='nodes'):
                top3[expl][idx]=[x[0] for x in sorted(list(zip(nodes_,ranking_)), key=lambda x: x[1], reverse=True)[0:3]]
            else:
                top3[expl][idx]=sorted(list(zip(nodes_,ranking_)), key=lambda x: x[1], reverse=True)[0:3]
    return top3

def get_pgm_top3(model_name,dataset_name):
    GNNGraphConv_MUTAG='GNNExplainers/Saved_Explanations/PGM_top3_GNNGraphConv_MUTAG_2022-11-30T21_31_20-98.pkl'
    GNNGraphConv_PROTEINS='GNNExplainers/Saved_Explanations/PGM_top3_GNNGraphConv_PROTEINS_2022-11-30T21_50_14-05.pkl'

    GCN_3L_PROTEINS='GNNExplainers/Saved_Explanations/PGM_top3_GCN_3L_PROTEINS_2022-11-30T21_54_07-66.pkl'
    GCN_3L_MUTAG='GNNExplainers/Saved_Explanations/PGM_top3_GCN_3L_MUTAG_2022-11-30T21_57_06-60.pkl'

    GCN_3L_paths={'PROTEINS':GCN_3L_PROTEINS,'MUTAG':GCN_3L_MUTAG}
    GNNGraphConv_paths={'PROTEINS':GNNGraphConv_PROTEINS,'MUTAG':GNNGraphConv_MUTAG}
    
    file_paths={'GCN_3L':GCN_3L_paths,'GNNGraphConv':GNNGraphConv_paths}

    return read_pickle(file_paths[model_name][dataset_name])

def get_disagreement(model_name,dataset_name,type,path):
    '''
        Function uses the node importance (without ranking) provided for each node in the graph for different explainers.
    
    '''
    if dataset_name not in path.split('_'):
        print('{} provided is not for the {}'.format(path,dataset_name))
        return None
    else :
        if '_' not in model_name:
            if model_name not in path.split('_') :
                print('{} provided is not for the {}'.format(path,model_name))
                return None
        else :
            if model_name!='GCN_3L':
                print('{} provided is not for the {}'.format(path,model_name))
                return None

    
    assert(model_name in ['GCN','GAT','GCN_3L','GNNGraphConv'])
    assert(dataset_name in ['Cora','CiteSeer','MUTAG','PROTEINS'])
    assert(type in ['GC','NC'])
    
    dataset=None
    data=None
    index=''

    if type=='NC':
        dataset = Planetoid(root='GNNModels/data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]  # Get the first graph object.
        index='node_indices'
    elif type=='GC':
        dataset = TUDataset(root='GNNModels/data/TUDataset', name=dataset_name)
        index='graph_indices'

        
        
    # obtain the saved explanation
    expl_dict=read_pickle(path)
    expl=get_valid_expl(expl_dict,index)

    now = datetime.datetime.now()
    timestamp_str=now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

    node_imp_jaccard,expl_list=get_jaccard(expl_dict,expl,index=index)
    plot_jacard(node_imp_jaccard,expl_list,title="Disagreement_Node_Imp_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Disagreement_Node_Imp_{}_{}.png'.format(model_name,dataset_name))
    
    expl_dict_hubsc={}
    expl_dict_auth={}
    if type=='NC':
        expl_dict_hubsc,expl_dict_auth=obtain_hits_NC(data,expl_dict,expl)
    else:
        expl_dict_hubsc,expl_dict_auth=obtain_hits_GC(dataset,expl_dict,expl)

    
    agg_hub_score=calc_agg_score(expl_dict_hubsc,expl_list,index)
    agg_auth_score=calc_agg_score(expl_dict_auth,expl_list,index)

    
    plot_jacard(compute_heatmap(agg_hub_score,expl_list),labels=expl_list,title="Agg-HubScore-CosineDist_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Agg-HubScore-CosineDist_{}_{}.png'.format(model_name,dataset_name))
    plot_jacard(compute_heatmap(agg_auth_score,expl_list),labels=expl_list,title="Agg-AuthScore-CosineDist_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Agg-AuthScore-CosineDist_{}_{}.png'.format(model_name,dataset_name))

    plot_score(agg_hub_score,expl_list,index=index,xlabel=index,ylabel='Avg Hubscore of Importance Nodes',title='Agg_Hubscore_{}_{}'.format(model_name,dataset_name),path='Disagreement/disagreement/Disagreement_Agg_Hubscore_{}_{}.png'.format(model_name,dataset_name))
    plot_score(agg_auth_score,expl_list,index=index,xlabel=index,ylabel='Avg Authscore of Importance Nodes',title='Agg_Auth_{}_{}'.format(model_name,dataset_name),path='Disagreement/disagreement/Disagreement_Agg_Auth_{}_{}.png'.format(model_name,dataset_name))

    # for degree centrality
    expl_dict_deg_centr=obtain_nx_scores(dataset,expl_dict,expl,index,score_name="degree_centrality")
    agg_degcentr_score=calc_agg_score(expl_dict_deg_centr,expl_list,index)
    plot_jacard(compute_heatmap(agg_degcentr_score,expl_list),labels=expl_list,title="Agg-DegCentr-CosineDist_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Agg-DegCentr-CosineDist_{}_{}.png'.format(model_name,dataset_name))
    plot_score(agg_degcentr_score,expl_list,index=index,xlabel=index,ylabel='Avg DegCentr of Importance Nodes',title='Agg_DegCentr_{}_{}'.format(model_name,dataset_name),path='Disagreement/disagreement/Disagreement_Agg_DegCentr_{}_{}.png'.format(model_name,dataset_name))

    return expl_dict,expl_list


def get_disagreement_from_ranking(model_name,dataset_name,type,expl_path,ranking_path):
    '''
        Function to generate all the results that use ranking of node importance etc.
    '''
    # check file for explanation data
    if dataset_name not in expl_path.split('_'):
        print('{} provided is not for the {}'.format(expl_path,dataset_name))
        return None
    else :
        if '_' not in model_name:
            if model_name not in expl_path.split('_') :
                print("here")
                print('{} provided is not for the {}'.format(expl_path,model_name))
                return None
        else :
            if model_name!='GCN_3L':
                print("here")
                print('{} provided is not for the {}'.format(expl_path,model_name))
                return None
     
    # check file of ranking data
    if dataset_name not in ranking_path.split('_'):
        print('{} provided is not for the {}'.format(ranking_path,dataset_name))
        return None
    else :
        if '_' not in model_name:
            if model_name not in ranking_path.split('_') :
                print('{} provided is not for the {}'.format(ranking_path,model_name))
                return None
        else :
            if model_name!='GCN_3L':
                print('{} provided is not for the {}'.format(ranking_path,model_name))
                return None

    
    assert(model_name in ['GCN','GAT','GCN_3L','GNNGraphConv'])
    assert(dataset_name in ['Cora','CiteSeer','MUTAG','PROTEINS'])
    assert(type in ['GC','NC'])
    
    dataset=None
    data=None
    index=''

    if type=='NC':
        # TODO
        print("Not implemented yet")
    elif type=='GC':
        dataset = TUDataset(root='GNNModels/data/TUDataset', name=dataset_name)
        index='graph_indices'

        
        
    # obtain the saved explanation
    expl_dict=read_pickle(expl_path)

    expl_dict_rank=read_pickle(ranking_path)
    expl_list=get_valid_expl(expl_dict_rank,index)

    top3=get_top3(expl_dict,expl_dict_rank,expl_list,index,out='nodes')

    top3_pgm=get_pgm_top3(model_name,dataset_name)
    top3['pgm']={}
    for idx in top3[index]:
        top3['pgm'][idx]=top3_pgm['pgm_top3'][idx]

    # expl_list.append('pgm')
    if model_name=='GCN_3L':
        expl_list=['ig','pgm','cam','gcam']
    else:
        expl_list=['ig','pgm']

    expl_dict_hubsc={}  
    expl_dict_auth={}
    
    if type=='GC':
        expl_dict_hubsc,expl_dict_auth=obtain_hits_GC(dataset,top3,expl_list)

    node_imp_jaccard,expl_list=get_jaccard(top3,expl_list,index=index)
    plot_jacard(node_imp_jaccard,expl_list,title="Top3_Disagreement_Node_Imp_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Top3_Disagreement_Node_Imp_{}_{}.png'.format(model_name,dataset_name))
    
     
    agg_hub_score=calc_agg_score(expl_dict_hubsc,expl_list,index)
    agg_auth_score=calc_agg_score(expl_dict_auth,expl_list,index)

    
    plot_jacard(compute_heatmap(agg_hub_score,expl_list),labels=expl_list,title="Top3_Agg-HubScore-CosineDist_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Top3_Agg-HubScore-CosineDist_{}_{}.png'.format(model_name,dataset_name))
    plot_jacard(compute_heatmap(agg_auth_score,expl_list),labels=expl_list,title="Top3_Agg-AuthScore-CosineDist_{}_{}".format(model_name,dataset_name),path='Disagreement/disagreement/Top3_Agg-AuthScore-CosineDist_{}_{}.png'.format(model_name,dataset_name))

    plot_score(agg_hub_score,expl_list,index=index,xlabel=index,ylabel='Top3 Avg Hubscore of Importance Nodes',title='Top3 Agg_Hubscore_{}_{}'.format(model_name,dataset_name),path='Disagreement/disagreement/Top3_Disagreement_Agg_Hubscore_{}_{}.png'.format(model_name,dataset_name))
    plot_score(agg_auth_score,expl_list,index=index,xlabel=index,ylabel='Top3 Avg Authscore of Importance Nodes',title='Top3 Agg_Auth_{}_{}'.format(model_name,dataset_name),path='Disagreement/disagreement/Top3_Disagreement_Agg_Auth_{}_{}.png'.format(model_name,dataset_name))

    return expl_dict,expl_list
