import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
from torch_geometric.datasets import Planetoid
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


def get_jaccard(expl_dict,expl):
    assert(len(expl)!=0)
    n_methods=len(expl)
    jacard = np.zeros((n_methods, n_methods))
    node_indices=expl_dict['node_indices']
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


def obtain_hits(data,expl_dict,expl=[]):
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

def get_valid_expl(expl_dict):
    # it returns the list of valid explanations available 
    # it checks if for any explanation there is a none entry

    available_expl=list(expl_dict.keys())[1:] # first one is always 'node_indices'
    valid_expl=[]
    node_indices=expl_dict['node_indices']
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

def calc_agg_score(expl_dict,expl_list):
    node_indices=expl_dict['node_indices']
    agg_score={}
    agg_score['node_indices']=node_indices
    for node_idx in node_indices:
        for exp_type in expl_list:
            if exp_type not in agg_score:
                agg_score[exp_type]=[]
            if(len(expl_dict[exp_type][node_idx])!=0):
                agg_score[exp_type].append(sum(expl_dict[exp_type][node_idx])/len(expl_dict[exp_type][node_idx]))
            else:
                agg_score[exp_type].append(0)
    
    return agg_score

def plot_score(agg_score,expl_list,xlabel,ylabel,title,path='',type='normal'):
    np.random.seed(12)
    plt.clf()
    x=agg_score['node_indices']
    random_nodes_index=np.random.choice(list(range(0,len(x))), size=10)
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
    fig.set_size_inches(18.5, 10.5)
    if path!='':
        plt.savefig(path)

def compute_cosine(list1,list2):
    return scipy.spatial.distance.cosine(list1,list2)

def compute_heatmap(expl_dict,expl_list):
    n=len(expl_list)
    matrix=np.zeros((n,n))
    for i in range(0,n):
        for j in range(i,n):
            matrix[i][j]=compute_cosine(expl_dict[expl_list[i]],expl_dict[expl_list[j]])
            matrix[j][i]=matrix[i][j]

    return matrix

def get_disagreement(model_name,dataset_name,type,path):
    if model_name not in path.split('_') or dataset_name not in path.split('_'):
        print('{} provided is not for the {} and {}'.format(path,model_name,dataset_name))
        return None
    
    assert(model_name in ['GCN','GAT','GCN_3L','GNNGraphConv'])
    assert(dataset_name in ['Cora','CiteSeer','MUTAG','PROTEIN'])
    assert(type in ['GC','NC'])
    
    dataset=None
    data=None
    if type=='NC':
        dataset = Planetoid(root='/tmp/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]  # Get the first graph object.
    elif type=='GC':
        # TODO
        print("No implementation yet")
        return
        
    # obtain the saved explanation
    expl_dict=read_pickle(path)
    expl=get_valid_expl(expl_dict)

    now = datetime.datetime.now()
    timestamp_str=now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

    node_imp_jaccard,expl_list=get_jaccard(expl_dict,expl)
    plot_jacard(node_imp_jaccard,expl_list,title="Disagreement_Node_Imp_{}_{}".format(model_name,dataset_name),path='disagreement/Disagreement_Node_Imp_{}_{}_{}.png'.format(model_name,dataset_name,timestamp_str))

    expl_dict_hubsc,expl_dict_auth=obtain_hits(data,expl_dict,expl)

    # these metrics dont really make sense
    # they do the thing same as the above jaccard

    # hubsc_jaccard,expl_list=get_jaccard(expl_dict_hubsc,expl)
    # plot_jacard(hubsc_jaccard,expl_list,title="Disagreement_Hubsc_{}_{}".format(model_name,dataset_name),path='disagreement/Disagreement_Hubsc_{}_{}_{}.png'.format(model_name,dataset_name,timestamp_str))

    # auth_jaccard,expl_list=get_jaccard(expl_dict_auth,expl)
    # plot_jacard(auth_jaccard,expl_list,title="Disagreement_Auth_{}_{}".format(model_name,dataset_name),path='disagreement/Disagreement_Auth_{}_{}_{}.png'.format(model_name,dataset_name,timestamp_str))

    agg_hub_score=calc_agg_score(expl_dict_hubsc,expl_list)
    agg_auth_score=calc_agg_score(expl_dict_auth,expl_list)

    
    plot_jacard(compute_heatmap(agg_hub_score,expl_list),labels=expl_list,title="Agg-HubScore-CosineDist_{}_{}",path='disagreement/Agg-HubScore-CosineDist_{}_{}.png'.format(model_name,dataset_name))
    plot_jacard(compute_heatmap(agg_auth_score,expl_list),labels=expl_list,title="Agg-AuthScore-CosineDist_{}_{}",path='disagreement/Agg-AuthScore-CosineDist_{}_{}.png'.format(model_name,dataset_name))

    plot_score(agg_hub_score,expl_list,xlabel='Node Index',ylabel='Avg Hubscore of Importance Nodes',title='Agg_Hubscore_{}_{}'.format(model_name,dataset_name),path='disagreement/Disagreement_Agg_Hubscore_{}_{}.png'.format(model_name,dataset_name))
    plot_score(agg_auth_score,expl_list,xlabel='Node Index',ylabel='Avg Authscore of Importance Nodes',title='Agg_Auth_{}_{}'.format(model_name,dataset_name),path='disagreement/Disagreement_Agg_Auth_{}_{}.png'.format(model_name,dataset_name))

    return expl_dict,expl_list