from Disagreement_Metric import *

# python script to generate disagreement plots for the various saved explanations 

path='Saved_Explanations/Explanations_GCN_Cora_2022-11-26T15:02:04-75.pkl'
get_disagreement(model_name='GCN',dataset_name='Cora',type='NC',path=path)

path='Saved_Explanations/Explanations_GCN_CiteSeer_2022-11-27T15:28:38-29.pkl'
get_disagreement(model_name='GCN',dataset_name='CiteSeer',type='NC',path=path)

path='Saved_Explanations/Explanations_GAT_CiteSeer_2022-11-26T21:22:37-74.pkl'
get_disagreement(model_name='GAT',dataset_name='CiteSeer',type='NC',path=path)

path='Saved_Explanations/Explanations_GAT_Cora_2022-11-26T20:57:56-89.pkl'
get_disagreement(model_name='GAT',dataset_name='Cora',type='NC',path=path)

