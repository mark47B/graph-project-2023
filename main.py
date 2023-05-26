import pandas as pd
import python_graphs.visualization as vis

Networks = ['email-Eu-core-temporal-Dept3', 'munmun_digg_reply', 'opsahi-ucsocial','radoslaw_email','soc-sign-bitcoinalpha', 'sx-mathoverflow']
networks_files_names = [ f'datasets/{i}/out.{i}' for i in Networks]
number_of_datasets = 6
datasets_info = {'Network': ['email-Eu-core-temporal', 'munmun_digg_reply', 'opsahi-ucsocial','radoslaw_email','soc-sign-bitcoinalpha', 'sx-mathoverflow'],
'Label': ['EU','D-rep','UC','Rado','bitA','SX-MO'],
'Category': ['Social',"Social","Information","Social","Social","Social"],
'Edge type': ['Multi','Simple','Multi','Multi','Simple','Multi'],
'Path': networks_files_names}
datasets_info = pd.DataFrame(datasets_info)

datasets_info = datasets_info.iloc[0:1]
print(datasets_info.head())

latex_feature_network_table,latex_auc_table = vis.graph_features_tables(datasets_info)
print(latex_feature_network_table)
print(latex_auc_table)