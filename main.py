import pandas as pd
import python_graphs.visualization as vis

Networks = ['email-Eu-core-temporal-Dept3','opsahl-ucsocial','radoslaw_email','soc-sign-bitcoinalpha', 'soc-sign-bitcoinotc']
networks_files_names = [ f'datasets/{i}/out.{i}' for i in Networks]
number_of_datasets = 5
datasets_info = {'Network': ['email-Eu-core-temporal', 'opsahl-ucsocial','radoslaw_email','soc-sign-bitcoinalpha', 'soc-sign-bitcoinotc'],
'Label': ['EU','UC','Rado','bitA','bitOT'],
'Category': ['Social',"Information","Social","Social","Social"],
'Edge type': ['Multi','Multi','Multi','Simple','Simple'],
'Path': networks_files_names}
datasets_info = pd.DataFrame(datasets_info)
datasets_info = datasets_info.iloc[4:5]
print(datasets_info)
latex_feature_network_table,latex_auc_table = vis.graph_features_tables(datasets_info)
print(latex_feature_network_table)
print(latex_auc_table)