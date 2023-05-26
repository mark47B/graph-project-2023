import pandas as pd
import python_graphs.visualization as vis

Networks = ['email-Eu-core-temporal-Dept3','opsahl-ucsocial','radoslaw_email',
            'soc-sign-bitcoinalpha','dnc-corecipient','munmun_digg_reply',
            'sx-mathoverflow']
networks_files_names = [ f'datasets/{i}/out.{i}' for i in Networks]
number_of_datasets = 5
datasets_info = {'Network': ['email-Eu-core-temporal', 'opsahl-ucsocial','radoslaw_email','soc-sign-bitcoinalpha',
                             'dnc-corecipient', 'munmun_digg_reply','sx-mathoverflow'],
'Label': ['EU','UC','Rado','bitA','Dem ','D-rep','SX-MO'],
'Category': ['Social',"Information","Social","Social","Social","Social"],
'Edge type': ['Multi','Multi','Multi','Simple','Multi','Simple','Multi'],
'Path': networks_files_names}
datasets_info = pd.DataFrame(datasets_info)

latex_feature_network_table,latex_auc_table = vis.graph_features_tables(datasets_info)
print(latex_feature_network_table)
print(latex_auc_table)