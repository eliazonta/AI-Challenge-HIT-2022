"""
EVALUATE 

. Example: Run the following command from the terminal.
    
"""


##
# LIBRARIES
from __future__ import print_function

import os
from options import Options
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from data_visualization_torch import *


from lib.data import load_data
from lib.model_simpleAE import AE_ES_1D

##
def evaluate():
    """ Evaluation
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    opt.isTrain = False

    # weight_dir = os.path.join(opt.outf, opt.name, 'train', 'weights','CAE.pt')
    # training_configuration = torch.load(weight_dir)['training_conf']
    # opt.training_reg = training_configuration['training_reg']
    # opt.validation_reg = training_configuration['validation_reg']

    ##
    # LOAD DATA
    dataloader = load_data(opt)


    ##
    # LOAD MODEL
    # model = GANCAE_ES_1D(opt, dataloader)
    model = AE_ES_1D(opt, dataloader)
    model.load_weights()
    # print(model.cae.dec1.weight)
    # print(model.cae.dec1.bias)

    df = pd.read_csv(f'./data/{opt.dataset}.csv')
    map_name_sectors_BH(df,256,1)
    # print(df)

    import matplotlib.pyplot as plt
    import numpy as np

    ### SEE INSIDE WEIGHTS
    # fig, ax = plt.subplots()    
    # with torch.no_grad():
    #     plt.xticks(rotation=90)
    #     ax.set_xticks(range(0,64))
    #     ax.set_xticklabels(df.columns[1:].values)
    #     plt.scatter(range(0,64),model.cae.dec1.weight[:,0],label='1 1')
    #     plt.scatter(range(0,64),model.cae.dec1.weight[:,1],label='0 3')
    #     plt.scatter(range(0,64),model.cae.dec1.weight[:,2],label='0 3')
    #     plt.scatter(range(0,64),model.cae.dec1.bias)
    # plt.grid()
    # plt.legend()
    # plt.show()

    # with torch.no_grad():
    #     selected_features = torch.argmax(torch.softmax(torch.exp(model.cae.concrete_selector.logits),dim=-1), dim=-1).numpy()
    #     print(selected_features+1)

    
    X_test = torch.stack([v["features"] for v in next(iter(dataloader['test']))])

    with torch.no_grad():
        X_hat_test = model.ae(X_test)
    
        reconstructed_df = pd.DataFrame( X_hat_test.numpy(), columns = df.columns[1:] )
        reconstructed_df = pd.concat([df['parameter'],reconstructed_df], axis = 1)
    
    U=1
    plot_es_parabola(df,256,U,4,model='BH',power=1,ylim =[0,14],xlim=[-4,4])#,save_figure=f'./plots/GAN/BH_reconstruction/BH256_U_{U}_original.png',save_fig_args={'dpi':300})
    plot_es_parabola(reconstructed_df,256,U,4,model='BH',power=1,ylim =[0,14],xlim=[-4,4])#,save_figure=f'./plots/GAN/BH_reconstruction/BH256_U_{U}_recons.png',save_fig_args={'dpi':300})
    plt.show()
    # for i in range(10):
    #     plt.figure(3)
    #     plt.ylim([0,4])
    #     plt.plot(df['parameter'],-np.log10(X_test[:,i]))
    #     plt.figure(4)
    #     plt.ylim([0,4])
    #     plt.plot(df['parameter'],-np.log10(X_hat_test[:,i]))
    # plt.show()
    # # plt.plot(parameter,-np.log(X_test[:,selected_features[0]]))
    # # plt.plot(parameter,-np.log(X_test[:,selected_features[1]]))
    # # plt.plot(parameter,-np.log(X_test[:,selected_features[2]]))
    # # plt.show()
    # #     for i in range(opt.lsize):
    # #         plt.scatter(range(0,64),torch.softmax(torch.exp(model.cae.concrete_selector.logits[i]),dim=0),label=f'{i}')
    # # plt.legend()
    # # plt.show()
    
    # ##
    # # TEST MODEL
    # return model.test()

if __name__ == '__main__':
    evaluate()
