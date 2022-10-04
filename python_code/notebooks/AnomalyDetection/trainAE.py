"""
TRAIN 

. Example: Run the following command from the terminal.
    
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model_simpleAE import AE_ES_1D

##
def train():
    """ Training
    """
    ##
    # ARGUMENTS
    opt = Options().parse()
    opt.isTrain = True

    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = AE_ES_1D(opt, dataloader)
    ##
    # TRAIN MODEL
    model.train()
    

if __name__ == '__main__':
    train()
