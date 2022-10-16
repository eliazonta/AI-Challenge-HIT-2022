"""GAN for phase transition detection
"""
import os

from collections import OrderedDict
from importlib import reload  # Python 3.4+
import networks
networks = reload(networks)
from networks import AE
from tqdm import tqdm
import loss
loss = reload(loss)
from loss import l2_loss
import torch.optim as optim
import torch.utils.data
# import wandb

class BaseModel():
    """ Base Model 
    """
    def __init__(self, opt, dataloader):
        
        # Initalize variables.
        self.opt = opt
        self.dataloader = dataloader
        # self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        # self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.losses = {}
        # # Initialize wandb logger
        # if (self.opt.isTrain):
        #     wandb.init(project=opt.wandb_proj, entity=opt.wandb_account, config=opt)
        #     wandb.run.name = opt.name
        #     wandb.run.save()

    def set_input(self, input:torch.Tensor):
        """ Set input

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input = input
    
    ##
    def get_losses(self):
        """ Reports the losses of the model.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        return self.losses

    # ##

    #Save weights
    def save_weights(self, epoch):
        #Save netG and netD weights for the current epoch.

         #Args:epoch ([int]): Current epoch number.
         

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

    #     ## together with the weigths and the epoch, also the training and validation windows parameters are stored
        torch.save({'epoch': self.opt.epoch + 1, 'state_dict': self.ae.state_dict()},
                    '%s/AE.pt' % (weight_dir))

    # load weights
    def load_weights(self):
    #     """Load netG and netD weights.
    #     """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights','AE.pt')
        pretrained_dict = torch.load(weight_dir)['state_dict']
        training_configuration = torch.load(weight_dir)['training_conf']
        self.opt.training_reg = training_configuration['training_reg']
        self.opt.validation_reg = training_configuration['validation_reg']
        self.opt.iter = torch.load(weight_dir)['epoch']

        try:
            self.ae.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("netG weights not found")
        print('   Loaded weights.')

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.ae.train()
        epoch_iter = 0

        for data in tqdm(self.dataloader['train'], leave=True, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            
            batch_input_features = torch.stack([v["features"] for v in data])
            self.set_input(batch_input_features)
            self.optimize_params()

        # Validate each epoch
        self.ae.eval()
        self.validate()

        errors = self.get_losses()
        # self.losses.update({'lr_G':self.optimizer_G.state_dict()['param_groups'][0]['lr']})
        # wandb.log(errors)
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))#, end='\r')
        print(f'>> Training loss: {errors["loss_tr"]}')
        print(f'>> Validation loss: {errors["loss_val"]}')
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    ### TODO: use display to be verbose on the screen with ongoing training performances
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0
        
        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        # Start the logger
        # wandb.run
        metrics = []
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            metrics.append(self.losses)
        # self.save_weights(self.epoch)
               
        print(f">> Training model {self.name}.[Done]")
        return metrics

    ##
    def validate(self):
        """ Evaluate GAN model on the whole test dataset.
        """
        
        X_test = torch.stack([v["features"] for v in next(iter(self.dataloader['test']))])
        X_train = torch.stack([v["features"] for v in next(iter(self.dataloader['train']))])

        with torch.no_grad():
            X_hat_test = self.ae(X_test)
            X_hat_train = self.ae(X_train)
            ### compute the losses and print them
            test_loss = l2_loss(X_test,X_hat_test)
        self.losses.update({'loss_val':test_loss.item()})

##
class AE_1D(BaseModel):
    """simple AE class 
    """

    @property
    def name(self): return 'AE_1D'

    def __init__(self, opt, dataloader):
        super(AE_1D, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.ae = AE(self.opt).to(self.device)
        
        ##
        # if self.opt.resume != '':
        #     print("\nLoading pre-trained networks.")
        #     self.opt.iter = torch.load(os.path.join(self.opt.resume, 'AE.pt'))['epoch']
        #     self.ae.load_state_dict(torch.load(os.path.join(self.opt.resume, 'AE.pt'))['state_dict'])
        #     print("\tDone.\n")


        self.l_rec = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.isize), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            ### TODO change the frequences according to opt
            # wandb.watch(self.ae, log_freq=50)
            # models in training mode
            self.ae.train()
            # definition of optimizers
            self.optimizer = optim.Adam(self.ae.parameters(), lr=self.opt.lr)

    ##
    def forward(self):
        """ Forward propagate through netG
        """
        self.reconstruction = self.ae(self.input)

    ### TODO: check if in this phase the adv loss can be used as l2 for the labels:
    ### in this case, the aim of the discriminator is to minimize l2(netD(input),netD(reconstruction))
    ##
    def backward(self):
        """ Backpropagate through the net
        """
        self.err_rec = self.l_rec(self.input, self.reconstruction)
        self.err_rec.backward(retain_graph=True)
         
    ##
    def update_losses(self):
        self.losses = OrderedDict([
            ('loss_tr',self.err_rec.item()),
            ])
    
    ##
    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

        # Update the losses every step
        self.update_losses()

