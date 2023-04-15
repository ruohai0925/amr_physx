"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from .embedding_model import EmbeddingModel, EmbeddingTrainingHead
from trphysx.config.configuration_phys import PhysConfig
from torch.autograd import Variable

logger = logging.getLogger(__name__)
# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

class CylinderEmbedding(EmbeddingModel):
    """Embedding Koopman model for the 2D flow around a cylinder system

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    model_name = "embedding_cylinder"

    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__(config)

        X, Y = np.meshgrid(np.linspace(-2, 14, 128), np.linspace(-4, 4, 64))
        self.mask = torch.tensor(np.sqrt(X**2 + Y**2) < 1, dtype=torch.bool)

        # Encoder conv. net
        self.observableNet = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 16, 32, 64
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32, 16, 32
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64, 8, 16
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128, 4, 8
            nn.Conv2d(128, config.n_embd // 32, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            # 4, 4, 8
        )

        def count_conv2d_parameters(model):
            num_conv2d_parameters = 0
            for module in model.modules():
                if hasattr(module, 'weight'):
                    num_conv2d_parameters += module.weight.numel()
                    if module.bias is not None:
                        num_conv2d_parameters += module.bias.numel()
            return num_conv2d_parameters

        num_conv2d_parameters = count_conv2d_parameters(self.observableNet) # 102196
        print(" ")
        print('observableNet Number of parameters: {}'.format(num_conv2d_parameters))

        # # Print the parameters of the Conv2d layers
        # print(" ")
        # print("observableNet info ")
        # for name, module in self.observableNet.named_modules():
        #     if hasattr(module, 'weight'):
        #         print(name, module)
        #         print(module.weight.data.shape)
        #         print(module.bias.data.shape)
        # print(" ")

        # The LayerNorm layer has a single input channel of size 128 and an eps value specified by config.layer_norm_epsilon. 
        # The LayerNorm layer has two trainable parameters per input channel: 
        # a scaling parameter and a bias parameter. Therefore, the LayerNorm layer has a total of 2 * 128 = 256 trainable parameters.
        # The Dropout layer has no trainable parameters, since it simply randomly drops out input elements with probability config.embd_pdrop.

        self.observableNetFC = nn.Sequential(
            # nn.Linear(config.n_embd // 32 * 4 * 8, config.n_embd-1),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            # nn.BatchNorm1d(config.n_embd, eps=config.layer_norm_epsilon),
            nn.Dropout(config.embd_pdrop)
        )

        num_conv2d_parameters = count_conv2d_parameters(self.observableNetFC) # 256
        print(" ")
        print('observableNetFC Number of parameters: {}'.format(num_conv2d_parameters))

        # # Print the parameters of the Conv2d layers
        # print(" ")
        # print("observableNetFC info ")
        # for name, module in self.observableNetFC.named_modules():
        #     if hasattr(module, 'weight'):
        #         print(name, module)
        #         print(module.weight.data.shape)
        #         print(module.bias.data.shape)
        # print(" ")


        # The formula to calculate the number of parameters in an upsampling layer is 0

        # Decoder conv. net
        self.recoveryNet = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(config.n_embd // 32, 128, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 128, 8, 16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 64, 16, 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 32, 32, 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            # 16, 64, 128
            nn.Conv2d(16, 3, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate'),
            # 3, 64, 128, here 3 means u, v, and p
        )

        num_conv2d_parameters = count_conv2d_parameters(self.recoveryNet) # 102051
        print(" ")
        print('recoveryNet Number of parameters: {}'.format(num_conv2d_parameters))

        # # Print the parameters of the Conv2d layers
        # print(" ")
        # print("recoveryNet info ")
        # for name, module in self.recoveryNet.named_modules():
        #     if hasattr(module, 'weight'):
        #         print(name, module)
        #         print(module.weight.data.shape)
        #         print(module.bias.data.shape)
        # print(" ")

        # Learned Koopman operator parameters
        self.obsdim = config.n_embd
        # We parameterize the Koopman operator as a function of the viscosity
        self.kMatrixDiagNet = nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.obsdim))


        num_conv2d_parameters = count_conv2d_parameters(self.kMatrixDiagNet) # 6628
        print(" ")
        print('kMatrixDiagNet Number of parameters: {}'.format(num_conv2d_parameters))

        # # Print the parameters of the Conv2d layers
        # print(" ")
        # print("kMatrixDiagNet info ")
        # for name, module in self.kMatrixDiagNet.named_modules():
        #     if hasattr(module, 'weight'):
        #         print(name, module)
        #         print(module.weight.data.shape)
        #         print(module.bias.data.shape)
        # print(" ")


        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 5):
            yidx.append(np.arange(i, self.obsdim))
            xidx.append(np.arange(0, self.obsdim-i))
        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))

        # print("yidx.shape ", self.yidx.shape) # torch.Size([502])
        # print("xidx.shape ", self.xidx.shape) # torch.Size([502])

        # The matrix here is a small NN since we need to make it dependent on the viscosity
        self.kMatrixUT = nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.size(0)))
        self.kMatrixLT = nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.Linear(50, self.xidx.size(0)))

        num_conv2d_parameters = count_conv2d_parameters(self.kMatrixUT) # 25702
        print(" ")
        print('kMatrixUT Number of parameters: {}'.format(num_conv2d_parameters))

        # # Print the parameters of the Conv2d layers
        # print(" ")
        # print("kMatrixUT info ")
        # for name, module in self.kMatrixUT.named_modules():
        #     if hasattr(module, 'weight'):
        #         print(name, module)
        #         print(module.weight.data.shape)
        #         print(module.bias.data.shape)
        # print(" ")

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor([0., 0., 0., 0.]))
        self.register_buffer('std', torch.tensor([1., 1., 1., 1.]))
        
        logger.info('Number of embedding parameters: {}'.format( super().num_parameters )) # 262535 = 25702 * 2 + 6628 + 102051 + 256 + 102196

    def forward(self, x: Tensor, visc: Tensor) -> TensorTuple:
        """Forward pass

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (TensorTuple): Tuple containing:

                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3, H, W] Recovered feature tensor
        """

        print("forward of CylinderEmbedding ")
        print(" ")

        # Concat viscosities as a feature map
        # print("x.shape 1 ", x.shape) # torch.Size([6, 3, 64, 128])
        # print("visc.shape ", visc.shape) # torch.Size([6, 1])
        
        x = torch.cat([x, visc.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:,:1])], dim=1)

        # print("x.shape 1 ", x.shape) # torch.Size([6, 4, 64, 128])

        x = self._normalize(x)
        
        # print("x.shape 2 ", x.shape) # torch.Size([6, 4, 64, 128])
        
        g0 = self.observableNet(x)

        # print("g0.shape ", g0.shape) # torch.Size([6, 4, 4, 8]

        g = self.observableNetFC(g0.view(g0.size(0),-1))
        
        # print("g.shape ", g.shape) # torch.Size([6, 128])

        # Decode
        out = self.recoveryNet(g.view(g0.shape)) # after using view, torch.Size([6, 4, 4, 8]

        # print("out.shape ", out.shape) # torch.Size([6, 3, 64, 128])

        xhat = self._unnormalize(out)

        # print("xhat.shape ", xhat.shape) # torch.Size([6, 3, 64, 128])

        # Apply cylinder mask
        # maskold = self.mask.repeat(xhat.size(0), xhat.size(1), 1, 1) is True # Cannot write in this way
        mask0 = self.mask.repeat(xhat.size(0), xhat.size(1), 1, 1) == True

        # mask1 = self.mask.repeat(xhat.size(0), xhat.size(1), 1, 1)
        # print("maskold ", maskold) # False

        # print("mask0 ", mask0.shape)

        # print("mask1 ", mask1.shape)
        # if (mask0 == mask1).all():
        #     print("All elements of mask0 and mask1 are the same") # mask1 == mask0
        # else:
        #     print("mask0 and mask1 differ in at least one element")

        xhat[mask0] = 0

        # exit()

        return g, xhat

    def embed(self, x: Tensor, visc: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables

        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        # Concat viscosities as a feature map
        x = torch.cat([x, visc.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:,:1])], dim=1)
        x = self._normalize(x)

        g = self.observableNet(x)
        g = self.observableNetFC(g.view(x.size(0), -1))
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables

        Returns:
            (Tensor): [B, 3, H, W] Physical feature tensor
        """
        x = self.recoveryNet(g.view(-1, self.obsdim//32, 4, 8))
        x = self._unnormalize(x)
        # Apply cylinder mask
        mask0 = self.mask.repeat(x.size(0), x.size(1), 1, 1) == True
        x[mask0] = 0
        return x

    def koopmanOperation(self, g: Tensor, visc: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables

        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
            visc (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        print("koopmanOperation ")
        print(" ")        
        
        # Koopman operator
        kMatrix = Variable(torch.zeros(g.size(0), self.obsdim, self.obsdim)).to(self.devices[0])
        
        print("kMatrix ", kMatrix.shape) # torch.Size([6, 128, 128])

        print(" ") 

        # Populate the off diagonal terms
        kMatrix[:,self.xidx, self.yidx] = self.kMatrixUT(100*visc) # Why to multiply 100 for visc?
        kMatrix[:,self.yidx, self.xidx] = self.kMatrixLT(100*visc)

        # Populate the diagonal
        ind = np.diag_indices(kMatrix.shape[1])

        print("kMatrix.shape[1] ", kMatrix.shape[1]) # 128
        print(" ")

        print("ind ", ind)
        print(" ") 

        self.kMatrixDiag = self.kMatrixDiagNet(100*visc)
        kMatrix[:, ind[0], ind[1]] = self.kMatrixDiag


        # g is a PyTorch tensor with shape (6, 128), representing a batch of 6 input vectors, 
        # each with 128 features. kMatrix is also a PyTorch tensor with shape (6, 128, 128), 
        # representing a batch of 6 square matrices, each with dimensions 128 x 128.

        # To apply the matrix multiplication kMatrix * g, 
        # we need to ensure that the dimensions of the input vectors and matrices are compatible. 
        # Specifically, the last dimension of kMatrix (128) must match the second dimension of g (128) to perform the matrix multiplication.

        # However, g has shape (6, 128) while kMatrix has shape (6, 128, 128). 
        # To perform the matrix multiplication, we need to reshape g to have shape (6, 128, 1), 
        # which can be done using g.unsqueeze(-1).

        # The unsqueeze() method adds an extra dimension of size 1 at the specified position, 
        # which in this case is the last dimension (-1). This effectively reshapes g to have shape (6, 128, 1), 
        # which allows us to perform the batch matrix multiplication using torch.bmm().

        # The resulting tensor gnext has shape (6, 128, 1), 
        # and represents the output of the batch matrix multiplication. 
        # Note that the extra dimension of size 1 can be removed using gnext.squeeze(-1), if desired.

        # Apply Koopman operation
        gnext = torch.bmm(kMatrix, g.unsqueeze(-1))

        print("gnext ", gnext.shape) # torch.Size([6, 128, 1])
        print(" ")

        self.kMatrix = kMatrix
        return gnext.squeeze(-1) # Squeeze empty dim from bmm

    @property
    def koopmanOperator(self, requires_grad: bool =True) -> Tensor:
        """Current Koopman operator

        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True

        Returns:
            Tensor: Full Koopman operator tensor
        """
        if not requires_grad:
            return self.kMatrix.detach()
        else:
            return self.kMatrix

    def _normalize(self, x: Tensor) -> Tensor:
        # See the chatgpt explanation!
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x: Tensor) -> Tensor: # why 3? Because we only need to output u, v, and p.
        return self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*x + self.mu[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    @property
    def koopmanDiag(self):
        return self.kMatrixDiag

class CylinderEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the the 2D flow around a cylinder model

    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    def __init__(self, config: PhysConfig) -> None:
        """Constructor method
        """
        super().__init__()
        self.embedding_model = CylinderEmbedding(config)

    def forward(self, states: Tensor, viscosity: Tensor) -> FloatTuple:
        """Trains model for a single epoch

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            viscosity (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            FloatTuple: Tuple containing:
            
                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        assert states.size(0) == viscosity.size(0), 'State variable and viscosity tensor should have the same batch dimensions.'

        print("forward of CylinderEmbeddingTrainer ")
        print("states ", states.shape) # torch.Size([6, 4, 3, 64, 128])
        print("viscosity ", viscosity.shape) # torch.Size([6, 1])
        print(" ")        

        self.embedding_model.train() # train stage
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        # In the case of states[:,0], the colon : before the comma indicates that we want to select all elements 
        # along the first dimension of the tensor, which represents the batch size. 
        # The 0 after the comma indicates that we want to select the first element along the second dimension of the tensor, 
        # which represents the time steps.
        xin0 = states[:,0].to(device) # Time-step
        print("xin0 ", xin0.shape) # torch.Size([6, 3, 64, 128])

        print(" ")
        viscosity = viscosity.to(device)

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0, viscosity)
        print("g0 ", g0.shape) # torch.Size([6, 128])
        print("xRec0 ", xRec0.shape) # torch.Size([6, 3, 64, 128])
        print(" ")  

        loss = (1e1)*mseLoss(xin0, xRec0) # Amplify the loss? How to get this parameters?
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0

        print("states.shape[1] ", states.shape[1]) # 4
        print(" ")  

        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            
            xin0 = states[:,t0,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0, viscosity)
            
            # Apply Koopman transform
            print("g1_old.shape ", g1_old.shape) # torch.Size([6, 128])
            print(" ")             
            g1Pred = self.embedding_model.koopmanOperation(g1_old, viscosity)

            print("g1Pred.shape ", g1Pred.shape) # torch.Size([6, 128])
            print(" ")              
            xgRec1 = self.embedding_model.recover(g1Pred)

            print("xgRec1.shape ", xgRec1.shape) # torch.Size([6, 3, 64, 128])
            print(" ")

            exit()

            # Loss function
            # Unlike loss_reconstruct, loss requires gradients to be 
            # computed during backpropagation because it is used to update the parameters of the neural network. 
            # Therefore, loss does not require detach() to be used.
            loss = loss + (1e1)*mseLoss(xgRec1, xin0) + (1e1)*mseLoss(xRec1, xin0) \
                + (1e-2)*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            # In the given example, mseLoss(xRec1, xin0) computes the mean squared error (MSE) between xRec1 and xin0. 
            # The resulting tensor requires gradients to be computed during backpropagation so that the parameters of the neural network can be updated. 
            # However, loss_reconstruct is used to accumulate the MSE loss over multiple iterations of training and does not require gradients. 
            # Therefore, detach() is used to create a new tensor that shares the same data as mseLoss(xRec1, xin0) but does not require gradients.
            # By using detach(), the computation of loss_reconstruct is decoupled from the computation of the gradients of the neural network, 
            # which can lead to faster and more memory-efficient training.

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(self, states: Tensor, viscosity: Tensor) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.

        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            viscosity (Tensor): [B] Viscosities of the fluid in the mini-batch

        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval() # eval stage
        device = self.embedding_model.devices[0]

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        yTarget = states[:,1:].to(device)
        xInput = states[:,:-1].to(device)
        yPred = torch.zeros(yTarget.size()).to(device)
        viscosity = viscosity.to(device)

        # Test accuracy of one time-step
        for i in range(xInput.size(1)):
            xInput0 = xInput[:,i].to(device)
            g0 = self.embedding_model.embed(xInput0, viscosity)
            yPred0 = self.embedding_model.recover(g0)
            yPred[:,i] = yPred0.squeeze().detach()

        test_loss = mseLoss(yTarget, yPred)

        return test_loss, yPred, yTarget