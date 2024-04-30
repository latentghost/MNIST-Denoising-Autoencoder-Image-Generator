import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import os
import random
from EncDec import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class AlteredMNIST(torch.utils.data.Dataset):
    """
    A custom dataset class for AlteredMNIST.

    Args:
        root (str): The root directory of the dataset. Default is "./Data/".
        val (bool): Whether the dataset is for validation. Default is False.

    Attributes:
        root (str): The root directory of the dataset.
        aug (str): The directory path for augmented images.
        clean (str): The directory path for clean images.
        aug_paths (list): A list of paths for augmented images.
        aug_len (int): The number of augmented images.
        clean_len (int): The number of clean images.
        w_r (torch.Tensor): The weight for the red channel.
        w_b (torch.Tensor): The weight for the blue channel.
        w_g (torch.Tensor): The weight for the green channel.
        targets (list): A list of lists containing the paths of clean images for each label.

    Methods:
        __init__(self, root="./Data/", val=False): Initializes the AlteredMNIST dataset.
        __getitem__(self, index): Retrieves an item from the dataset.
        __len__(self): Returns the length of the dataset.
        to_grayscale(self, img): Converts an image to grayscale.

    """

    def __init__(self, root="./Data/", val=False):
        """
        Initializes the AlteredMNIST dataset.

        Args:
            root (str): The root directory of the dataset. Default is "./Data/".
            val (bool): Whether the dataset is for validation. Default is False.
        """
        super(AlteredMNIST, self).__init__()
        self.root = root
        self.aug = os.path.join(root, "aug/")
        self.clean = os.path.join(root, "clean/")
        self.aug_paths = (os.listdir(self.aug))
        self.aug_len = len(self.aug_paths)
        self.clean_len = len(os.listdir(self.clean))

        self.w_r = nn.Parameter(torch.tensor(0.2126), requires_grad=False)
        self.w_b = nn.Parameter(torch.tensor(0.7152), requires_grad=False)
        self.w_g = nn.Parameter(torch.tensor(0.0722), requires_grad=False)

        self.targets = [[] for _ in range(10)]
        for file in os.listdir(self.clean):
            if file.endswith(".png"):
                label = int(file.split("_")[2].split(".")[0])
                image_path = os.path.join(self.clean, file)
                self.targets[label].append(image_path)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            img (torch.Tensor): The augmented image.
            target (torch.Tensor): The corresponding clean image.
            label (int): The label of the image.
        """
        img_path = self.aug_paths[index]
        img = (torchvision.io.read_image(os.path.join(self.aug, img_path))).float()
        idx = int(img_path.split("_")[1])
        label = int(img_path.split("_")[2].split(".")[0])

        if img.shape[0] == 3:
            img = self.to_grayscale(img).unsqueeze(0)

        if idx < self.clean_len:
            target = (torchvision.io.read_image(os.path.join(self.clean, f"clean_{idx}_{label}.png"))).float()
        else:
            random_els = random.sample(self.targets[label], 10)
            best_target = random_els[0]
            best_score = 0.5 * (ssim(img, torchvision.io.read_image(best_target).float())) + 0.5 * (psnr(img, torchvision.io.read_image(best_target).float()))

            for candidate in random_els[1:]:
                psnr_score = 0.5 * (ssim(img, torchvision.io.read_image(candidate).float())) + 0.5 * (psnr(img, torchvision.io.read_image(candidate).float()))
                if psnr_score > best_score:
                    best_score = psnr_score
                    best_target = candidate
            target = torchvision.io.read_image(best_target).float()

        img = img.to(torch.float32).div_(255.)
        target = target.to(torch.float32).div_(255.)
        return img, target, label

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.aug_len

    def to_grayscale(self, img):
        """
        Converts an image to grayscale.

        Args:
            img (torch.Tensor): The image to convert.

        Returns:
            torch.Tensor: The grayscale image.
        """
        return self.w_r * img[0] + self.w_g * img[1] + self.w_b * img[2]
    

def one_hot(x,num_classes):
    out = torch.zeros(x.size(0),num_classes)
    for i,l in enumerate(x):
        out[i][l] = 1
    return out


class EncoderBlock(nn.Module):
    """
    EncoderBlock class represents a single block in an encoder network.
    It consists of convolutional layers, batch normalization, and skip connections.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding added to the input.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the third convolution.
        downsample (nn.Sequential): Downsample layer consisting of a convolutional layer and batch normalization.
        relu (nn.SELU): Activation function.

    Methods:
        forward(x): Performs forward pass through the encoder block.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.SELU()
        
    def forward(self, x):
        """
        Performs forward pass through the encoder block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the encoder block.

        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """
    Encoder class for a convolutional neural network.

    Args:
        in_channels (int): Number of input channels. Default is 1.
        out_channels (int): Number of output channels. Default is 16.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer with kernel size 4, stride 2, padding 1, and bias.
        block1 (EncoderBlock): First encoder block.
        block2 (EncoderBlock): Second encoder block.
        block3 (EncoderBlock): Third encoder block.
        block4 (EncoderBlock): Fourth encoder block.
        fc_mu (nn.Linear): Linear layer for mean calculation.
        fc_logvar (nn.Linear): Linear layer for log variance calculation.
        elu (nn.ELU): ELU activation function.
        selu (nn.SELU): SELU activation function.
        fc1 (nn.Linear): Linear layer for the first fully connected layer.
        fc2 (nn.Linear): Linear layer for the second fully connected layer.
        fc3 (nn.Linear): Linear layer for the third fully connected layer.
        mu (None): Mean value.
        logvar (None): Log variance value.
        z_mu (None): Mean value for CVAE.
        z_logvar (None): Log variance value for CVAE.

    Methods:
        forward(x, label=None, VAE=False, CVAE=False): Forward pass of the encoder.
        reparametrize(VAE=False, CVAE=False): Reparametrization trick for VAE and CVAE.

    """

    def __init__(self, in_channels=1, out_channels=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.block1 = EncoderBlock(2, 4, 3, 2, 2)
        self.block2 = EncoderBlock(4, 8, 3, 2, 1)
        self.block3 = EncoderBlock(8, out_channels, 2, 1, 0)
        self.block4 = EncoderBlock(out_channels, out_channels, 3, 1, 1)
        self.fc_mu = nn.Linear(out_channels*3*3, out_channels)
        self.fc_logvar = nn.Linear(out_channels*3*3, out_channels)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.fc1 = nn.Linear(out_channels*3*3 + 10, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.fc3 = nn.Linear(64, out_channels)
        self.mu = None
        self.logvar = None
        self.z_mu = None
        self.z_logvar = None

    def forward(self, x, label=None, VAE=False, CVAE=False):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.
            label (torch.Tensor): Label tensor. Default is None.
            VAE (bool): Flag for Variational Autoencoder (VAE) mode. Default is False.
            CVAE (bool): Flag for Conditional Variational Autoencoder (CVAE) mode. Default is False.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.conv1(x)
        x = self.selu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if VAE or CVAE:
            x_ = x.view(x.shape[0], -1)
            mu = self.fc_mu(x_)
            logvar = self.fc_logvar(x_)
            if CVAE:
                targets = one_hot(label, 10).to(x_.device)
                inputs = torch.cat((x_, targets), dim=1)
                h1 = self.elu(self.fc1(inputs))
                z_mu = self.fc2(h1)
                z_logvar = self.fc3(h1)
                self.z_mu = z_mu
                self.z_logvar = z_logvar
                return self.reparametrize(VAE, CVAE), z_mu, z_logvar
            self.mu = mu
            self.logvar = logvar
            return self.reparametrize(VAE, CVAE), mu, logvar
        return x

    def reparametrize(self, VAE=False, CVAE=False, generate=False):
        """
        Reparametrization trick for VAE and CVAE.

        Args:
            VAE (bool): Flag for Variational Autoencoder (VAE) mode. Default is False.
            CVAE (bool): Flag for Conditional Variational Autoencoder (CVAE) mode. Default is False.

        Returns:
            torch.Tensor: Reparametrized tensor.

        """
        if generate:
            if not CVAE:
                raise ValueError("CVAE flag must be set to True for generation.")
            std = torch.mean(torch.exp_(0.5*self.z_logvar),dim=0,keepdim=True)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(torch.mean(self.z_mu,dim=0,keepdim=True))
        if VAE:
            std = torch.exp_(0.5*self.logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(self.mu)
        elif CVAE:
            std = torch.exp_(0.5*self.z_logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(self.z_mu)
        else:
            return None
            

class DecoderBlock(nn.Module):
    """
    A class representing a decoder block in a neural network.

    This class inherits from the `nn.Module` class of the PyTorch library.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the convolutional kernel.
        stride (int or tuple): The stride of the convolution.
        padding (int or tuple): The padding added to the input.

    Methods:
        __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            Initializes a DecoderBlock object.
        forward(self, x):
            Performs a forward pass through the decoder block.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Initializes a DecoderBlock object.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int or tuple): The size of the convolutional kernel.
            stride (int or tuple): The stride of the convolution.
            padding (int or tuple): The padding added to the input.

        """
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass through the decoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.conv4(out)
        out += self.upsample(identity)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """
    Decoder module for a neural network.

    Args:
        in_channels (int): Number of input channels. Default is 16.
        out_channels (int): Number of output channels. Default is 1.

    Attributes:
        fc_cvae (nn.Linear): Fully connected layer for conditional variational autoencoder.
        fc_vae (nn.Linear): Fully connected layer for variational autoencoder.
        elu (nn.ELU): Exponential Linear Unit activation function.
        selu (nn.SELU): Scaled Exponential Linear Unit activation function.
        conv1 (nn.ConvTranspose2d): Transposed convolutional layer.
        block1 (DecoderBlock): Decoder block 1.
        block2 (DecoderBlock): Decoder block 2.
        block3 (DecoderBlock): Decoder block 3.
        block4 (DecoderBlock): Decoder block 4.
        conv3 (nn.ConvTranspose2d): Transposed convolutional layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        forward(x, label=None, VAE=False, CVAE=False): Forward pass of the decoder.

    """

    def __init__(self, in_channels=16, out_channels=1):
        super(Decoder, self).__init__()
        self.fc_cvae = nn.Linear(in_channels+10, in_channels*3*3)
        self.fc_vae = nn.Linear(in_channels, in_channels*3*3)

        self.elu = nn.ELU()
        self.selu = nn.SELU()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=True)

        self.block1 = DecoderBlock(in_channels, 8, 3, 2, 1)
        self.block2 = DecoderBlock(8, 4, 3, 2, 0)
        self.block3 = DecoderBlock(4, 2, 4, 1, 0)
        self.block4 = DecoderBlock(2, 2, 3, 1, 0)

        self.conv3 = nn.ConvTranspose2d(2, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, label=None, VAE=False, CVAE=False):
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor.
            label (torch.Tensor, optional): Label tensor. Default is None.
            VAE (bool, optional): Flag indicating whether to use variational autoencoder. Default is False.
            CVAE (bool, optional): Flag indicating whether to use conditional variational autoencoder. Default is False.

        Returns:
            torch.Tensor: Output tensor.

        """
        if VAE or CVAE:
            if CVAE:
                targets = one_hot(label,10).to(x.device)
                x = torch.cat((x,targets),dim=1)
                x = self.elu(self.fc_cvae(x))
            else:
                x = self.elu(self.fc_vae(x))
            x = x.view(x.size(0),-1,3,3)

        x = self.conv1(x)
        x = self.selu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x
    

class AELossFn(nn.Module):
    """
    A custom loss function for Autoencoders.

    This loss function combines Binary Cross Entropy (BCE) with Logits Loss and Mean Squared Error (MSE) Loss.
    The BCE loss is weighted by 0.1 and the MSE loss is weighted by 0.9.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output. Default is 'mean'.

    Attributes:
        bceloss (nn.BCEWithLogitsLoss): Binary Cross Entropy with Logits Loss.
        mseloss (nn.MSELoss): Mean Squared Error Loss.
        reduction (str): Specifies the reduction to apply to the output.

    """

    def __init__(self, reduction: str = 'mean'):
        super(AELossFn, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss()
        self.mseloss = nn.MSELoss()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Forward pass of the loss function.

        Args:
            logits (torch.Tensor): The predicted logits from the model.
            targets (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The computed loss value.

        """
        logits = logits.clone().requires_grad_(True)
        targets = targets.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        loss = 0.1 * (self.bceloss(logits, targets)) + 0.9 * (self.mseloss(logits, targets))
        return loss


class VAELossFn(nn.Module):
    """
    This class represents the loss function for a Variational Autoencoder (VAE) model.
    
    Args:
        reduction (str, optional): Specifies the reduction type for the loss. Default is 'mean'.
    
    Attributes:
        bceloss (nn.BCEWithLogitsLoss): Binary Cross Entropy loss function.
        mseloss (nn.MSELoss): Mean Squared Error loss function.
        reduction (str): The reduction type for the loss.
    """
     
    def __init__(self, reduction:str = 'mean'):
        super(VAELossFn, self).__init__()
        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()
        self.reduction = reduction

    def forward(self, logits, targets, mu, logvar):
        """
        Computes the loss for the VAE model.
        
        Args:
            logits (torch.Tensor): The predicted logits from the VAE model.
            targets (torch.Tensor): The target values for the VAE model.
            mu (torch.Tensor): The mean of the latent space distribution.
            logvar (torch.Tensor): The log variance of the latent space distribution.
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        logits = logits.clone().requires_grad_(True)
        targets = targets.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        reconstruct_loss = 0.1*(self.bce_loss(logits,targets)) + 0.9*(self.mse_loss(logits, targets))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow_(2) - logvar.exp_())
        return reconstruct_loss + KLD


def ParameterSelector(E, D):
    return (list(E.parameters()) + list(D.parameters()))


def plot_tsne(logits,epoch,labels=None,VAE=False,CVAE=False):
    logits = logits.view(logits.shape[0],-1).numpy()
    tsne = TSNE(n_components=3)
    tsne_logits = tsne.fit_transform(logits)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_logits[:,0], tsne_logits[:,1], tsne_logits[:,2], c=labels, cmap='tab10')
    plt.savefig("tsnePlots/{}AE_epoch_{}.png".format(("V" if VAE else "CV" if CVAE else ""),epoch))
    plt.close()


def save_checkpoint(encoder, decoder, path):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'mu' : encoder.mu,
        'logvar' : encoder.logvar,
        'z_mu' : encoder.z_mu,
        'z_logvar' : encoder.z_logvar
    }, path)

def load_checkpoint(encoder, decoder, path):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.mu = checkpoint['mu']
    encoder.logvar = checkpoint['logvar']
    encoder.z_mu = checkpoint['z_mu']
    encoder.z_logvar = checkpoint['z_logvar']
    return encoder, decoder


class AETrainer:
    """
    AETrainer is a class that trains an autoencoder model using a given data loader, encoder, decoder, loss function, and optimizer.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader object that provides the training data.
        encoder (Encoder): The encoder model used in the autoencoder.
        decoder (Decoder): The decoder model used in the autoencoder.
        loss_fn: The loss function used for training the autoencoder.
        optimizer: The optimizer used for updating the model parameters during training.
        gpu (str or bool, optional): Specifies whether to use GPU for training. Defaults to "F" (False).

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader object that provides the training data.
        encoder (Encoder): The encoder model used in the autoencoder.
        decoder (Decoder): The decoder model used in the autoencoder.
        device (str): The device (CPU or GPU) on which the model is trained.
    
    Methods:
        train(optimizer, loss_fn): Trains the autoencoder model using the provided optimizer and loss function.

    """

    def __init__(self, data_loader, encoder, decoder, loss_fn, optimizer, gpu="F"):
        self.data_loader = data_loader
        self.encoder = encoder
        self.decoder = decoder
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer, loss_fn)

    def train(self, optimizer, loss_fn):
        """
        Trains the autoencoder model using the provided optimizer and loss function.

        Args:
            optimizer: The optimizer used for updating the model parameters during training.
            loss_fn: The loss function used for training the autoencoder.
        """

        for epoch in range(EPOCH):
            if (epoch+1) % 10 == 0:
                logits_tsne = torch.tensor([])
                labels_tsne = torch.tensor([])

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target, labels) in enumerate(self.data_loader):
                data, target, labels = data.to(self.device), target.to(self.device), labels.to(self.device)
                logits = self.encoder(data)
                output = self.decoder(logits)

                if (epoch+1) % 10 == 0:
                    logits_tsne = torch.cat((logits_tsne, logits.clone().cpu().detach()), dim=0)
                    labels_tsne = torch.cat((labels_tsne, labels.clone().cpu().detach()), dim=0)

                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])], dtype=torch.float32))

                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch, loss, similarity))

            self.encoder.eval()
            self.decoder.eval()

            epoch_loss = []
            epoch_sim = []

            for minibatch, (data, target, _) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = self.encoder(data)
                output = self.decoder(logits)

                loss = loss_fn(output, target)
                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])], dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, torch.mean(torch.tensor(epoch_loss)), torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1) % 10 == 0:
                plot_tsne(logits_tsne, epoch, labels_tsne)
            
        save_checkpoint(self.encoder, self.decoder, "checkpoints/checkpointAE_trained.pth")


class VAETrainer:
    """
    A class that represents a Variational Autoencoder (VAE) trainer.

    Parameters:
    - data_loader (torch.utils.data.DataLoader): The data loader object that provides the training data.
    - encoder (Encoder): The encoder model used in the VAE.
    - decoder (Decoder): The decoder model used in the VAE.
    - loss_fn: The loss function used for training the VAE.
    - optimizer: The optimizer used for updating the model parameters.
    - gpu (str or bool, optional): Specifies whether to use GPU for training. Defaults to "F" (False).

    Attributes:
    - data_loader (torch.utils.data.DataLoader): The data loader object that provides the training data.
    - encoder (Encoder): The encoder model used in the VAE.
    - decoder (Decoder): The decoder model used in the VAE.
    - device (str): The device (CPU or GPU) on which the model is trained.

    Methods:
    - train(optimizer, loss_fn): Trains the VAE model using the provided optimizer and loss function.

    """

    def __init__(self, data_loader, encoder, decoder, loss_fn, optimizer, gpu="F"):
        """
        Initializes a new instance of the VAETrainer class.

        Parameters:
        - data_loader (torch.utils.data.DataLoader): The data loader object that provides the training data.
        - encoder (Encoder): The encoder model used in the VAE.
        - decoder (Decoder): The decoder model used in the VAE.
        - loss_fn: The loss function used for training the VAE.
        - optimizer: The optimizer used for updating the model parameters.
        - gpu (str or bool, optional): Specifies whether to use GPU for training. Defaults to "F" (False).
        """
        self.data_loader = data_loader
        self.encoder = Encoder()
        self.decoder = Decoder()
        optimizer.param_groups[0]['params'] = ParameterSelector(self.encoder,self.decoder)
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer,loss_fn)

    def train(self, optimizer, loss_fn):
        """
        Trains the VAE model using the provided optimizer and loss function.

        Parameters:
        - optimizer: The optimizer used for updating the model parameters.
        - loss_fn: The loss function used for training the VAE.
        """
        for epoch in range(EPOCH):
            if (epoch+1)%10 == 0:
                logits_tsne = torch.tensor([])
                labels_tsne = torch.tensor([])

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target, labels) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                reparam,mu,logvar = self.encoder(data,VAE=True)
                output = self.decoder(reparam,VAE=True)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,reparam.clone().cpu().detach()),dim=0)
                    labels_tsne = torch.cat((labels_tsne,labels.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))

            self.encoder.eval()
            self.decoder.eval()

            epoch_loss = []
            epoch_sim = []

            for minibatch, (data, target, _) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                reparam,mu,logvar = self.encoder(data,VAE=True)
                output = self.decoder(reparam,VAE=True)

                loss = loss_fn(output, target, mu, logvar)
                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch,labels_tsne,VAE=True)

        save_checkpoint(self.encoder,self.decoder,"checkpoints/checkpointVAE_trained.pth")


class AE_TRAINED:
    """
    A class representing a trained Autoencoder (AE) model.

    Attributes:
    - device (str): The device on which the model is loaded ('cuda', 'mps', or 'cpu').
    - encoder (Encoder): The encoder component of the AE model.
    - decoder (Decoder): The decoder component of the AE model.

    Methods:
    - __init__(self, gpu=False): Initializes the AE_TRAINED object.
    - from_path(self, sample, original, type): Computes the similarity score between 'sample' and 'original' images.

    """

    def __init__(self, gpu=False):
        """
        Initializes the AE_TRAINED object.

        Parameters:
        - gpu (bool): Specifies whether to use GPU acceleration. Default is False.

        """
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpointAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def from_path(self, sample, original, type):
        """
        Compute similarity score of both 'sample' and 'original' images and return it as a float.

        Parameters:
        - sample (str): The path to the sample image.
        - original (str): The path to the original image.
        - type (str): The type of similarity score to compute. Valid options are 'SSIM' or 'PSNR'.

        Returns:
        - float: The similarity score between the sample and original images.

        Raises:
        - Exception: If an invalid type is provided. Use 'SSIM' or 'PSNR'.

        """
        sample_img = torchvision.io.read_image(sample).float().div_(255.0)
        original_img = torchvision.io.read_image(original).float().div_(255.0)
        sample_img = sample_img.unsqueeze(0).to(self.device)
        original_img = original_img.unsqueeze(0).to(self.device)
        logits = self.encoder(sample_img)
        output = self.decoder(logits)
        if type == "SSIM":
            return ssim(output, original_img).item()
        elif type == "PSNR":
            return psnr(output, original_img)
        else:
            raise Exception("Invalid type. Use 'SSIM' or 'PSNR'.")
        

class VAE_TRAINED:
    """
    A class representing a trained Variational Autoencoder (VAE).

    Attributes:
        device (str): The device on which the VAE is loaded.
        encoder (Encoder): The encoder network of the VAE.
        decoder (Decoder): The decoder network of the VAE.

    Methods:
        __init__(self, gpu=False): Initializes the VAE_TRAINED object.
        from_path(self, sample, original, type): Computes the similarity score between 'sample' and 'original' images.

    """

    def __init__(self, gpu=False):
        """
        Initializes the VAE_TRAINED object.

        Args:
            gpu (bool, optional): Specifies whether to use GPU for computation. Defaults to False.

        """
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if (gpu=='T' or gpu==True) else 'cpu'
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpointVAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def from_path(self, sample, original, type):
        """
        Computes the similarity score between 'sample' and 'original' images.

        Args:
            sample (str): The path to the sample image.
            original (str): The path to the original image.
            type (str): The type of similarity score to compute. Valid options are 'SSIM' or 'PSNR'.

        Returns:
            float: The similarity score between the images.

        Raises:
            Exception: If an invalid type is provided.

        """
        sample_img = torchvision.io.read_image(sample).float().div_(255.0)
        original_img = torchvision.io.read_image(original).float().div_(255.0)
        sample_img = sample_img.unsqueeze(0).to(self.device)
        original_img = original_img.unsqueeze(0).to(self.device)
        reparam,_,_ = self.encoder(sample_img,VAE=True)
        output = self.decoder(reparam,VAE=True)
        if type == "SSIM":
            return ssim(output, original_img).item()
        elif type == "PSNR":
            return psnr(output, original_img)
        else:
            raise Exception("Invalid type. Use 'SSIM' or 'PSNR'.")


class CVAELossFn(nn.Module):
    """
    CVAELossFn is a custom loss function for Conditional Variational Autoencoder (CVAE) models.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the loss. Default is 'mean'.
    
    Attributes:
        bce_loss (nn.BCELoss): Binary Cross Entropy loss function.
        mse_loss (nn.MSELoss): Mean Squared Error loss function.
        reduction (str): The reduction to apply to the loss.
    """
    
    def __init__(self, reduction:str = 'mean'):
        super(CVAELossFn, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.reduction = reduction

    def forward(self, logits, targets, mu, logvar, label):
        """
        Forward pass of the CVAELossFn.
        
        Args:
            logits (torch.Tensor): The predicted logits from the model.
            targets (torch.Tensor): The target values.
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.
            label (torch.Tensor): The label for conditional generation.
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        logits = logits.clone().requires_grad_(True)
        logits = logits.view(logits.shape[0],-1)
        targets = targets.view(targets.shape[0],-1)
        reconstruct_loss = 0.1*(self.bce_loss(logits,targets)) + 0.9*(self.mse_loss(logits, targets))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow_(2) - logvar.exp_())
        return torch.mean((reconstruct_loss + KLD)*label)


class CVAE_Trainer:
    """
    Class for training a Conditional Variational Autoencoder (CVAE).

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader object for loading the training data.
        encoder (Encoder): The encoder model for the CVAE.
        decoder (Decoder): The decoder model for the CVAE.
        loss_fn: The loss function for the CVAE.
        optimizer: The optimizer for the CVAE.

    Attributes:
        data_loader (torch.utils.data.DataLoader): The data loader object for loading the training data.
        encoder (Encoder): The encoder model for the CVAE.
        decoder (Decoder): The decoder model for the CVAE.
        device (str): The device on which the CVAE will be trained ('cuda', 'mps', or 'cpu').

    Methods:
        train(optimizer, loss_fn): Trains the CVAE model using the given optimizer and loss function.

    """

    def __init__(self, data_loader:torch.utils.data.DataLoader, encoder:Encoder, decoder:Decoder, loss_fn, optimizer):
        self.data_loader = data_loader
        self.encoder = Encoder()
        self.decoder = Decoder()
        optimizer.param_groups[0]['params'] = ParameterSelector(self.encoder,self.decoder)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.train(optimizer,loss_fn)

    def train(self, optimizer, loss_fn):
        """
        Trains the CVAE model using the given optimizer and loss function.

        Args:
            optimizer: The optimizer for the CVAE.
            loss_fn: The loss function for the CVAE.

        Returns:
            None

        """
        for epoch in range(EPOCH):
            if (epoch+1)%10 == 0:
                logits_tsne = torch.tensor([])
                labels_tsne = torch.tensor([])

            self.encoder.train()
            self.decoder.train()

            for minibatch, (data, target, label) in enumerate(self.data_loader):
                data, target, label = (data.to(self.device),
                                       target.to(self.device),
                                       label.to(self.device))
                reparam,mu,logvar = self.encoder(data,label,CVAE=True)
                output = self.decoder(reparam,label,CVAE=True)

                if (epoch+1)%10 == 0:
                    logits_tsne = torch.cat((logits_tsne,reparam.clone().cpu().detach()),dim=0)
                    labels_tsne = torch.cat((labels_tsne,label.clone().cpu().detach()),dim=0)

                loss = loss_fn(output, target, mu, logvar, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                if (minibatch+1) % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
            
            self.encoder.eval()
            self.decoder.eval()

            epoch_loss = []
            epoch_sim = []

            for minibatch, (data, target, label) in enumerate(self.data_loader):
                data, target, label = data.to(self.device), target.to(self.device), label.to(self.device)
                reparam,mu,logvar = self.encoder(data,label,CVAE=True)
                output = self.decoder(reparam,label,CVAE=True)

                loss = loss_fn(output, target, mu, logvar, label)
                similarity = torch.mean(torch.tensor([ssim(output[i].clone().cpu().detach(),
                                                           target[i].clone().cpu().detach())
                                                           for i in range(target.shape[0])],dtype=torch.float32))
                epoch_loss.append(loss.item())
                epoch_sim.append(similarity)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,torch.mean(torch.tensor(epoch_loss)),torch.mean(torch.tensor(epoch_sim))))

            if (epoch+1)%10 == 0:
                plot_tsne(logits_tsne,epoch,labels_tsne,CVAE=True)
        
        save_checkpoint(self.encoder,self.decoder,"checkpoints/checkpointCVAE_trained.pth")


class CVAE_Generator:
    """
    Class representing a Conditional Variational Autoencoder (CVAE) generator.

    Attributes:
        device (str): The device on which the model is trained ('cuda', 'mps', or 'cpu').
        encoder (Encoder): The encoder model used in the CVAE.
        decoder (Decoder): The decoder model used in the CVAE.

    Methods:
        __init__(): Initializes the CVAE_Generator object.
        save_image(digit, save_path): Generates and saves an image using the CVAE.

    """

    def __init__(self):
        """
        Initializes the CVAE_Generator object.

        The constructor sets up the device, initializes the encoder and decoder models,
        loads the trained checkpoint, and sets the models to evaluation mode.

        """
        self.device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.encoder, self.decoder = load_checkpoint(self.encoder,self.decoder,"checkpointCVAE_trained.pth")
        self.encoder.eval()
        self.decoder.eval()

    def save_image(self, digit, save_path):
        """
        Generates and saves an image using the CVAE.

        Args:
            digit (int): The class label of the generated image.
            save_path (str): The path where the generated image will be saved.

        Raises:
            Exception: If the encoder is not trained.

        """
        if self.encoder.z_mu is None:
            raise Exception("Encoder not trained.")
        
        z = self.encoder.reparametrize(VAE=False,CVAE=True,generate=True).to(self.device)
        label = torch.tensor(digit).unsqueeze(0).to(self.device)

        output = self.decoder(z,label,CVAE=True)*255.0
        output = output.to(torch.uint8)
        torchvision.io.write_png(output.squeeze(0),os.path.join(save_path,f"generatedImages/CVAE_Class-{digit}.png"))


def psnr(img1, img2, max_val:float=255.0):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    diff = img1.to(torch.float64).sub_(img2.to(torch.float64))
    mse = torch.mean(diff ** 2)
    if mse==0: return float("inf")
    return 20 * torch.log10(max_val/torch.sqrt(mse)).item()


def ssim(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    channel = 1
    window_size = 11
    K = [0.02, 0.08]
    C1 = K[0]**2
    C2 = K[1]**2

    sigma = 1.
    gauss = [torch.exp_(torch.tensor(-(x - window_size//2)**2/(2*sigma**2))) for x in range(window_size)]
    gauss = torch.tensor(gauss)
    window1d = (gauss/gauss.sum()).unsqueeze(1)
    window2d = window1d.mm(window1d.transpose(0,1)).float().unsqueeze(0).unsqueeze(0)
    window = Variable(window2d.expand(channel, 1, window_size, window_size).contiguous())
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow_(2)
    mu2_sq = mu2.pow_(2)
    mu12 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu12

    numerator = (2*mu12 + C1)*(2*sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)
    ssim_score = numerator/denominator
    return torch.clamp(ssim_score.mean(),min=0,max=1)