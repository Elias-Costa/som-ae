import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
class AutoencoderConv(nn.Module):
    def __init__(self, dimImagem, device=torch.device("cpu")):
        super(AutoencoderConv, self).__init__()

        self.dimImagem = dimImagem
        self.img = torch.rand(dimImagem[0], dimImagem[1], dimImagem[2]).to(device)

        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(dimImagem[0], 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.conv4 = nn.Conv2d(16, 32, 3, 1)

        self.convTransp1 = nn.ConvTranspose2d(32, 16, 3, 1)
        self.convTransp2 = nn.ConvTranspose2d(16, 8, 3, 1)
        self.convTransp3 = nn.ConvTranspose2d(8, 4, 3, 1)
        self.convTransp4 = nn.ConvTranspose2d(4, dimImagem[0], 3, 1)

        if dimImagem[0] == 3:
            self.fcClassificacao1 = nn.Linear(32*4*4, 64)
            self.fcClassificacao2 = nn.Linear(64, 32)
            self.fcClassificacao3 = nn.Linear(32, 10)
        else:
            self.fcClassificacao1 = nn.Linear(32*3*3, 128)
            self.fcClassificacao2 = nn.Linear(128, 64)
            self.fcClassificacao3 = nn.Linear(64, 10)

        nn.init.kaiming_normal_(self.fcClassificacao1.weight, nonlinearity='relu')
        nn.init.constant_(self.fcClassificacao1.bias, 0)
        nn.init.kaiming_normal_(self.fcClassificacao2.weight, nonlinearity='relu')
        nn.init.constant_(self.fcClassificacao2.bias, 0)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        nn.init.constant_(self.conv4.bias, 0)
        nn.init.kaiming_normal_(self.convTransp1.weight, nonlinearity='relu')
        nn.init.constant_(self.convTransp1.bias, 0)
        nn.init.kaiming_normal_(self.convTransp2.weight, nonlinearity='relu')
        nn.init.constant_(self.convTransp2.bias, 0)
        nn.init.kaiming_normal_(self.convTransp3.weight, nonlinearity='relu')
        nn.init.constant_(self.convTransp3.bias, 0)
        nn.init.xavier_uniform_(self.convTransp4.weight)
        nn.init.constant_(self.convTransp4.bias, 0)

    def codificar(self, entrada):
        latente = self.relu(self.conv1(entrada)) #26
        latente = self.relu(self.conv2(latente)) #24
        latente = self.maxPool(latente) #12
        latente = self.relu(self.conv3(latente)) #10
        latente = self.maxPool(latente) #5
        latente = self.relu(self.conv4(latente)) #3
        return latente

    def decodificar(self, latente):
        reconstrucao = self.relu(self.convTransp1(latente)) #5

        reconstrucao = reconstrucao.unsqueeze(0)
        reconstrucao = self.upsample(reconstrucao) #10
        reconstrucao = reconstrucao.squeeze(0)

        reconstrucao = self.relu(self.convTransp2(reconstrucao)) #12

        reconstrucao = reconstrucao.unsqueeze(0)
        reconstrucao = self.upsample(reconstrucao) #24
        reconstrucao = reconstrucao.squeeze(0)

        reconstrucao = self.relu(self.convTransp3(reconstrucao)) #26
        reconstrucao = self.sigmoid(self.convTransp4(reconstrucao)) #28
        return reconstrucao
    
    def ajustarPesosBMU(self, imgEntrada, epocas=1, lr=0.001): 
        transform_list = [
            transforms.RandomRotation(degrees=[-5,5]),
            transforms.RandomCrop(size=(self.dimImagem[1], self.dimImagem[2]),padding=5)
        ]
        transform = transforms.RandomChoice(transform_list)

        funcaoPerda = nn.MSELoss()
        otimizador = optim.Adam([{'params': self.conv1.parameters()},
                              {'params': self.conv2.parameters()},
                              {'params': self.conv3.parameters()},
                              {'params': self.conv4.parameters()},
                              {'params': self.convTransp1.parameters()},
                              {'params': self.convTransp2.parameters()},
                              {'params': self.convTransp3.parameters()},
                              {'params': self.convTransp4.parameters()}], lr)

        for _ in range(epocas):
            transform_img = transform(imgEntrada)
            latente = self.codificar(transform_img)
            reconstrucao = self.decodificar(latente)
            perda = funcaoPerda(reconstrucao, imgEntrada)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
        
    def ajustarPesosVizinho(self, imgEntrada, BMU, lr, h):
        with torch.no_grad():
            self.conv1.weight.data += lr*h*(BMU.conv1.weight.data - self.conv1.weight.data)
            self.conv1.bias.data += lr*h*(BMU.conv1.bias.data - self.conv1.bias.data)
            self.conv2.weight.data += lr*h*(BMU.conv2.weight.data - self.conv2.weight.data)
            self.conv2.bias.data += lr*h*(BMU.conv2.bias.data - self.conv2.bias.data)
            self.conv3.weight.data += lr*h*(BMU.conv3.weight.data - self.conv3.weight.data)
            self.conv3.bias.data += lr*h*(BMU.conv3.bias.data - self.conv3.bias.data)
            self.conv4.weight.data += lr*h*(BMU.conv4.weight.data - self.conv4.weight.data)
            self.conv4.bias.data += lr*h*(BMU.conv4.bias.data - self.conv4.bias.data)
            self.convTransp1.weight.data += lr*h*(BMU.convTransp1.weight.data - self.convTransp1.weight.data)
            self.convTransp1.bias.data += lr*h*(BMU.convTransp1.bias.data - self.convTransp1.bias.data)
            self.convTransp2.weight.data += lr*h*(BMU.convTransp2.weight.data - self.convTransp2.weight.data)
            self.convTransp2.bias.data += lr*h*(BMU.convTransp2.bias.data - self.convTransp2.bias.data)
            self.convTransp3.weight.data += lr*h*(BMU.convTransp3.weight.data - self.convTransp3.weight.data)
            self.convTransp3.bias.data += lr*h*(BMU.convTransp3.bias.data - self.convTransp3.bias.data)
            self.convTransp4.weight.data += lr*h*(BMU.convTransp4.weight.data - self.convTransp4.weight.data)
            self.convTransp4.bias.data += lr*h*(BMU.convTransp4.bias.data - self.convTransp4.bias.data)
        
            self.img = self.codificar(imgEntrada)
            self.img = self.decodificar(self.img)
        
    def congelarPesosCodificacao(self):
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.conv4.weight.requires_grad = False
        self.conv4.bias.requires_grad = False

    def forwardClassificacao(self, latente):
        output = self.relu(self.fcClassificacao1(torch.flatten(latente)))
        output = self.relu(self.fcClassificacao2(output))
        output = self.fcClassificacao3(output)
        return output

    def treinarClassificacaoBMU(self, imgEntrada, rotulo, epocas=1, lr=0.001):
        self.congelarPesosCodificacao()
        funcaoPerda = nn.CrossEntropyLoss()
        otimizador = optim.Adam([{'params': self.fcClassificacao1.parameters()},
                              {'params': self.fcClassificacao2.parameters()},
                              {'params': self.fcClassificacao3.parameters()}], lr)

        for _ in range(epocas):
            output = self.forwardClassificacao(self.codificar(imgEntrada))
            perda = funcaoPerda(output, rotulo)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()

    def ajustarPesosClassificacao(self, BMU, lr, h):
        with torch.no_grad():
            self.fcClassificacao1.weight.data += lr*h*(BMU.fcClassificacao1.weight.data - self.fcClassificacao1.weight.data)
            self.fcClassificacao1.bias.data += lr*h*(BMU.fcClassificacao1.bias.data - self.fcClassificacao1.bias.data)
            self.fcClassificacao2.weight.data += lr*h*(BMU.fcClassificacao2.weight.data - self.fcClassificacao2.weight.data)
            self.fcClassificacao2.bias.data += lr*h*(BMU.fcClassificacao2.bias.data - self.fcClassificacao2.bias.data)
            self.fcClassificacao3.weight.data += lr*h*(BMU.fcClassificacao3.weight.data - self.fcClassificacao3.weight.data)
            self.fcClassificacao3.bias.data += lr*h*(BMU.fcClassificacao3.bias.data - self.fcClassificacao3.bias.data)

    def classificar(self, img):
        latente = self.codificar(img)
        output = self.forwardClassificacao(latente)
        return torch.argmax(output).item()
