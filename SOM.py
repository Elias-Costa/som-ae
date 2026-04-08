import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap(nn.Module):
    def __init__(self, classeAutoencoder, dimGrade, dimImagem, device=torch.device("cpu")):
        super(SelfOrganizingMap, self).__init__()

        self.dimGrade = dimGrade
        self.dimImagem = dimImagem
        self.classeAutoencoder = classeAutoencoder
        self.device = device

        if classeAutoencoder is not None:
            self.grade = nn.ModuleList(
                [
                    nn.ModuleList(
                        [classeAutoencoder(dimImagem, device).to(device) for _ in range(dimGrade[1])]
                    )
                    for _ in range(dimGrade[0])
                ]
            )
        else:
            self.grade = torch.rand(dimGrade[0], dimGrade[1], dimImagem[0]*dimImagem[1]*dimImagem[2]).to(device)

    def vizinhanca(self, coordenadasVencedor, coordenadasNeuronio, sigma):
        if self.classeAutoencoder is not None:
            distancia = torch.norm(coordenadasVencedor - coordenadasNeuronio)
            return torch.exp(-(distancia**2)/(2.0*sigma**2))
        else:
            distancia = torch.norm(coordenadasVencedor - coordenadasNeuronio, dim=2)
            return torch.exp(-(distancia**2)/(2.0*sigma**2)).unsqueeze(-1)

    def mexicanHat(self, coordenadasVencedor, coordenadasNeuronio, sigma):
        distancia = torch.norm(coordenadasVencedor - coordenadasNeuronio)
        return (1-(distancia**2)/(sigma**2))*torch.exp(-(distancia**2)/(2.0*sigma**2))

    
    def BMU(self, img):
        with torch.no_grad():
            distMin = torch.norm(self.grade[0][0].img - img, p=1)
            pos = (0,0)
            for i in range(self.dimGrade[0]):
                for j in range(self.dimGrade[1]):
                    dist = torch.norm(self.grade[i][j].img - img, p=1)
                    if(dist < distMin):
                        distMin = dist
                        pos = (i, j)

        return (self.grade[pos[0]][pos[1]], torch.tensor([pos[0], pos[1]], device=self.device))

    def BMU_rec(self, img):
        with torch.no_grad():
            reconstrucao = self.grade[0][0].codificar(img)
            reconstrucao = self.grade[0][0].decodificar(reconstrucao)
            
            distMin = torch.norm(reconstrucao - img, p=1)
            pos = (0,0)

            for i in range(self.dimGrade[0]):
                for j in range(self.dimGrade[1]):
                    reconstrucao = self.grade[i][j].codificar(img)
                    reconstrucao = self.grade[i][j].decodificar(reconstrucao)
                    
                    dist = torch.norm(reconstrucao - img, p=1)
                    if(dist < distMin):
                        distMin = dist
                        pos = (i, j)

        return (self.grade[pos[0]][pos[1]], torch.tensor([pos[0], pos[1]], device=self.device))

    def decaimentoAssintotico(self, valorInicial, epocaAtual, totalEpocas):
        return valorInicial/(1+epocaAtual/(totalEpocas/2))

    def decaimentoExponencial(self, valorInicial, valorFinal, epocaAtual, totalEpocas):
        T = totalEpocas / np.log(valorInicial / valorFinal)
        return valorInicial * np.exp(-epocaAtual / T)

    def treinar(self, trainloader, epocas, lr, sigma):

        if self.classeAutoencoder is not None:
            coordenadasNeuronio = torch.tensor([0.0, 0.0]).to(self.device)
            lr_t = torch.tensor(lr, device=self.device)
            sigma_t = torch.tensor(sigma, device=self.device)

            for e in range(epocas):
                print(f"EPOCA: [{e+1}/{epocas}]")

                lr_t.fill_(self.decaimentoExponencial(lr, 0.001, e, epocas))
                sigma_t.fill_(self.decaimentoExponencial(sigma, 0.4, e, epocas))

                for imagens,_ in trainloader:
                    """if self.device == torch.device("cuda"):
                        imagens = imagens.to(self.device)"""           
                    for img in imagens:
                        #bmu, coord = self.BMU(img)
                        bmu, coord_BMU = self.BMU_rec(img)
                        bmu.ajustarPesosBMU(img)

                        for i in range(self.dimGrade[0]):
                            for j in range(self.dimGrade[1]):
                                coordenadasNeuronio[0].fill_(i)
                                coordenadasNeuronio[1].fill_(j)
                                h = self.vizinhanca(coord_BMU, coordenadasNeuronio, sigma_t)
                                self.grade[i][j].ajustarPesosVizinho(img, bmu, lr_t, h)
        else:
            coordenadasNeuronio = torch.cartesian_prod(torch.arange(self.dimGrade[0]), torch.arange(self.dimGrade[1])).reshape(self.dimGrade[0],self.dimGrade[1],2).float()
            coordenadasNeuronio = coordenadasNeuronio.to(self.device)
            coordenadasVencedor = torch.tensor([0, 0]).to(self.device)
            for e in range(epocas):
                lr_t = self.decaimentoExponencial(lr, 0.001, e, epocas)
                sigma_t = self.decaimentoExponencial(sigma, 0.4, e, epocas)
                print(f"EPOCA: [{e+1}/{epocas}]")
                for imagens, _ in trainloader:
                    for img in imagens:
                        img = torch.flatten(img)
                        distancias = torch.norm(self.grade - img, dim=2).reshape(self.dimGrade[0],self.dimGrade[1])
                        indiceDistMin = torch.argmin(distancias)
                        indiceDistMin = torch.unravel_index(indiceDistMin, self.dimGrade)
                        coordenadasVencedor[0] = indiceDistMin[0].item()
                        coordenadasVencedor[1] = indiceDistMin[1].item()
                        vizinhancaGeral = self.vizinhanca(coordenadasVencedor, coordenadasNeuronio, sigma_t)
                        self.grade += lr_t*vizinhancaGeral*(img - self.grade)
                
    def treinarClassificacao(self, trainloader, epocas, lr, sigma, ajustarVizinhos=True):
        coordenadasNeuronio = torch.tensor([0.0, 0.0]).to(self.device)
        lr_t = torch.tensor(lr, device=self.device)
        sigma_t = torch.tensor(sigma, device=self.device)
        
        for e in range(epocas):
            print(f"EPOCA_CLASSIF [{e+1}/{epocas}]")
            lr_t.fill_(self.decaimentoExponencial(lr, 0.001, e, epocas))
            sigma_t.fill_(self.decaimentoExponencial(sigma, 0.4, e, epocas))

            for imagens, rotulos in trainloader:
                #imagens = imagens.to(self.device)
                #rotulos = rotulos.to(self.device)
                for img, rotulo in zip(imagens, rotulos):
                    bmu, coord_BMU = self.BMU_rec(img)
                    bmu.treinarClassificacaoBMU(img, rotulo, epocas=10)
                    if ajustarVizinhos:
                        for i in range(self.dimGrade[0]):
                            for j in range(self.dimGrade[1]):
                                coordenadasNeuronio[0].fill_(i)
                                coordenadasNeuronio[1].fill_(j)
                                h = self.vizinhanca(coord_BMU, coordenadasNeuronio, sigma_t)
                                self.grade[i][j].ajustarPesosClassificacao(bmu, lr_t, h)

    def acuracia(self, testloader):
        acertos = 0
        totalImgs = 0
        with torch.no_grad():
            for imagens, rotulos in testloader:
                imagens = imagens.to(self.device)
                rotulos = rotulos.to(self.device)
                for img, rotulo in zip(imagens, rotulos):
                    bmu,_ = self.BMU_rec(img)
                    pred = bmu.classificar(img)

                    if pred == rotulo.item():
                        acertos += 1
                    totalImgs += 1

        return acertos/totalImgs
    
    def exibirGrade(self):
        #versao com autoencoder
        if self.classeAutoencoder is not None:
            #img escala de cinza
            if self.dimImagem[0] == 1:
                _, graficos = plt.subplots(self.dimGrade[0], self.dimGrade[1], figsize=(self.dimGrade[0], self.dimGrade[1]))
                for i in range(self.dimGrade[0]):
                    for j in range(self.dimGrade[1]):
                        img_neuronio = self.grade[i][j].img.to(torch.device("cpu")).detach().numpy().reshape(28,28)
                        graficos[i][j].imshow(img_neuronio, cmap='gray', vmin=0.0, vmax=1.0)
                        graficos[i][j].axis('off')
                plt.show()

            #img colorida
            else:
                _, graficos = plt.subplots(self.dimGrade[0], self.dimGrade[1], figsize=(self.dimGrade[0], self.dimGrade[1]))
                for i in range(self.dimGrade[0]):
                    for j in range(self.dimGrade[1]):
                        img_neuronio = self.grade[i][j].img.to(torch.device("cpu")).detach().numpy().transpose(1, 2, 0)
                        graficos[i][j].imshow(img_neuronio)
                        graficos[i][j].axis('off')
                plt.show()
        #versao padrao do SOM
        else:
            _, graficos = plt.subplots(self.dimGrade[0], self.dimGrade[1], figsize=(self.dimGrade[0], self.dimGrade[1]))
            for i in range(self.dimGrade[0]):
                for j in range(self.dimGrade[1]):
                    img_neuronio = self.grade[i][j].to(torch.device("cpu")).detach().numpy().reshape(28,28)
                    graficos[i][j].imshow(img_neuronio, cmap='gray', vmin=0.0, vmax=1.0)
                    graficos[i][j].axis('off')
            plt.show()
