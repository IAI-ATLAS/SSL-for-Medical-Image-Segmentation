from typing import List
from torch_cka import CKA
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from warnings import warn
from tqdm import tqdm
import math

class CudaCKA(object):
    def __init__(self, device='cuda'):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


class CKASimplified(CKA):
    
    
    def __init__(self, model1: nn.Module, model2: nn.Module, model1_name: str = None, model2_name: str = None, model1_layers: List[str] = None, model2_layers: List[str] = None, device: str = 'cuda'):
        super().__init__(model1, model2, model1_name, model2_name, model1_layers, model2_layers, device)
        self.cuda_cka = CudaCKA()

    
    def compare(self,
                dataloader: DataLoader,
    ):
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = len(dataloader)
        
        cka_sum = 0

        for (x1, *_) in tqdm(dataloader, desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x1.to(self.device))
            
            cka_vals = []
            for i, ((name1, feat1),(name2, feat2)) in enumerate(zip(self.model1_features.items(), self.model2_features.items())):
                X = feat1.flatten(1)
                Y = feat2.flatten(1)
                cka_vals.append(float(torch.nan_to_num(self.cuda_cka.linear_CKA(X, Y))))
            cka_sum += sum(cka_vals) / len(cka_vals)
        return cka_sum / num_batches
    
    def export(self):
        return self.hsic_matrix
                
                