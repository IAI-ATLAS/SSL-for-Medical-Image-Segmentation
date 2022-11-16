from torch_cka import CKA
from pathlib import Path
from torchvision.models import resnet50

from DLIP.utils.evaluation.cka_simplified import CKASimplified


def calculate_cka(data,directory,model,ref_model):
        Path(f"{directory}/cka").mkdir(parents=True, exist_ok=True)
        
        data.shuffle = False
        # batch size should be small -> 2 models on gpu
        data.batch_size = 12
        
        for i in range(1,5):
                layers = []
                for name,weights in model.named_modules():
                        if f'backbone.{i}' in name:
                                layers.append(name)
                cka = CKASimplified(model, ref_model,
                        model1_name="ResNet18",
                        model2_name="ResNet34",
                        model1_layers=layers,
                        model2_layers=layers,
                        device='cuda')
                cka_val = cka.compare(data.test_dataloader()) # secondary dataloader is optional
                print(f'Conv{i} : {cka_val}')
        print()

        # cka = CKA(model, ref_model,
        #         model1_name="model",   # good idea to provide names to avoid confusion
        #         model2_name="ref_model",   
        #         model1_layers=layers,
        #         model2_layers=layers,
        #         device='cuda')

        # cka.compare(data.val_dataloader()) # secondary dataloader is optional
        # results = cka.export()  # returns a dict that contains model names, layer names
        #                         # and the CKA matrix
        # import matplotlib.pyplot as plt
        # cka.plot_results(save_path=f"{directory}/cka/cka.png")