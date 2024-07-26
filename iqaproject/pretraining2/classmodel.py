import torch
import copy
from torch import nn
import timm


def create_model(model_name):
    if model_name.startswith('resnet'):
        model = timm.create_model(model_name, features_only=True, out_indices=(1, 2, 3), pretrained=True)

    else:
        model = timm.create_model(model_name, pretrained=True)

    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    print(f'{model_name} number of parameters: {num_params_m:.2f}M')

    return model






class ClassModel(nn.Module):
    def __init__(self, embed_dim=384):
        super(ClassModel, self).__init__()

        self.backbone, selected_blocks= self.build_backbone(backbone_name='vit_small_patch16_224')
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.class_branch = selected_blocks
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 2)  # Change the output size to 2
        )

    def build_backbone(self, backbone_name, layer_num=6):
        backbone = create_model(backbone_name)

        selected_blocks = copy.deepcopy(backbone.blocks[layer_num:9])

        del backbone.blocks[layer_num:]
        # for i, block in enumerate(backbone.blocks):
        #     print(f"Remaining block {i}: {block}")

        return backbone, selected_blocks


    def forward(self, x):


        x = self.backbone.forward_features(x)

        x = self.class_branch(x)
        # for i, block in enumerate(self.class_branch):
        #     print(f"Block {i}: {block}")

        x = self.fc(x.mean(dim=1))


        return x








if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = ClassModel()
    print(model(x))
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    print(f'the number of parameters: {num_params_m:.2f}M')




