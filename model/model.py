import torch.nn as nn
import torch

from base import BaseModel

# model 1


class PlantTraitsModel_ViTb_Dense(BaseModel):

    def __init__(self,
                 input_dim=163,
                 num_classes=6,
                 image_output_dim=64,
                 tabular_output_dim=64,
                 hidden_dim=[64, 128, 256, 128, 64],
                 dropout=0.5):
        super(PlantTraitsModel_ViTb_Dense, self).__init__()
        self.image_encoder = self._create_image_encoder(
            output_dim=image_output_dim)
        self.tabular_encoder = self._create_tabular_encoder(
            input_dim=input_dim,
            output_dim=tabular_output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout)
        in_features = image_output_dim + tabular_output_dim
        self.head = nn.Linear(in_features, num_classes)
        self.aux_head = nn.Linear(in_features, num_classes)

    def forward(self, image, features):
        image = self.image_encoder(image)
        features = self.tabular_encoder(features)
        x = torch.cat([image, features], dim=1)
        x1 = self.head(x)
        x2 = self.aux_head(x)
        return {'head': x1, 'aux_head': x2}

    def _create_tabular_encoder(self, input_dim, output_dim, hidden_dim,
                                dropout):
        hidden_network = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for i in range(1, len(hidden_dim)):
            hidden_network.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            hidden_network.append(nn.ReLU())
            hidden_network.append(nn.Dropout(dropout))
        hidden_network.append(nn.Linear(hidden_dim[-1], output_dim))

        return nn.Sequential(*hidden_network)

    def _create_image_encoder(self, output_dim):
        from torchvision.models import vit_b_16
        image_encoder = vit_b_16(weights=None)

        head_in_features = image_encoder.heads.head.in_features
        image_encoder.heads.head = nn.Linear(head_in_features, output_dim)

        return image_encoder


class PlantTraitsModel_ViTl_Dense_attention(BaseModel):

    def __init__(self,
                 input_dim=163,
                 num_classes=6,
                 embed_dim=64,
                 num_heads=8,
                 hidden_dim=[64, 128, 256, 128, 64],
                 dropout=0.5):
        super(PlantTraitsModel_ViTl_Dense_attention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.image_encoder = self._create_image_encoder(
            output_dim=embed_dim)
        self.tabular_encoder = self._create_tabular_encoder(
            input_dim=input_dim,
            output_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        in_features = embed_dim
        self.head = nn.Linear(in_features, num_classes)
        self.aux_head = nn.Linear(in_features, num_classes)

    def forward(self, image, features):
        image = self.image_encoder(image)
        features = self.tabular_encoder(features)
        x = self.attention(query=image, key=features, value=features, need_weights=False)[0]
        x1 = self.head(x)
        x2 = self.aux_head(x)
        return {'head': x1, 'aux_head': x2}

    def _create_tabular_encoder(self, input_dim, output_dim, hidden_dim,
                                dropout):
        hidden_network = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for i in range(1, len(hidden_dim)):
            hidden_network.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            hidden_network.append(nn.ReLU())
            hidden_network.append(nn.Dropout(dropout))
        hidden_network.append(nn.Linear(hidden_dim[-1], output_dim))

        return nn.Sequential(*hidden_network)

    def _create_image_encoder(self, output_dim):
        from torchvision.models import vit_b_16
        image_encoder = vit_b_16(weights=None)

        head_in_features = image_encoder.heads.head.in_features
        image_encoder.heads.head = nn.Linear(head_in_features, output_dim)

        return image_encoder
    


class PlantTraitsModel_Resnet_Dense_attention(BaseModel):

    def __init__(self,
                 input_dim=163,
                 num_classes=6,
                 embed_dim=64,
                 num_heads=8,
                 hidden_dim=[64, 128, 256, 128, 64],
                 dropout=0.5):
        super(PlantTraitsModel_Resnet_Dense_attention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.image_encoder = self._create_image_encoder(
            output_dim=embed_dim)
        self.tabular_encoder = self._create_tabular_encoder(
            input_dim=input_dim,
            output_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        in_features = embed_dim
        self.head = nn.Linear(in_features, num_classes)
        self.aux_head = nn.Linear(in_features, num_classes)

    def forward(self, image, features):
        image = self.image_encoder(image)
        features = self.tabular_encoder(features)
        x = self.attention(query=image, key=features, value=features, need_weights=False)[0]
        x1 = self.head(x)
        x2 = self.aux_head(x)
        return {'head': x1, 'aux_head': x2}

    def _create_tabular_encoder(self, input_dim, output_dim, hidden_dim,
                                dropout):
        hidden_network = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for i in range(1, len(hidden_dim)):
            hidden_network.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            hidden_network.append(nn.ReLU())
            hidden_network.append(nn.Dropout(dropout))
        hidden_network.append(nn.Linear(hidden_dim[-1], output_dim))

        return nn.Sequential(*hidden_network)

    def _create_image_encoder(self, output_dim):
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        image_encoder = resnet50(weights=weights)

        # freeze the model
        for param in image_encoder.parameters():
            param.requires_grad = False

        # add fc layer
        in_features = image_encoder.fc.in_features
        image_encoder.fc = nn.Linear(in_features, output_dim)

        return image_encoder
    

class PlantTraitsModel_Resnet_Dense(BaseModel):

    def __init__(self,
                 input_dim=163,
                 num_classes=6,
                 image_output_dim=64,
                 tabular_output_dim=64,
                 hidden_dim=[64, 128, 256, 128, 64],
                 dropout=0.5):
        super(PlantTraitsModel_Resnet_Dense, self).__init__()
        self.image_encoder = self._create_image_encoder(
            output_dim=image_output_dim)
        self.tabular_encoder = self._create_tabular_encoder(
            input_dim=input_dim,
            output_dim=tabular_output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout)
        in_features = image_output_dim + tabular_output_dim
        self.head = nn.Linear(in_features, num_classes)
        self.aux_head = nn.Linear(in_features, num_classes)

    def forward(self, image, features):
        image = self.image_encoder(image)
        features = self.tabular_encoder(features)
        x = torch.cat([image, features], dim=1)
        x1 = self.head(x)
        x2 = self.aux_head(x)
        return {'head': x1, 'aux_head': x2}

    def _create_tabular_encoder(self, input_dim, output_dim, hidden_dim,
                                dropout):
        hidden_network = [
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for i in range(1, len(hidden_dim)):
            hidden_network.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            hidden_network.append(nn.ReLU())
            hidden_network.append(nn.Dropout(dropout))
        hidden_network.append(nn.Linear(hidden_dim[-1], output_dim))

        return nn.Sequential(*hidden_network)

    def _create_image_encoder(self, output_dim):
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        image_encoder = resnet50(weights=weights)

        # freeze the model
        for param in image_encoder.parameters():
            param.requires_grad = False

        # add fc layer
        in_features = image_encoder.fc.in_features
        image_encoder.fc = nn.Linear(in_features, output_dim)

        return image_encoder