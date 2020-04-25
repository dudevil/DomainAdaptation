import torch
import torch.nn as nn

import configs.dann_config as dann_config
import models.backbone_models as backbone_models
import models.blocks as blocks


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass


class DANNModel(BaseModel):
    def __init__(self):
        super(DANNModel, self).__init__()
        self.features, self.pooling, self.class_classifier, \
            domain_input_len, self.classifier_before_domain_cnt = backbone_models.get_backbone_model()
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(domain_input_len, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )        

    def forward(self, input_data, rev_grad_alpha=dann_config.GRADIENT_REVERSAL_LAYER_ALPHA):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (map of tensors) - map with model output tensors
        """
        features = self.features(input_data)
        features = self.pooling(features)
        features = torch.flatten(features, 1)

        output_classifier = features
        domain_features = features
        classifier_layers_outputs = []
        for i, block in enumerate(self.class_classifier):
            if i == self.classifier_before_domain_cnt:
                domain_features = output_classifier
            output_classifier = block(output_classifier)
            classifier_layers_outputs.append(output_classifier)

        reversed_features = blocks.GradientReversalLayer.apply(domain_features, rev_grad_alpha)
        output_domain = self.domain_classifier(reversed_features)

        output = {
            "class": output_classifier,
            "domain": output_domain,
        }
        if dann_config.LOSS_NEED_INTERMEDIATE_LAYERS:
            output["classifier_layers"] = classifier_layers_outputs

        return output

    def predict(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        return self.forward(input_data)["class"]
