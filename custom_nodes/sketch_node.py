import folder_paths
import node_helpers
from PIL import Image, ImageOps, ImageSequence, ImageFile
import numpy as np
import torch
import hashlib
import os
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
norm_layer = nn.InstanceNorm2d
import requests

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Sketcher(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Sketcher, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        self.model2 = []
        for _ in range(n_residual_blocks):
            self.model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*self.model2)
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        x = x[:,:3,:,:]
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

class LineartDector:
    def __init__(self):
        self.model = Sketcher(3, 1, 3)
        self.model.load_state_dict(torch.load('models/model_simple_lines.pth', map_location=torch.device('cuda')))
        self.model.eval()

    def predict(self, input_img):
        input_img = torch.tensor(input_img)
        sketching = 0
        with torch.no_grad():
            sketching = self.model(input_img)[0].detach()
            sketching = sketching.permute(1, 2, 0)
        return sketching

class Example:
    def __init__(self):
        self.lineart_detector = LineartDector()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "image": ("IMAGE", ),
                             }}

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Example"
    FUNCTION = "sketch"

    def sketch(self, image):
        image = image.permute(0, 3, 1, 2)
        sketch = self.lineart_detector.predict(image)
        sketch = sketch.squeeze()  # 变为 [h, w]
        sketch = sketch.unsqueeze(-1).repeat(1, 1, 3)  # 变为 [h, w, 3]
        sketch = sketch.unsqueeze(0)
        return (sketch,)

# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/hello")
async def get_hello(request):
    return web.json_response("hello")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Example": Example
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Generate Sketch Node"
}