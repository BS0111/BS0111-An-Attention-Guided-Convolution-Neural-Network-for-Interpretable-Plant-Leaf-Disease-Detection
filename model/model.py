# LRP
# ---------------------------------------------
# 1. Import required libraries
# ---------------------------------------------
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
import os
import shutil
from sklearn.model_selection import train_test_split

# Check CUDA availability and set device
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------
# 2. Load and process the dataset
# ---------------------------------------------

# Set file paths
CSV_FILE = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
IMAGE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7/images"
OUTPUT_DIR = "/kaggle/working/processed_data"

# Load CSV into dataframe
df = pd.read_csv(CSV_FILE)

# Create a new label column by selecting the disease type with the highest probability
df['label'] = df[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)

# Split into training and testing datasets (80-20 split with stratification)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Helper function to create label-wise subdirectories
def create_directories(root_dir, classes):
    for class_name in classes:
        os.makedirs(os.path.join(root_dir, class_name), exist_ok=True)

# Create output directories
train_output_dir = os.path.join(OUTPUT_DIR, "train")
test_output_dir = os.path.join(OUTPUT_DIR, "test")
create_directories(train_output_dir, df['label'].unique())
create_directories(test_output_dir, df['label'].unique())

# Copy image files to their respective class folders
def organize_images(df, output_dir):
    for _, row in df.iterrows():
        src_path = os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")
        dest_path = os.path.join(output_dir, row['label'], f"{row['image_id']}.jpg")
        shutil.copy(src_path, dest_path)

# Organize images into train and test folders
organize_images(train_df, train_output_dir)
organize_images(test_df, test_output_dir)
print("Dataset organized into train and test folders.")

# ---------------------------------------------
# 3. Load image datasets using ImageFolder
# ---------------------------------------------
TRAIN_ROOT = train_output_dir
TEST_ROOT = test_output_dir

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_ROOT)
test_dataset = torchvision.datasets.ImageFolder(root=TEST_ROOT)

# ---------------------------------------------
# 4. Define the CBAM module
# ---------------------------------------------

# Basic convolution block used by spatial gate
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.relu: x = self.relu(x)
        return x

# Flatten module for the channel attention MLP
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Channel attention gate
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool = F.max_pool2d(x, (x.size(2), x.size(3)))
            elif pool_type == 'lp':
                pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)))
            elif pool_type == 'lse':
                pool = logsumexp_2d(x)

            att = self.mlp(pool)
            channel_att_sum = att if channel_att_sum is None else channel_att_sum + att

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flat = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flat, dim=2, keepdim=True)
    return s + (tensor_flat - s).exp().sum(dim=2, keepdim=True).log()

# Spatial attention gate
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size=7, padding=3, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

# Combined CBAM module
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# ---------------------------------------------
# 5. Define VGG16 with CBAM architecture
# ---------------------------------------------
class VGG16_CBAM(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16_CBAM, self).__init__()
        original_vgg = models.vgg16(pretrained=True)

        # Split VGG16 into 5 stages
        self.stage1 = nn.Sequential(*original_vgg.features[:5])
        self.stage2 = nn.Sequential(*original_vgg.features[5:10])
        self.stage3 = nn.Sequential(*original_vgg.features[10:17])
        self.stage4 = nn.Sequential(*original_vgg.features[17:24])
        self.stage5 = nn.Sequential(*original_vgg.features[24:31])

        # Attach CBAM after each stage
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x, explain=False):
        x = self.stage1(x)
        if not explain: x = self.cbam1(x)

        x = self.stage2(x)
        if not explain: x = self.cbam2(x)

        x = self.stage3(x)
        if not explain: x = self.cbam3(x)

        x = self.stage4(x)
        if not explain: x = self.cbam4(x)

        x = self.stage5(x)
        if not explain: x = self.cbam5(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------------------------------------
# 6. Model training setup
# ---------------------------------------------
model = VGG16_CBAM(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 20

# ---------------------------------------------
# 7. Prepare DataLoaders with transforms
# ---------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load training and testing datasets
train_dataset = datasets.ImageFolder(TRAIN_ROOT, transform=transform)
test_dataset = datasets.ImageFolder(TEST_ROOT, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = train_dataset.classes

# ---------------------------------------------
# 8. Training loop
# ---------------------------------------------
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

# ---------------------------------------------
# 9. Save the trained model
# ---------------------------------------------
torch.save(model.state_dict(), "vgg16_cbam_grape_leaf_Apple.pth")
# ---------------------------------------------
# 10. Reload the trained model for explanation
# ---------------------------------------------
# Instantiate the same model architecture
model = VGG16_CBAM(num_classes=4)
# Load the saved weights from disk
model.load_state_dict(torch.load("/kaggle/working/vgg16_cbam_grape_leaf_Apple.pth"))
# Ensure the model is on the correct device (GPU/CPU)
model.to(device)

# ---------------------------------------------
# 11. Install and import Zennit for LRP
# ---------------------------------------------
# Install the Zennit library (for Layer-wise Relevance Propagation)
!pip install zennit

# Import various Zennit components for attribution
from zennit.attribution import Gradient
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from zennit.image import imgify
from zennit.torchvision import VGGCanonizer
from zennit.rules import ZPlus, Epsilon
from zennit.composites import EpsilonPlusFlat, EpsilonPlus

# ---------------------------------------------
# 12. Prepare model and image for attribution
# ---------------------------------------------
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Re‐instantiate and load the model in eval mode on CPU (for explanation)
model = VGG16_CBAM(num_classes=4)
model.load_state_dict(torch.load(
    '/kaggle/working/vgg16_cbam_grape_leaf_Apple.pth',
    map_location='cpu'         # load onto CPU
))
model.eval()                   # set to evaluation mode

# Load a sample image from the test set
image_path = '/kaggle/working/processed_data/test/rust/Train_1023.jpg'
image = Image.open(image_path).convert('RGB')

# Define basic transform (resize + to‐tensor) for Zennit (no normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# Add batch dimension
input_tensor = transform(image).unsqueeze(0)

# ---------------------------------------------
# 13. Wrapper to force CBAM bypass during explain
# ---------------------------------------------
from zennit.image import imsave, CMAPS

class ExplainWrapper(torch.nn.Module):
    """Wrap the model so that every forward pass uses explain=True."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Pass explain=True to bypass CBAM modules
        return self.model(x, explain=True)

# ---------------------------------------------
# 14. Set up Zennit composites and run LRP
# ---------------------------------------------
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import (
    EpsilonGammaBox,
    EpsilonPlus,
    EpsilonPlusFlat,
    EpsilonAlpha2Beta1Flat,
    ExcitationBackprop
)
from zennit.attribution import Gradient as ZennitGradient

# Wrap model for explanation
model = ExplainWrapper(model)

# Define canonizers required by Zennit composites
canonizers = [SequentialMergeBatchNorm()]

# Create a dictionary of different LRP rules to apply
zennit_composites = {
    'epsilon_gamma_box': EpsilonGammaBox(low=-3., high=3., canonizers=canonizers),
    'epsilon_plus': EpsilonPlus(canonizers=canonizers),
    'epsilon_plus_flat': EpsilonPlusFlat(canonizers=canonizers),
    'epsilon_alpha2_beta1_flat': EpsilonAlpha2Beta1Flat(canonizers=canonizers),
    'excitation_backprop': ExcitationBackprop(canonizers=canonizers),
}

# Predict the class for the input image
with torch.no_grad():
    out = model(input_tensor)               # forward pass with explain=True
    pred_class = out.argmax().item()        # get predicted class index

# Create one‐hot relevance at the output layer (shape [1, num_classes])
relevance_at_output = torch.eye(4)[[pred_class]]

# Helper function to visualize and save the heatmap
def visualize_absolute(relevance, method_name):
    # Convert tensor to numpy and sum across channels
    rel_np = relevance.squeeze().detach().cpu().numpy()
    rel_map = np.sum(np.abs(rel_np), axis=0)  # absolute sum over channels

    plt.figure(figsize=(4, 4))
    plt.title(method_name)
    plt.axis('off')
    # Display the relevance map as a heatmap
    plt.imshow(rel_map, cmap='hot')
    plt.colorbar()
    plt.tight_layout()
    # Save the figure to disk
    plt.savefig(f'lrp_results_new_apple_/{method_name}.png')
    plt.show()

# Apply each Zennit composite and visualize
for name, composite in zennit_composites.items():
    # Use ZennitGradient context manager to get relevance
    with ZennitGradient(model, composite) as attributor:
        _, relevance = attributor(input_tensor, relevance_at_output)
        visualize_absolute(relevance, f"Zennit_{name}")

# ---------------------------------------------
# 15. Apply Captum attribution methods
# ---------------------------------------------
from captum.attr import (
    GuidedBackprop,
    Deconvolution,
    IntegratedGradients,
    GradientShap,
    Occlusion,
    Saliency,
    NoiseTunnel
)

# Re‐instantiate model wrapper for Captum
model = VGG16_CBAM(num_classes=4)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
model = ExplainWrapper(model)

# Load and preprocess a second test image
image_path = "/kaggle/working/processed_data/test/rust/Train_1204.jpg"
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0)

# Predict the class again to set Captum target
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax().item()

# Dictionary of Captum attribution methods to run
captum_methods = {
    "Gradient": Saliency(model),
    "IntegratedGradients": IntegratedGradients(model),
    "SmoothGrad": NoiseTunnel(Saliency(model)),  # wrapped in NoiseTunnel
    "GuidedBackprop": GuidedBackprop(model),
    "Deconvolution": Deconvolution(model),
    "Occlusion": Occlusion(model),
}

# Compute and visualize attributions for each Captum method
for name, method in captum_methods.items():
    if name == "SmoothGrad":
        # SmoothGrad requires specifying the noise tunnel type
        relevance = method.attribute(input_tensor, nt_type='smoothgrad', target=pred_class)
    elif name == "Occlusion":
        # Occlusion requires sliding window parameters
        relevance = method.attribute(
            input_tensor,
            strides=(1, 3, 3),
            sliding_window_shapes=(1, 15, 15),
            target=pred_class
        )
    else:
        # Standard attribute call for other methods
        relevance = method.attribute(input_tensor, target=pred_class)

    visualize_absolute(relevance, f"Captum_{name}")
# CBAM Attention Maps
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ---------------------------------------------
# 1. Define basic building blocks for CBAM
# ---------------------------------------------

class BasicConv(nn.Module):
    """A Conv2d + (optional) BatchNorm + (optional) ReLU block."""
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # BatchNorm if requested
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        # ReLU activation if requested
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    """Flatten feature map to (batch_size, channels*height*width)."""
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_2d(tensor):
    """Compute log-sum-exp over spatial dimensions for LSE pooling."""
    flattened = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(flattened, dim=2, keepdim=True)
    return s + (flattened - s).exp().sum(dim=2, keepdim=True).log()

class ChannelGate(nn.Module):
    """Channel attention: uses multiple pooling types + small MLP."""
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super().__init__()
        # MLP to compute channel weights
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        # Apply each pooling type and accumulate MLP outputs
        for p in self.pool_types:
            if p == 'avg':
                pool = F.avg_pool2d(x, x.shape[2:])
            elif p == 'max':
                pool = F.max_pool2d(x, x.shape[2:])
            elif p == 'lp':
                pool = F.lp_pool2d(x, 2, x.shape[2:])
            elif p == 'lse':
                pool = logsumexp_2d(x)
            att_raw = self.mlp(pool)
            channel_att_sum = att_raw if channel_att_sum is None else channel_att_sum + att_raw
        # Sigmoid to get weights, reshape & scale
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    """Concatenate channel-wise max and mean pools along channel dim."""
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([max_pool, avg_pool], dim=1)

class SpatialGate(nn.Module):
    """Spatial attention: compress along channel, then conv to produce 1-channel map."""
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        # 7×7 conv to produce spatial attention map
        self.spatial = BasicConv(2, 1, kernel_size=7, padding=3, relu=False)

    def forward(self, x):
        x_comp = self.compress(x)
        x_out = self.spatial(x_comp)
        scale = torch.sigmoid(x_out)  # broadcast over channels
        return x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM): channel + (optional) spatial."""
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        if not self.no_spatial:
            x = self.SpatialGate(x)
        return x

# ---------------------------------------------
# 2. Define VGG16 + CBAM Model
# ---------------------------------------------
class VGG16_CBAM(nn.Module):
    """VGG16 architecture with CBAM modules after each block."""
    def __init__(self, num_classes=4):
        super().__init__()
        original = models.vgg16(pretrained=True)
        # Split VGG features into 5 stages
        self.stage1 = nn.Sequential(*original.features[:5])
        self.stage2 = nn.Sequential(*original.features[5:10])
        self.stage3 = nn.Sequential(*original.features[10:17])
        self.stage4 = nn.Sequential(*original.features[17:24])
        self.stage5 = nn.Sequential(*original.features[24:31])
        # CBAM modules
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)
        # Classifier head (same as VGG but adjusted for num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(inplace=False), nn.Dropout(),
            nn.Linear(4096, 4096),     nn.ReLU(inplace=False), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x, explain=False):
        # Apply each stage + CBAM (unless explain=True)
        x = self.stage1(x)
        if not explain: x = self.cbam1(x)
        x = self.stage2(x)
        if not explain: x = self.cbam2(x)
        x = self.stage3(x)
        if not explain: x = self.cbam3(x)
        x = self.stage4(x)
        if not explain: x = self.cbam4(x)
        x = self.stage5(x)
        if not explain: x = self.cbam5(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ---------------------------------------------
# 3. Prepare data directories & loaders
# ---------------------------------------------
# Paths
CSV_FILE  = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
IMAGE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7/images"
OUTPUT_DIR= "/kaggle/working/processed_data"

# Load labels
df = pd.read_csv(CSV_FILE)
df['label'] = df[['healthy','multiple_diseases','rust','scab']].idxmax(axis=1)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2,
                                     stratify=df['label'], random_state=42)

# Create folder structure
def create_directories(root, classes):
    for c in classes:
        os.makedirs(os.path.join(root, c), exist_ok=True)

train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir  = os.path.join(OUTPUT_DIR, "test")
create_directories(train_dir, df['label'].unique())
create_directories(test_dir,  df['label'].unique())

# Copy images into per-class subfolders
def organize_images(df_, root):
    for _, row in df_.iterrows():
        src = os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")
        dst = os.path.join(root, row['label'], f"{row['image_id']}.jpg")
        shutil.copy(src, dst)

organize_images(train_df, train_dir)
organize_images(test_df,  test_dir)

# Define transforms & loaders
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
train_ds = datasets.ImageFolder(train_dir, transform=transform)
test_ds  = datasets.ImageFolder(test_dir,  transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)
class_names = train_ds.classes

# ---------------------------------------------
# 4. Load model & extract CBAM feature maps
# ---------------------------------------------
# Instantiate and load pretrained weights
model = VGG16_CBAM(num_classes=4).to(device)
model.load_state_dict(torch.load("/kaggle/input/cbam-vgg16-apple-code/vgg16_cbam_grape_leaf_Apple.pth"))
model.eval()

def extract_cbam_features(model, img_tensor):
    """Return CBAM outputs from all 5 stages for a single image tensor."""
    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.stage1(x); cb1 = model.cbam1(x)
        x = model.stage2(cb1); cb2 = model.cbam2(x)
        x = model.stage3(cb2); cb3 = model.cbam3(x)
        x = model.stage4(cb3); cb4 = model.cbam4(x)
        x = model.stage5(cb4); cb5 = model.cbam5(x)
    return [cb1, cb2, cb3, cb4, cb5]

# ---------------------------------------------
# 5. Visualization helpers
# ---------------------------------------------
def show_attention_map(feature_map, title="CBAM Output"):
    """Display channel-averaged CBAM map with a colorbar."""
    fmap = feature_map.squeeze(0).mean(dim=0).cpu().numpy()
    plt.imshow(fmap, cmap='jet')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()

def denormalize_image(tensor):
    """Revert normalization to display original image."""
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (tensor.cpu()*std + mean).clamp(0,1)

def overlay_cbam_on_image(img_tensor, att_map, alpha=0.5):
    """
    Overlay a single CBAM map on the original image.
    Returns (original_image, overlay_image).
    """
    img = denormalize_image(img_tensor).permute(1,2,0).numpy()
    # Resize attention map to image size
    att = att_map.cpu().squeeze(0).mean(dim=0).numpy()
    att = cv2.resize(att, (img.shape[1], img.shape[0]))
    # Normalize & colorize
    att = (att - att.min())/(att.max()-att.min())
    heat = cv2.applyColorMap((255*att).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)/255.0
    # Blend
    overlay = (1-alpha)*img + alpha*heat
    return img, overlay

# ---------------------------------------------
# 6. Example: show CBAM maps & overlays
# ---------------------------------------------
# Pick an example image
img, lbl = test_ds[0]
maps = extract_cbam_features(model, img)

# Display maps from each stage
for i, fmap in enumerate(maps, start=1):
    show_attention_map(fmap, title=f"CBAM Stage {i}")

# Overlay for a chosen stage
orig, ovl = overlay_cbam_on_image(img, maps[3], alpha=0.6)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(orig); plt.title("Original"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(ovl);  plt.title("Overlay Stage 4"); plt.axis('off')
plt.show()

# ---------------------------------------------
# 7. Extended visualization with prediction
# ---------------------------------------------
def visualize_cbam_with_prediction(model, dataset, index=0,
                                   save_dir="./cbam_outputs",
                                   filename_prefix="cbam_pred"):
    """Plot original + CBAM overlays for all stages, include model prediction."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    img, lbl = dataset[index]
    x = img.unsqueeze(0).to(device)
    # Predict
    with torch.no_grad():
        pred = class_names[model(x).argmax(1).item()]
    true = class_names[lbl]
    # Extract maps
    maps = extract_cbam_features(model, img)
    # Plot grid: original + 5 overlays
    fig, axs = plt.subplots(1, 6, figsize=(20,4))
    orig_img = denormalize_image(x.squeeze(0)).permute(1,2,0).numpy()
    axs[0].imshow(orig_img); axs[0].set_title("Original"); axs[0].axis('off')
    for i, fmap in enumerate(maps):
        _, ov = overlay_cbam_on_image(img, fmap)
        axs[i+1].imshow(ov); axs[i+1].set_title(f"Stage {i+1}"); axs[i+1].axis('off')
    # Annotate with true vs. predicted
    color = "green" if pred == true else "red"
    plt.suptitle(f"True: {true} | Predicted: {pred}", color=color, fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    # Save figure
    path = os.path.join(save_dir, f"{filename_prefix}_{index}.png")
    plt.savefig(path)
    print(f"✅ Saved visualization to {path}")
    plt.show()

# Run extended visualization example
visualize_cbam_with_prediction(model, test_ds, index=5)
# UMAP

!pip install umap-learn


import umap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

features = []
labels = []

model.eval()
for images, targets in tqdm(test_loader):
    for img, label in zip(images, targets):
        cbam_feats = extract_cbam_features(model, img.to(device))
    
        pooled = [torch.mean(f, dim=(2, 3)).squeeze().cpu().numpy() for f in cbam_feats]
        flattened = np.concatenate(pooled)
        features.append(flattened)
        labels.append(label.item())

features = np.array(features)
labels = np.array(labels)



umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_result = umap_model.fit_transform(features)


colors = ['red', 'green', 'blue', 'orange']
class_names = ['Black Rot', 'Esca', 'Leaf Blight', 'Healthy']
plt.figure(figsize=(10, 8))
for i, color in enumerate(colors):
    idx = labels == i
    plt.scatter(umap_result[idx, 0], umap_result[idx, 1], c=color, s=10, label=class_names[i])
plt.legend()
plt.title("UMAP on CBAM Attention Features")
plt.savefig("umap_cbam.png", dpi=300, bbox_inches='tight')
plt.show()


plt.savefig("umap_CBAM_features.png", dpi=300, bbox_inches='tight')


# t-SNE

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def extract_cbam_features(model, image_tensor, device):
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        x = model.stage1(x)
        cbam1 = model.cbam1(x)

        x = model.stage2(cbam1)
        cbam2 = model.cbam2(x)

        x = model.stage3(cbam2)
        cbam3 = model.cbam3(x)

        x = model.stage4(cbam3)
        cbam4 = model.cbam4(x)

        x = model.stage5(cbam4)
        cbam5 = model.cbam5(x)

    return [cbam1, cbam2, cbam3, cbam4, cbam5]


features = []
labels = []

model.eval()
with torch.no_grad():
    for images, targets in tqdm(test_loader):  
        for img, label in zip(images, targets):
            feature_vec = extract_cbam_feature_vector(model, img, device)
            features.append(feature_vec)
            labels.append(label.item())

features = np.array(features)  
labels = np.array(labels)      

# %% [code] {"jupyter":{"outputs_hidden":false}}
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_features = tsne.fit_transform(features)


plt.figure(figsize=(10, 8))
class_names = ['Black Rot', 'Esca', 'Leaf Blight', 'Healthy']
colors = ['r', 'g', 'b', 'orange']

for i, class_name in enumerate(class_names):
    idxs = labels == i
    plt.scatter(tsne_features[idxs, 0], tsne_features[idxs, 1], label=class_name, alpha=0.6, c=colors[i])

plt.legend()
plt.title("t-SNE on CBAM Attention Features")

plt.grid(True)
plt.tight_layout()
plt.show()
# GradCAM, GradCAM++
import os
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------
# 1. Grad-CAM generation function
# ---------------------------------------------
def generate_gradcam(model, input_tensor, target_layer):
    """
    Compute Grad-CAM overlay for a single input image.
    
    Args:
        model         : The neural network (with CBAM) for inference.
        input_tensor  : A single image tensor of shape [C, H, W], unnormalized.
        target_layer  : The convolutional layer to hook for activations/gradients.
    Returns:
        overlay       : The original image blended with the Grad-CAM heatmap.
    """
    from torchvision import transforms

    # Ensure batch dimension and send to device
    img = input_tensor.unsqueeze(0).to(device)
    model.eval()

    # Dictionaries to store activations and gradients
    activation = {}
    grad = {}

    # Forward hook: save target layer output
    def save_activation(module, inp, out):
        activation["value"] = out

    # Backward hook: save gradients wrt target layer
    def save_grad(module, grad_in, grad_out):
        grad["value"] = grad_out[0]

    # Register hooks
    hook_a = target_layer.register_forward_hook(save_activation)
    hook_g = target_layer.register_backward_hook(save_grad)

    # Forward pass
    output = model(img)
    pred_class = output.argmax(dim=1)
    # Compute gradient of score for predicted class
    score = output[0, pred_class]
    score.backward()

    # Retrieve stored activation maps and gradients
    activations = activation["value"]   # shape [1, C, H, W]
    gradients = grad["value"]           # same shape

    # Global average pool gradients over spatial dims
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # shape [1, C, 1, 1]
    # Weight activations by pooled gradients
    weighted_acts = activations * weights
    # Sum over channels to get single heatmap
    heatmap = weighted_acts.sum(dim=1).squeeze(0)
    # Apply ReLU and normalize to [0,1]
    heatmap = F.relu(heatmap)
    heatmap /= heatmap.max()

    # Convert to numpy and resize to original image size
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    # Colorize heatmap using OpenCV's JET colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Denormalize original image for visualization
    original = denormalize_image(img.squeeze(0)).permute(1, 2, 0).cpu().numpy()
    original_uint8 = (original * 255).astype(np.uint8)
    # Blend colored heatmap with original
    overlay = cv2.addWeighted(original_uint8, 0.6, heatmap_color, 0.4, 0)

    # Remove hooks to avoid side‐effects
    hook_a.remove()
    hook_g.remove()

    return overlay

# ---------------------------------------------
# 2. Compare CBAM vs. Grad-CAM side by side
# ---------------------------------------------
def compare_cbam_gradcam(model, dataset, index=0, target_layer=None,
                         save_dir="./cbam_vs_gradcam", prefix="compare"):
    """
    Plot original image, Grad-CAM overlay, and CBAM overlays for all stages.
    Saves the combined figure to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load image and true label
    img, label = dataset[index]
    img = img.to(device)
    true_label = class_names[label]

    # Model prediction
    model.eval()
    with torch.no_grad():
        out = model(img.unsqueeze(0))
        pred_label = class_names[out.argmax(1).item()]

    # Generate Grad-CAM overlay
    gradcam_overlay = generate_gradcam(model, img, target_layer)

    # Extract CBAM feature maps and overlays
    cbam_maps = extract_cbam_features(model, img)
    
    # Prepare plot: Original + Grad-CAM + CBAM Stages 1–5
    fig, axs = plt.subplots(1, 7, figsize=(22, 4))

    # 1) Original image
    orig = denormalize_image(img).permute(1, 2, 0).cpu().numpy()
    axs[0].imshow(orig)
    axs[0].set_title("Original")
    axs[0].axis("off")

    # 2) Grad-CAM overlay
    axs[1].imshow(gradcam_overlay)
    axs[1].set_title("Grad-CAM")
    axs[1].axis("off")

    # 3–7) CBAM overlays for each stage
    for i in range(5):
        _, overlay = overlay_cbam_on_image(img, cbam_maps[i])
        axs[i+2].imshow(overlay)
        axs[i+2].set_title(f"CBAM {i+1}")
        axs[i+2].axis("off")

    # Overall title with true vs. predicted labels
    color = "green" if pred_label == true_label else "red"
    plt.suptitle(f"True: {true_label} | Pred: {pred_label}", fontsize=14, color=color)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and display
    filename = f"{prefix}_idx{index}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.show()
    print(f"✅ Saved comparison to: {path}")

# Choose a convolutional layer from stage5
target_layer = model.stage5[-1]
compare_cbam_gradcam(model, test_dataset, index=138,
                     target_layer=target_layer,
                     save_dir="./cbam_vs_gradcam",
                     prefix="grape_comparison")

# ---------------------------------------------
# 3. Install & import pytorch-grad-cam
# ---------------------------------------------
!pip install grad-cam --quiet

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------------------------------
# 4. Generate Grad-CAM++ overlay (using library)
# ---------------------------------------------
# Ensure model on correct device
model.to(device)
model.eval()

# Instantiate GradCAM++ with our model and target layer
cam_pp = GradCAMPlusPlus(model=model, target_layers=[target_layer])

# Select the same test image
idx = 138
img_tensor, lbl = test_dataset[idx]
input_tensor = img_tensor.unsqueeze(0).to(device)

# Compute grayscale CAM for true label
targets = [ClassifierOutputTarget(lbl)]
grayscale_cam_pp = cam_pp(input_tensor=input_tensor, targets=targets)[0]

# Prepare normalized RGB image for overlay
rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# Overlay CAM onto the image
overlay_pp = show_cam_on_image(rgb_img, grayscale_cam_pp, use_rgb=True)

# Display result
plt.imshow(overlay_pp)
plt.title(f"Grad-CAM++ | True: {class_names[lbl]}")
plt.axis("off")
plt.show()

# ---------------------------------------------
# 5. Compare all methods: Grad-CAM, Grad-CAM++, CBAM
# ---------------------------------------------
def compare_all_visuals(model, dataset, index, class_names, save_dir="final_comparison_outputs"):
    """
    Create a figure with:
    Original | Grad-CAM | Grad-CAM++ | CBAM Stages 1–5
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Load image and true label
    img_tensor, lbl = dataset[index]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    true_label = class_names[lbl]

    # Predict to get predicted label
    with torch.no_grad():
        out = model(input_tensor)
        pred_label = class_names[out.argmax(1).item()]

    # Prepare RGB image for overlays
    rgb_img = denormalize_image(img_tensor).permute(1, 2, 0).cpu().numpy()

    # 1) Grad-CAM (library)
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    gc_map = gradcam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(lbl)])[0]
    overlay_gc = show_cam_on_image(rgb_img, gc_map, use_rgb=True)

    # 2) Grad-CAM++ (library)
    gradcam_pp = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    gcpp_map = gradcam_pp(input_tensor=input_tensor, targets=[ClassifierOutputTarget(lbl)])[0]
    overlay_gcpp = show_cam_on_image(rgb_img, gcpp_map, use_rgb=True)

    # 3) CBAM overlays
    cbam_maps = extract_cbam_features(model, img_tensor.to(device))
    cbam_overlays = [overlay_cbam_on_image(img_tensor.to(device), m)[1] for m in cbam_maps]

    # Create subplots: total 8 images
    fig, axes = plt.subplots(1, 8, figsize=(40, 5))
    titles = ["Original", "Grad-CAM", "Grad-CAM++"] + [f"CBAM {i}" for i in range(1, 6)]
    images = [rgb_img, overlay_gc, overlay_gcpp] + cbam_overlays

    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=14)
        ax.axis("off")

    # Super‐title with prediction status
    color = "green" if pred_label == true_label else "red"
    plt.suptitle(f"True: {true_label} | Predicted: {pred_label}", fontsize=18, color=color)
    plt.tight_layout()
    
    # Save and show
    out_path = os.path.join(save_dir, f"visual_compare_idx{index}.png")
    plt.savefig(out_path)
    plt.show()
    print(f"✅ Saved final comparison to: {out_path}")

# Run final comparison for image index 138
compare_all_visuals(model, test_dataset, index=138, class_names=class_names)
