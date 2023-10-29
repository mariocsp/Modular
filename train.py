"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torch import nn
import data_setup, engine, model_builder, utils
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_dir", help="train directory")
parser.add_argument("test_dir", help="train directory")

parser.add_argument('-NE',"--NUM_EPOCHS", help="Number of epoch",
                    type = int, default = 100)
parser.add_argument("-LR","--LEARNING_RATE", help="learning rate", 
                    type = float,  default = 0.01)
parser.add_argument("-BS","--BATCH_SIZE", help="batch_size", 
                    type = int,  default = 32)
parser.add_argument("-MN","--MODEL_NAME", help="batch_size", 
                    type = str,  default = "Tiny_VGG")

args = parser.parse_args()


# Setup hyperparameters
train_dir = args.train_dir
test_dir =  args.test_dir

NUM_EPOCHS = args.NUM_EPOCHS
BATCH_SIZE = args.BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
MODEL_NAME = args.MODEL_NAME

HIDDEN_UNITS = 5
IMAGE_SIZE = 128


data_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                         transforms.ToTensor()])

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.Model_base(input_shape = 3,
                            hidden_units = HIDDEN_UNITS,
                            output_shape = len(class_names)).to(device)

# Set loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(),lr = LEARNING_RATE)#,weight_decay = 0.0001
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                               start_factor=0.3333333333333333,
                                               end_factor=1.0, total_iters=50,
                                               last_epoch= -1, verbose=False)

print(type(loss_fn))
print(type(optimizer))
print(type(scheduler))
print('\n')

dict_info, log_train, log_test= engine.training(model = model,
                                        train_dataloader = train_dataloader,
                                        test_dataloader = test_dataloader,
                                        loss_fn = loss_fn,
                                        optimizer = optimizer,
                                        epochs = NUM_EPOCHS,
                                        device = device,
                                        scheduler = scheduler,
                                        class_name = class_names)


# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name=f"{MODEL_NAME}.pth")