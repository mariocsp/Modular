"""
Contains functions for training and testing a PyTorch model.
"""


import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryF1Score, BinaryPrecision
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional import accuracy

def train_loop(model:torch.nn.Module,
               train_dataloader:torch.utils.data.DataLoader,
               loss_fn: torch.nn.modules,
               optimizer: torch.optim.Optimizer,
               device:torch.device,
               scheduler:torch.optim.lr_scheduler=None,
               class_name = []):

    """
    Trains a PyTorch model for a single epoch.
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        scheduler: shceduler use to decay learning rate

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy, train_f1,
    train_precission). For example:

    ({loss: 3.07982
     akurasi: 0.50473
     F1 :0.32210
     presisi:0.41534}, (3.07982), (0.50473), (0.32210), (0.41534)  )
  """

    loss_train = 0
    acc_train = 0
    f1_train = 0
    prec_train = 0
    len_batch = len(train_dataloader)

    model.to(device)
    model.train()

    for batch, (X,y) in enumerate(train_dataloader):

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logit = model(X)
        pred = torch.softmax(logit, dim=1).argmax(dim=1)

        loss = loss_fn(logit,y)
        loss_train += loss.item()
        loss.backward()
        optimizer.step()

        acc = accuracy(pred,y,task='multiclass', num_classes = len(class_name)).item()
        metric_f1 = MulticlassF1Score(num_classes = len(class_name)).to(device)
        metric_prec = MulticlassPrecision(num_classes = len(class_name)).to(device)
        f1 = metric_f1(pred,y).item()
        prec = metric_prec(pred,y).item()

        acc_train += acc
        f1_train += f1
        prec_train += prec

    if not scheduler == None:
        scheduler.step()
        print()

    loss_train = loss_train/len_batch
    acc_train = acc_train/len_batch
    f1_train = f1_train/len_batch
    prec_train = prec_train/len_batch


    return loss_train,acc_train,f1_train, prec_train

def model_metrics(model:torch.nn.Module,
                  test_dataloader:torch.utils.data.DataLoader,
                  loss_fn: torch.nn.modules,
                  device: torch.device,
                  class_name:list = []):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy, test__F1, test_precision).
        For example:

        (0.0223, 0.8985, 0.9985, 0983)
        """

    model.to(device)
    model.eval()

    loss_test = 0
    acc_test = 0
    f1_test = 0
    prec_test = 0
    len_batch = len(test_dataloader)

    with torch.inference_mode():

        for batch, (X,y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)

            logit = model(X)
            pred = torch.softmax(logit, dim=1).argmax(dim=1)
            loss = loss_fn(logit,y)

            loss_test += loss.item()

            acc = accuracy(pred,y,task='multiclass', num_classes = len(class_name)).item()
            metric_f1 = MulticlassF1Score(num_classes = len(class_name)).to(device)
            metric_prec = MulticlassPrecision(num_classes = len(class_name)).to(device)
            f1 = metric_f1(pred,y).item()
            prec = metric_prec(pred,y).item()

            acc_test += acc
            f1_test += f1
            prec_test += prec

        loss_test = loss_test/len_batch
        acc_test = acc_test/len_batch
        f1_test = f1_test/len_batch
        prec_test = prec_test/len_batch


    return loss_test, acc_test, f1_test, prec_test

def training(model:torch.nn.Module,
             train_dataloader: torch.utils.data.DataLoader,
             test_dataloader:torch.utils.data.DataLoader,
             loss_fn:torch.nn.Module,
             optimizer:torch.optim.Optimizer,
             epochs:int,
             device:torch.device,
             scheduler = None,
             class_name:list = []):

    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        scheduler: Scheduler used to decay lr

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {'loss_train':[],
                 'acc_train':[],
                 'F1_train':[],
                 'prec_train':[],
                 'loss_test':[],
                 'acc_test':[],
                 'F1_test':[],
                'prec_test':[]}

    and the final log of training and test epoch
    """

    model_name = model.__class__.__name__

    dict_info = {
                'loss_train':[],
                'acc_train':[],
                'F1_train':[],
                'prec_train':[],
                'loss_test':[],
                'acc_test':[],
                'F1_test':[],
                'prec_test':[]}

    for epoch in range(epochs+1):
        l,a,F1,p = train_loop(model,
                                train_dataloader,
                                loss_fn,
                                optimizer,
                                device,
                                scheduler,
                                class_name = class_name)

        dict_info['loss_train'].append(l)
        dict_info['acc_train'].append(a)
        dict_info['F1_train'].append(F1)
        dict_info['prec_train'].append(p)

        log_train = {'model': model_name,
                    'loss_train':l,
                    'acc_train':a,
                    'F1_train':F1,
                    'prec_train':p}

        info_str = f"epoch {epoch:}| loss:{l:.5G} | acc:{a:.5f} |\
F1: {F1:.5f} | prec: {p:5f}|"
        print('Train'.center(len(info_str),'_'))
        print(info_str,'\n')

        bagi = 10 if epochs >= 10 else 1

        if epoch%(epochs//bagi) == 0:
            lt,at,F1t,pt = model_metrics(model,test_dataloader,loss_fn,device,class_name)

            dict_info['loss_test'].append(lt)
            dict_info['acc_test'].append(at)
            dict_info['F1_test'].append(F1t)
            dict_info['prec_test'].append(pt)

            log_test = {'model': model_name,
                        'loss_test':lt,
                        'acc_test':at,
                        'F1_test':F1t,
                        'prec_test':pt}

            info_str = f"epoch {epoch}| loss:{lt:.5G} | acc:{at:.5f} |\
F1: {F1t:.5f} | prec: {pt:.5f}|"

            print(f"Test".center(len(info_str),'_'))
            print(info_str,"\n")

    return dict_info, log_train, log_test
