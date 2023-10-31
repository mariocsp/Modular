"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import pathlib
from pathlib import Path
import requests
import zipfile
import matplotlib.pyplot as plt
import numpy as np



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)



def downloading_zip_data(url:str,
                         data_path:pathlib.PosixPath,
                         zip_name:str):
    
    """
    Function to download zip data from remote

     Args:
        url: Url where remote data is stored
        data_path: Path where you want to put the data 
        zip_name: Remote Zipfile name

    Return:
       Image_path: pathlib.PosixPath

    Example Usage:

        url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        data_path = Path('data/')
        zip_name = "pizza_steak_sushi"
        downloading_zip_data(url,data_path, zip_name)
    
    """

    image_path = data_path/zip_name

    if image_path.is_dir():
        print(f"[INFO] {image_path} already exist")

    else:
        print(f'making {image_path} dir ')
        image_path.mkdir(parents = True, exist_ok = True)

        with open(data_path / f'{zip_name}.zip','wb') as files:
            req = requests.get(url)
            print('[INFO] Downloading the data')
            files.write(req.content)

        with zipfile.ZipFile(data_path / f'{zip_name}.zip','r') as files:
            print('[INFO] unzipping zip')
            files.extractall(image_path)
    
    return image_path


def manual_tensorboard(dict_info:dict):
    fig, axs = plt.subplots(3,3)
    fig.set_figheight(12)
    fig.set_figwidth(12)

    for j,i in enumerate(dict_info):
        value = dict_info[i]
        x = np.linspace(0,100,len(value))
        fig.add_subplot(3, 3, j+ 1 )
        plt.plot(x,value)

        plt.title(i)
        plt.xticks([])
        plt.yticks([])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig.tight_layout(pad=1)
    fig.show()

    return True
    
        