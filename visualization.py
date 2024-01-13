import numpy as np
import pandas as pd

def quick_loss_plot(train_losses, val_losses, model_type = 4, loss_type = "MSE Loss", sparse_n = 0):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    model_dict = {
        1: "Linear",
        2: "MLP",
        3: "CNN",
        4: "LSTM"
    }

    data_label_list = [(train_losses, val_losses, model_dict[model_type])]

    for i,(train_data,val_data,label) in enumerate(data_label_list):    
        plt.plot(train_data,linestyle='--',color=f"C{i}", label=f"{label} Train")
        plt.plot(val_data,color=f"C{i}", label=f"{label} Val",linewidth=3.0)

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.show()