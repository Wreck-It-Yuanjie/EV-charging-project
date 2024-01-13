import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

### test the model with testing data
def test_model(model, test_dl, loss_function, optimizer):
    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        test_losses = []
        batch_sizes = []

        ## Loop through the validation dataloader
        for x_test, y_test in test_dl:
            # print("test data x: ", x_test)
            # print("test data y: ", y_test)

            ## Calculate the loss WITHOUT BACK PROPOGATION
            test_loss, batch_size = model_loss(model, loss_function, x_test, y_test)

            ## Append validation loss and batch size
            test_losses.append(test_loss)
            batch_sizes.append(batch_size)

        ## Calculate the average losses over all batches
        test_loss = np.sum(np.multiply(test_losses, batch_sizes)) / np.sum(batch_sizes)

        return test_loss
    print('Test Loss: {:.4f}'.format(test_loss.item()))