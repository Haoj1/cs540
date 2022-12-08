import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.EMNIST('./data_alpha', split="bymerge", train=True, download=True,
                               transform=custom_transform)
    test_set = datasets.EMNIST('./data_alpha', split="bymerge", train=False,
                              transform=custom_transform)
    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size=50)
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 47)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        episode_loss = 0
        correct_count = 0
        total_count = 0
        for data in train_loader:
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            episode_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_count += len(outputs)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_count += 1
        accuracy = 100 * float(correct_count) / total_count
        episode_loss = episode_loss / len(train_loader)
        print("Train Epoch: {:d} Accuracy: {:d}/{:d}({:.2f}%) Loss: {:.3f}".format(epoch, correct_count,
                                                                                   total_count, accuracy, episode_loss))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    loss = 0
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_count += len(outputs)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_count += 1
    accuracy = 100 * float(correct_count) / total_count
    episode_loss = loss / len(test_loader)
    if show_loss:
        print("Average loss: {:.4f}".format(episode_loss))
    print("Accuracy: {:.2f}%".format(accuracy))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    model.eval()
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    with torch.no_grad():
        inputs = test_images[index]
        output = model(inputs)
        prob = F.softmax(output, dim=1)
    values, indices = torch.sort(prob, descending=True)
    values = torch.squeeze(values)
    indices = torch.squeeze(indices)
    values = values * 100
    for i in range(3):
        print("{:s}: {:.2f}%".format(class_names[indices[i]], values[i]))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded. 
    '''
    train_loader = get_data_loader(True)
    # print(len(train_loader))
    # test_loader = get_data_loader(False)
    # custom_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # test_set = datasets.MNIST('./data', train=False,
    #                           transform=custom_transform)
    model = build_model()
    # for data in train_loader:
    #     inputs, labels = data
    #     print(labels)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, T=5)
    # evaluate_model(model, test_loader, criterion, show_loss=True)
    # # torch.save(model, "number_model.pt")
    # pred_set = []
    # for i, (images, label) in enumerate(test_loader.dataset):
    #     pred_set.append(images)
    # pred_set = torch.stack(pred_set)
    # predict_label(model, pred_set, 1)
