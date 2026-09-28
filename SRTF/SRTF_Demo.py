import torch
from utils.Mydataset import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as scio
from utils.Funs import spilt_train, kappa_statistic
from utils.network import MyNetwork
from sklearn.decomposition import PCA
import os
from tqdm import tqdm


def experiment(data_all, data_gt, class_num, select_rate, win_size, batch_size=32, learning_rate=1e-4, epochs=100,
               loss_balance=0.001, TSNE=False, embedding_bands=128, num_heads=2, block_nums=1, drop_rate=0.3):
    """
    01-Data Preprocessing
    """
    [n1, n2, bands] = np.shape(data_all)
    mask_indexs = np.where(data_gt.flatten() > 0)[0]

    data_all = torch.from_numpy(data_all.astype('float32'))
    data_all = torch.permute(data_all, (2, 0, 1))
    data_gt = torch.from_numpy(data_gt.astype('float32')).squeeze()

    """
    02-Random Sampling
    """
    train_label, del_train_label, train_index = spilt_train(data_gt.clone(), rate=select_rate, need_index=True)
    traning_data = MyDataset(data_all, train_label, win_size, transforms=True)
    test_data = MyDataset(data_all, data_gt + 1, win_size )

    train_loader = DataLoader(dataset=traning_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size * 20, shuffle=False, num_workers=0)

    """
    03-Loading the Classification Network
    """
    net = MyNetwork(win_size=win_size, bands=bands, embedding_bands=embedding_bands, num_heads=num_heads,
                    block_nums=block_nums,
                    class_nums=class_num, drop_rate=0.3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    """
    04-Training process
    """
    weights_save_name = './weight/data_no'
    min_lr = learning_rate * 0.2
    net.train()

    with tqdm(range(epochs)) as train_bar:
        for epoch in train_bar:
            correct, total, total_loss = 0., 0., 0.
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                optimizer.zero_grad()
                labels = torch.as_tensor(labels - 1, dtype=torch.long)
                Y_out, res_loss = net(images.to(device))

                # loss
                loss_out = loss_function(Y_out, labels.long().to(device))
                loss_all = loss_out + loss_balance * res_loss
                correct += torch.sum(torch.argmax(Y_out, dim=1) == labels.long().to(device)).item()
                total += len(labels)
                accuracy = correct / total
                total_loss += loss_all

                # update
                loss_all.backward()
                optimizer.step()
            train_bar.set_postfix(loss=total_loss.item(), acc=accuracy * 100)

            if learning_rate > min_lr:
                learning_rate = learning_rate * 0.001 * torch.randint(low=990, high=1000, size=(1, 1)).item()
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    """
    05-Classification process
    """
    net.eval()
    if os.path.exists(weights_save_name):
        net.load_state_dict(torch.load(weights_save_name))
    else:
        print('Warning: Saved weight data not read...')

    with torch.no_grad():
        pred_label_list = []
        item = 0
        with tqdm(test_loader) as test_bar:
            for images, labels in test_bar:
                [Y_res_list, sy_loss_list] = net(images.to(device))
                if TSNE and item == 0:
                    res_fes = Y_res_list
                    item = 1
                elif TSNE and item == 1:
                    res_fes = torch.cat((res_fes, Y_res_list), dim=0)

                pred_labels = torch.argmax(Y_res_list, dim=1).tolist()
                pred_label_list.append(pred_labels)

    last_epoch = np.array(pred_label_list[-1]).flatten()
    pred_labels = np.array(pred_label_list[:-1]).flatten()
    pred_label_list = np.concatenate((pred_labels, last_epoch)) + 1

    data_gt[train_index[:, 0], train_index[:, 1]] = 0
    Ground_Truth = data_gt.flatten().detach().cpu().numpy()
    no_back_index = np.where(Ground_Truth > 0)
    Ground_Truth = Ground_Truth[no_back_index]
    Pred_labels = pred_label_list[no_back_index]
    result = kappa_statistic(Ground_Truth, Pred_labels)

    return result

def read_HSI():
    data_file_name = './data/Indian.mat'
    data_img = scio.loadmat(data_file_name)['Indian']
    data_gt  = scio.loadmat(data_file_name)['Indian_gt']
    class_num = data_gt.max()
    return data_img, data_gt, class_num

def pca_HSI(data):
    [n1, n2, bands] = np.shape(data)
    pca_fun = PCA(n_components=pca_bands, whiten=True)
    data = pca_fun.fit_transform(data.reshape(n1 * n2, bands))
    data = data.reshape(n1, n2, pca_bands)
    return data


if __name__ == '__main__':
    device = torch.device('cuda:0')

    """
    Parameter Settings
    """
    select_rate = 50
    batch_size = 32
    learning_rate = 1e-4
    epochs = 300
    win_size = 31
    pca_bands = 30
    loss_balance = torch.Tensor([0.001]).to(device)

    [data_all, data_gt, class_num] = read_HSI()
    bands = np.shape(data_all)[-1]

    # Dimensionality reduction
    if pca_bands > 0:
        data_all = pca_HSI(data_all)

    # Main program
    result = experiment(data_all=data_all, data_gt=data_gt, class_num=class_num,
                        select_rate=select_rate, win_size=win_size, learning_rate=learning_rate,
                        epochs=epochs, loss_balance=loss_balance, TSNE=False, embedding_bands=128,
                        num_heads=4, block_nums=3, drop_rate=0.3)
    print(result)


