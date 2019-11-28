import math

import torch
import torch.utils
import torch.nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import sklearn.metrics
import pdb

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NOISE_RATIO = 0.2
SPARSITY_COEF = 1e-2
MODE = 'train'  # eval, train
EPSILON = 1e-20
Z_SIZE = 32

DEVICE = 'cuda'
if not torch.cuda.is_available():
    DEVICE = 'cpu'

path_tmp = '/tmp'
if not os.path.exists(path_tmp):
    os.makedirs(path_tmp)

datasets = []
data_loaders = []

for is_train in [True, False]:
    dataset = torchvision.datasets.FashionMNIST(
        root=path_tmp,
        download=True,
        train=is_train,
        transform=torchvision.transforms.ToTensor()
    )
    datasets.append(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=is_train
    )
    data_loaders.append(data_loader)


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        encoder_pretrained = torchvision.models.densenet121(pretrained=True)

        # preserve trained weights
        features_pretrained = next(iter(encoder_pretrained.children()))
        conv0 = features_pretrained.conv0
        weights_conv0 = conv0.weight.data

        conv0_new = torch.nn.Conv2d(
            in_channels=1,
            out_channels=conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=conv0.stride,
            padding=conv0.padding,
            bias=False
        )
        conv0_new.weight.data[:] = torch.unsqueeze(weights_conv0[:, 1, :, :], dim=1)

        self.encoder = torch.nn.Sequential()
        z_num_features = 0
        for name, module in features_pretrained.named_children():
            if name == 'conv0':
                module = conv0_new
            elif name == 'norm5':
                z_num_features = module.num_features
            self.encoder.add_module(name, module)
        self.encoder.add_module('avg_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))

        # (B, C, W, H) = > # (B, C * W * H) = > # (B, C * 1 * 1) => (B, C, 1)
        self.encoder.add_module('reshape_z', Reshape((-1, z_num_features, 1)))

        self.encoder.add_module('conv_z', torch.nn.Conv1d(
            in_channels=z_num_features,
            out_channels=Z_SIZE,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        ))
        self.encoder.add_module('reshape_z_out', Reshape((-1, Z_SIZE)))
        self.encoder.add_module('tanh_z', torch.nn.Tanh())

        self.decoder = torch.nn.Sequential()
        self.decoder.add_module('reshape_z', Reshape((-1, Z_SIZE, 1, 1)))

        # 1 2 4 8 16
        in_num_features = Z_SIZE
        for idx_layer in range(8):
            out_num_features = int(math.ceil(in_num_features / 2.5))  # 512, 256, ... 1
            stride = 1
            padding = 0
            kernel_size = 3
            if idx_layer % 2 == 0:
                stride = 2
                padding = 1

            if idx_layer == 7:
                padding = 3
                kernel_size = 6

            self.decoder.add_module(
                f'deconv{idx_layer}',  # 1 => 2
                torch.nn.ConvTranspose2d(
                    in_channels=in_num_features,
                    out_channels=out_num_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            self.decoder.add_module(
                f'relu{idx_layer}',  # 1 => 2
                torch.nn.ReLU()
            )
            self.decoder.add_module(
                f'relu{idx_layer}',  # 1 => 2
                torch.nn.BatchNorm2d(num_features=out_num_features)
            )
            in_num_features = out_num_features

        self.decoder.add_module(
            f'final_sigm',  # 0..1
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        # TODO interesting stuff with z
        y = self.decoder(z)
        return y, z


model = Model()

if MODE == 'eval':
    model.load_state_dict(torch.load('sae_best.pt'))

model = model.to(DEVICE)

if MODE == 'train':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    print('training started')
    best_accuracy = float('-Inf')

    for epoch in range(1, NUM_EPOCHS + 1):

        for idx_loader, data_loader in enumerate(data_loaders):
            losses = []
            accuracy = []

            if idx_loader == 1:  # test
                model = model.eval()
            else:
                model = model.train()

            iter_count = 100

            for x, y_idx in data_loader:
                x = x.to(DEVICE)

                # Noisy augmentation
                # noise = torch.zeros_like(x).data.uniform_()
                noise = torch.sigmoid(torch.zeros_like(x).data.normal_())
                noisy_x = NOISE_RATIO * noise + (1.0 - NOISE_RATIO) * x

                # plt.imshow(x[0][0].data.numpy())
                # plt.show()
                # plt.imshow(noise[0][0].data.numpy())
                # plt.show()
                # plt.imshow(noisy_x[0][0].data.numpy())
                # plt.show()

                # if idx_loader == 0 and epoch > 1:
                #    pdb.set_trace()

                y, z = model.forward(noisy_x)

                loss_l1 = 0
                for name, param in model.named_parameters():
                    if 'conv' in name and 'weight' in name:
                        loss_l1 += torch.mean(torch.abs(param))

                # reconstruction loss
                # pdb.set_trace()
                loss_sparse = SPARSITY_COEF * loss_l1

                y_p = y.view(-1, y.size(1) * y.size(2))
                x_p = x.view(-1, x.size(1) * x.size(2))
                loss_bce = -torch.mean(x_p * torch.log(y_p) + (1.0 - x_p) * torch.log(1.0 - y_p))

                loss = loss_bce + loss_sparse
                losses.append(loss.item())

                np_x = x.to('cpu').detach().data.numpy()
                np_y = y.to('cpu').detach().data.numpy()

                np_x_1d = np.reshape(np_x, (np_x.shape[0], np.prod(np_x.shape[1:])))
                np_y_1d = np.reshape(np_y, np_x_1d.shape)

                r2_score = sklearn.metrics.r2_score(np_x_1d, np_y_1d)
                accuracy.append(r2_score)

                # -1 .. 1

                if idx_loader == 0:  # train stage
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # iter_count -= 1
                # if iter_count < 0:
                #    break
                # break

            stage = 'train'
            if idx_loader == 1:
                stage = 'test'

            avg_accuracy = np.average(accuracy)
            print(f'epoch: {epoch} stage: {stage} loss: {np.average(losses)} r2: {avg_accuracy}')

            if idx_loader == 1:  # test
                if best_accuracy < avg_accuracy:
                    best_accuracy = avg_accuracy
                    torch.save(model.to('cpu').state_dict(), 'sae_best.pt')
                    model = model.to(DEVICE)
                    print('best so far, saved')

                n_col = 2
                n_samples = 4
                _, axs = plt.subplots(int(math.ceil(n_samples * 2 / n_col)), n_col, figsize=(12, 12))
                axs = axs.flatten()

                for x_idx in range(n_samples):
                    axs[x_idx * 2].imshow(np_x[x_idx][0])
                    axs[x_idx * 2 + 1].imshow(np_y[x_idx][0])
                plt.show()

if MODE == 'eval' or 'play':

    np_img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

    plt.imshow(np_img)
    plt.show()

    x0 = next(iter(data_loaders[0]))[0][0]  # form train dataset first sample
    size_w = x0.size(1)  # 28
    size_h = x0.size(2)  # 28

    torch_img = torch.FloatTensor(np_img)  # (W,H)
    torch_img = torch.unsqueeze(torch_img, dim=0)  # (B, C, W, H) (1, 1, W, H)
    torch_img = torch.unsqueeze(torch_img, dim=0)
    torch_img = F.adaptive_avg_pool2d(torch_img, output_size=(size_w, size_h))

    model = model.eval()

    with torch.no_grad():
        torch_img = torch_img.to(DEVICE)
        z_img = model.encoder.forward(torch_img)

        z_img_batch = z_img.repeat(BATCH_SIZE, 1, 1, 1)

        pairs_all = []

        x_idx = 0
        for x, y_idx in data_loaders[1]:
            if x.size(0) < BATCH_SIZE:
                break
            x = x.to(DEVICE)

            z_x = model.encoder.forward(x)

            z_img_batch = z_img_batch.view((-1, z_img_batch.size(1)))
            z_x = z_x.view((-1, z_img_batch.size(1)))

            distances = F.pairwise_distance(z_x, z_img_batch)

            distances = distances.to('cpu').data.numpy().tolist()

            pairs_batch = list(zip(distances, np.arange(x_idx, x_idx + BATCH_SIZE).tolist()))
            pairs_all += pairs_batch
            x_idx += BATCH_SIZE
            # break

        pairs_all = sorted(pairs_all, key=lambda it: it[0], reverse=False)
        x_most_simillar_idxes = np.array(pairs_all)[:16, 1]

        x_idx = 0
        for x, y_idx in data_loaders[1]:
            for idx in np.arange(x_idx, x_idx + BATCH_SIZE):
                if int(idx) in x_most_simillar_idxes:
                    np_x = x[idx % BATCH_SIZE][0].data.numpy()
                    plt.imshow(np_x)
                    plt.show()
            x_idx += BATCH_SIZE

if MODE == 'play':
    hash_table_z = []

    model = model.eval()
    with torch.nn_grad():
        for x, y_idx in data_loaders[1]:  # test data loader
            x = x.to(DEVICE)
            z_x = model.encoder.forward(x)

            z_x = z_x.to('cpu')
            z_x = z_x.view(-1, z_img_batch.size(1))  # 1D vector
            hash_table_z.append(z_x)

if MODE == 'play':

    z = np.zeros(32).eval()
    z[0] = 0.0  # param (type: "slider", min:-1, max: 1, step: 0)
    z[1] = 0.0  # param (type: "slider", min:-1, max: 1, step: 0)
    z[2] = 0.0  # param (type: "slider", min:-1, max: 1, step: 0)
    z[3] = 0.0  # param (type: "slider", min:-1, max: 1, step: 0)
    z[4] = 0.0  # param (type: "slider", min:-1, max: 1, step: 0)

    z = torch.FloatTensor(z)
    hash_table_z = torch.FloatTensor(hash_table_z)
    z_copy = torch.repeat(hash_table_z.size(0), 1)
    distances = F.pairwise_distance(z, hash_table_z)

    sample_idx = 0
    for x, y_idx in data_loaders[1]:  # test data loader
        sample_idx += x.size[0]
        if closest_sample_idx < sample_idx:
            batch_idx
            sample_idx - BATCH_SIZE + closest_sample_idx
            closest_x = x[batch_idx]
            break

    plt.imshow(closest_x.data.numpy())
    plt.show()











