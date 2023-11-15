"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/10 下午4:18
@Email: 2909981736@qq.com
"""
import torch
import tqdm
import utils
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.DCGAN import Discriminator, Generator
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CDCGAN:
    def __init__(self, epochs=50, batch_size=256):
        self.epochs = epochs
        self.epoch = 0
        self.batch_size = batch_size
        self.first = True

    def train_step(self, gen, dis, train_loader, g_loss_func, d_loss_func, g_optim, d_optim):
        gen.train()
        dis.train()

        g_loss_val = 0
        d_loss_val = 0
        noise = torch.randn(train_loader.batch_size, 100, 1, 1).to(device)
        num_of_batch = len(train_loader)
        desc = f'Epoch {self.epoch + 1}/{self.epochs}, Step'
        data_loader_itr = iter(train_loader)
        for _ in tqdm.trange(num_of_batch, desc=desc, total=num_of_batch):
            batch_img, batch_label = next(data_loader_itr)
            real_img = batch_img.to(device)

            conditional_label = batch_label.view([batch_img.size(0), 2, 1, 1]).to(device)

            real_label_img = torch.cat([real_img, torch.ones_like(real_img) * conditional_label], dim=1).to(device)
            noise_label_img = torch.cat([noise[:batch_img.size(0)], conditional_label], dim=1).to(device)

            fake_img = gen(noise_label_img)
            fake_label_img = torch.cat([fake_img, torch.ones_like(fake_img) * conditional_label], dim=1)

            # train D
            d_optim.zero_grad()
            real_logits = dis(real_label_img)
            fake_logits = dis(fake_label_img)
            d_loss = d_loss_func(real_logits, fake_logits)
            d_loss_val += d_loss.cpu().item()
            d_loss.backward(retain_graph=True)
            d_optim.step()

            # train G
            g_optim.zero_grad()
            fake_label_img = torch.cat([fake_img, torch.ones_like(fake_img) * conditional_label], dim=1)
            fake_logits = dis(fake_label_img)
            g_loss = g_loss_func(fake_logits)
            g_loss_val += g_loss.cpu().item()
            g_loss.backward(retain_graph=False)
            g_optim.step()

        return g_loss_val, d_loss_val

    def test_step(self, gen, noise, test_loader):
        gen.eval()
        with torch.no_grad():
            data_loader_itr = iter(test_loader)
            test_data, test_label = next(data_loader_itr)
            if self.first:
                plt.figure(figsize=(8, 8))
                for i in range(8):
                    for j in range(8):
                        plt.subplot(8, 8, i * 8 + j + 1)
                        plt.imshow(test_data[i * 8 + j, 0, :, :].cpu().numpy(), cmap='gray')
                        plt.axis(False)
                plt.savefig(utils.path.join(utils.REAL_MNIST_DIR, 'labels.png'))
                plt.tight_layout()
                plt.close()
                self.first = False
            label = test_label.view([self.batch_size, 2, 1, 1]).to(device)
            noise_label = torch.cat([noise, label], dim=1)
            fake_img = gen(noise_label)
            plt.figure(figsize=(8, 8))
            for i in range(8):
                for j in range(8):
                    plt.subplot(8, 8, i * 8 + j + 1)
                    plt.imshow(fake_img[i * 8 + j, 0, :, :].cpu().numpy(), cmap='gray')
                    plt.axis(False)
            plt.savefig(utils.path.join(utils.FAKE_MNIST_DIR, str(self.epoch) + '.png'))
            plt.tight_layout()
            plt.close()

    def train(self):
        data = np.load("data/reconstruct.npy")
        print(data.shape)
        labels = np.load("data/ut.npy")
        print(labels.shape)

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1,
                                                                            random_state=42)
        # 转tensor
        train_data = torch.from_numpy(train_data).float()
        train_data = train_data.unsqueeze(1)
        train_labels = torch.from_numpy(train_labels).float()

        test_data = torch.from_numpy(test_data).float()
        test_data = test_data.unsqueeze(1)
        test_labels = torch.from_numpy(test_labels).float()

        print(train_data.shape)
        print(train_labels.shape)
        # data = torch.cat([train_data.data, test_data.data]) / 127.5 - 1
        # target = torch.cat([train_data.targets, test_data.targets])

        dataset = TensorDataset(train_data, train_labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        test_dataset = TensorDataset(test_data, test_labels)
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        gen = Generator().to(device)
        dis = Discriminator().to(device)

        def d_loss_func(real, fake):
            return torch.mean(torch.square(real - 1.0)) + torch.mean(torch.square(fake)) + 1

        def g_loss_func(fake):
            return torch.mean(torch.square(fake - 1.0))

        d_optim = torch.optim.Adam(dis.parameters(), lr=0.00005)
        g_optim = torch.optim.Adam(gen.parameters(), lr=0.00020)
        test_noise = torch.randn(self.batch_size, 100, 1, 1).to(device)
        for _ in range(self.epochs):
            g_loss, d_loss = self.train_step(gen, dis, data_loader, g_loss_func, d_loss_func, g_optim, d_optim)
            print('Epoch {}/{}, g_loss = {:.4f}, d_loss = {:.4f}'.format(self.epoch + 1, self.epochs, g_loss, d_loss))
            self.test_step(gen, test_noise, test_data_loader)
            self.epoch += 1


if __name__ == '__main__':
    gan = CDCGAN(epochs=5000, batch_size=128)

    gan.train()
