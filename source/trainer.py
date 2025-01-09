from source.celeba import CelebADataset
from source.utils import *
from torch.utils.data import DataLoader
import torch, os
import torch.nn as nn
from source.gan_net import Generator, Discriminator
from source.conv_net import ImageEnhancementConvNet
from source.rbf_net import ImageEnhancementRBFNet
from source.train_loops import train_model, train_gan

class Trainer:
    def __init__(self, net_type='gan', num_samples=50000, num_epochs=50, batch_size=64, noise=False, params=None):
        self.net_type = net_type
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.noise = noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.net_type=='gan':
            self.backbone = params['backbone']
            self.g_lr = params['g_lr']
            self.d_lr = params['d_lr']
        else:
            self.lr = params['lr']

        self.results = None
        self._set_file_name()
        self._make_data_loaders()
        self._initialize_model()

    def _set_file_name(self):
        self.file_name = 'noise' if self.noise else 'celeba'
        self.file_name += f'_{self.net_type}'
        if self.net_type=='gan': self.file_name += f'_{self.backbone}'
        self.file_name += f'_s{self.num_samples}_e{self.num_epochs}_bs{self.batch_size}'
        if self.net_type=='gan': self.file_name += f'_{self.g_lr}_d_lr{self.d_lr}'
        else: self.file_name += f'_{self.lr}'

        print(f'Initialized trainer with file name:\n{self.file_name}')

    def _make_data_loaders(self):
        dataset = CelebADataset(noise=self.noise, num_samples=self.num_samples)
        dataset.load()
        resize = True if not self.net_type == 'rbf' else False
        x_train, y_train, x_test, y_test = dataset.get_train_test_split()
        train_dataset = ImageEnhancementDataset(x_train, y_train, resize=resize)
        test_dataset = ImageEnhancementDataset(x_test, y_test, resize=resize)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _initialize_model(self):
        if self.net_type=='gan':
            self.generator = Generator(backbone=self.backbone).to(self.device)
            self.discriminator = Discriminator(backbone=self.backbone).to(self.device)
            self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
            self.criterion = nn.BCELoss()
        elif self.net_type in ['conv', 'rbf']:
            self.model = ImageEnhancementConvNet() if self.net_type=='conv' else ImageEnhancementRBFNet()
            self.model.to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        if self.net_type=='gan':
            self.results = train_gan(self.generator,
                                self.discriminator,
                                self.train_loader,
                                self.optimizer_g,
                                self.optimizer_d,
                                self.criterion,
                                self.device,
                                epochs=self.num_epochs)
        elif self.net_type in ['conv', 'rbf']:
            self.results = train_model(self.model,
                                  self.train_loader,
                                  self.criterion,
                                  self.optimizer,
                                  self.device,
                                  epochs=self.num_epochs)

    def evaluate(self):
        metrics, x, y, y_hat, title = None, None, None, None, None
        if self.net_type=='gan':
            metrics, x, y, y_hat = evaluate_image_quality(self.generator, self.test_loader, self.device)
            title = f'GAN Image Enhancement ({self.backbone} backbone)'
        elif self.net_type in ['conv', 'rbf']:
            title = f'{self.net_type.upper()} Image Enhancement'
            metrics, x, y, y_hat = evaluate_image_quality(self.model, self.test_loader, self.device)

        self.results['test'] = metrics
        plot_examples_with_predictions(x, y, y_hat,
                                       save_path=f'./plots/{self.file_name}.png',
                                       title=title)
        save_metrics_to_json(self.results, f'./results/{self.file_name}.json')
        print("Examples and results saved")