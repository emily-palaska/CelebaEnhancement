import numpy as np
from PIL import Image
import os, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import plot_examples

class CelebADataset:
    def __init__(self, root_dir='../data/', test_split=0.4, norm='min-max', verbose=True, noise=False, num_samples=None):
        self.verbose = verbose
        self.noise = noise
        self.norm = norm
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.height = 178
        self.width = 218
        self.test_split = test_split

        # Place-holders
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load(self):
        """
        Load all the images or retrieve from HDF5 for faster loading.
        """
        start_time = time.time()

        if self.noise:
            self.y = np.random.randint(0, 256, (self.num_samples, self.height, self.width, 3), dtype=np.uint8)
        else:
            base_folder = 'img_align_celeba'
            image_dir = os.path.join(self.root_dir, base_folder)
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            num_images = self.num_samples if self.num_samples else len(image_files)
            image_files = image_files[:num_images]

            y_data = np.zeros((num_images, self.height, self.width, 3), dtype=np.uint8)
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(image_dir, image_file)
                img = Image.open(image_path).resize((self.width, self.height))
                y_data[i] = np.array(img).astype(np.uint8)

            self.y = y_data


        end_time = time.time()

        if self.verbose:
            print(f"Loaded {len(self.y)} images in {end_time - start_time:.2f}s.")

        self._create_x()
        self._split()

    def get_train_test_split(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def statistical_analysis(self, save_path='./examples.png'):
        """
        Perform statistical analysis such as mean, standard deviation, min, max for each channel.
        Also plot a grid with some random examples of x and y pairs.
        """
        means = self.y.mean(axis=(0, 1, 2))
        stds = self.y.std(axis=(0, 1, 2))
        mins = self.y.min(axis=(0, 1, 2))
        maxs = self.y.max(axis=(0, 1, 2))

        stats = {
            'mean': means.tolist(),
            'std': stds.tolist(),
            'min': mins.tolist(),
            'max': maxs.tolist()
        }
        if self.verbose:
            print("Statistical analysis:", stats)
        plot_examples(self.x_train, self.y_train, save_path=save_path)            
        return stats

    def _normalize(self):
        """
        Normalize the dataset based on the specified method.
        """
        if self.norm == 'min-max':
            self.y = (self.y - self.y.min()) / (self.y.max() - self.y.min())
        elif self.norm == 'z-score':
            self.y = (self.y - self.y.mean(axis=(0, 1, 2))) / self.y.std(axis=(0, 1, 2))
        else:
            raise NotImplementedError(f'Normalization method {self.norm} not implemented.')

    def _create_x(self):
        """
        Create degraded versions of y to form x.
        For example, apply blurring to reduce quality.
        """
        self.x = np.copy(self.y)
        for i in range(self.x.shape[0]):
            img = Image.fromarray(self.x[i])
            img = img.resize((self.width // 5, self.height // 5)).resize((self.width, self.height))
            self.x[i] = np.array(img)

    def _split(self):
        """
        Split the dataset into training and testing sets based on the test_split.
        """

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=self.test_split,
                                                                                random_state=42,
                                                                                stratify=None)
       