import numpy as np
from PIL import Image
import os

# add dataset extension
class CelebADataset():
    def __init__(self, root_dir='..\data\celeba\img_align_celeba', test_split=0.4, norm='min-max', verbose=True, noise=False):
        self.verbose = verbose
        self.noise = noise
        self.norm = norm
        self.root_dir = root_dir
        self.height = 178
        self.width = 218

        # Place-holders
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    
    def load(self):
        """
        Load all the images from the root_dir and their corresponding labels.
        Assumes the images are in the folder structure:
        root_dir/image_name.jpg
        """

        # Return zeros for noise
        if self.noise:
            num_images = 100
            self.y = np.zeros((num_images, self.height, self.width, 3), dtype=np.uint8)  # RGB images
            return self.y

        image_files = [f for f in os.listdir(self.root_dir) if f.endswith('.jpg')]  # List all image files
        num_images = len(image_files)
        
        # Create arrays to hold the image data and labels
        y_data = np.zeros((num_images, self.height, self.width, 3), dtype=np.uint8)  # RGB images
            
        # Loop through each image and load it
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(self.root_dir, image_file)
            img = Image.open(image_path).resize((self.width, self.height))  # Resize if needed
            img = np.array(img)  # Convert to numpy array
            y_data[i] = img  # Store image

        self.y = y_data  # All images
        
        if self.verbose:
            print(f"Loaded {num_images} images from {self.root_dir}.")
        
        return y_data
    
    def get_train_test_split(self):
        # return x, y train and test
        return NotImplemented
    
    def statistical_analysis(self):
        # statistical analysis similar to cifar10
        return NotImplemented
    
    def _normalize(self):  
        if self.norm == 'min-max':
            # add logic for normalization
            return NotImplemented
        else:
            # add logic for other normalization methods
            return NotImplemented

    def _create_x(self):
        # the x is degraded version of y with lower quality
        return NotImplemented             

    def _split(self):
        # here we make a split based on self.test_split with stratification
        return NotImplemented
    
    def __len__(self):
        return NotImplemented
    
    def __get_item__(self):
        return NotImplemented

if __name__ == '__main__':
    dataset = CelebADataset()
    dataset.load()