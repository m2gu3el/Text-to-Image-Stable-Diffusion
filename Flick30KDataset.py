import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Flickr30kDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None):
        self.images_dir = images_dir
        self.captions_file = captions_file
        self.transform = transform
        self.image_filenames = os.listdir(images_dir)
        self.captions_df = self.load_captions()

    def load_captions(self):

      captions_df = pd.read_csv(self.captions_file, delimiter='|', header=None, names=['combined'])
      captions_split = captions_df['combined'].str.split('|', expand=True)
      captions_split.reset_index(inplace=True)
      captions_split.columns = ['image_name', 'comment_number', 'comment']
      if len(captions_split.columns) != 3:
        raise ValueError("Split operation did not produce three columns.")
    
      grouped_captions = captions_split.groupby('image_name')['comment'].apply(list).reset_index()
    
      #captions = (zip(grouped_captions['image_name'], grouped_captions['comment']))
    
      return grouped_captions





    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        caption = self.captions_df.loc[self.captions_df['image_name'] == image_filename]['comment'].values[0]

        if self.transform:
            image = self.transform(image)

        return image, caption

