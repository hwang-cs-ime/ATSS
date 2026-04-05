import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class SimMatrixDataset(Dataset):
    def __init__(self, image_dir, text_dir, label_dir, file_list, num_frames=8):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        image_feat = np.load(os.path.join(self.image_dir, f'{name}.npy'))
        text_feat = np.load(os.path.join(self.text_dir, f'{name}.npy'))
        label = np.load(os.path.join(self.label_dir, f'{name}.npy'))

        # L2 normalize
        image_feat = image_feat / np.linalg.norm(image_feat, axis=1, keepdims=True)
        text_feat = text_feat / np.linalg.norm(text_feat, axis=1, keepdims=True)

        # Compute similarity matrices
        sim_img = np.dot(image_feat, image_feat.T)
        sim_txt = np.dot(text_feat, text_feat.T)
        sim_cross = np.dot(image_feat, text_feat.T)

        # Sample frames
        indices = np.linspace(0, sim_img.shape[0] - 1, self.num_frames).astype(int)
        sim_img = sim_img[indices][:, indices]
        sim_txt = sim_txt[indices][:, indices]
        sim_cross = sim_cross[indices][:, indices]

        sim_img = torch.from_numpy(sim_img).float()
        sim_txt = torch.from_numpy(sim_txt).float()
        sim_cross = torch.from_numpy(sim_cross).float()
        label = torch.tensor(label).long()

        return sim_img, sim_txt, sim_cross, label


def split_dataset_custom(test_pos_dirs, test_neg_path, num_frames=8):
    def get_file_names(root, min_frames=8):
        label_path = os.path.join(root, 'label')
        image_feat_path = os.path.join(root, 'feat_image')

        valid_files = []
        for f in os.listdir(label_path):
            if f.endswith('.npy'):
                name = f.split('.npy')[0]
                try:
                    image_feat = np.load(os.path.join(image_feat_path, f'{name}.npy'))
                    if image_feat.shape[0] >= min_frames:
                        valid_files.append(name)
                    else:
                        print(f"Skipping {name} - only {image_feat.shape[0]} frames available")
                except Exception as e:
                    print(f"Error checking {name}: {e}")
        return valid_files

    def build_ds(root, files):
        return SimMatrixDataset(os.path.join(root, 'feat_image'),
                                os.path.join(root, 'feat_text'),
                                os.path.join(root, 'label'),
                                files,
                                num_frames=num_frames)

    test_neg_files_all = get_file_names(test_neg_path, min_frames=num_frames)
    test_datasets = []

    for test_pos_dir in test_pos_dirs:
        test_pos_files = get_file_names(test_pos_dir, min_frames=num_frames)
        random.shuffle(test_pos_files)

        test_pos_ds = build_ds(test_pos_dir, test_pos_files)
        test_neg_files = random.sample(test_neg_files_all, len(test_pos_files))
        test_neg_ds = build_ds(test_neg_path, test_neg_files)

        test_datasets.append(torch.utils.data.ConcatDataset([test_pos_ds, test_neg_ds]))

    return test_datasets
