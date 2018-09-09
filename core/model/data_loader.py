import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sample):
        if type(sample['action']) is not int:
            sample['action'] = sample['action'].tolist()
        return {'state': torch.from_numpy(sample['state']).type(torch.float).to(self.device),
                'action': torch.LongTensor([sample['action']]).type(torch.long).to(self.device),
                'delta': torch.from_numpy(sample['delta']).type(torch.float).to(self.device)}


class Continuous2DWorldTransitions(Dataset):
    """ Transitions of GridWorld of the form (state, action, next_state)"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = list(map(lambda fname: os.path.join(data_dir, fname),
                              os.listdir(self.data_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
            sample = {'state': instance[0],
                      'action': instance[1],
                      'delta': instance[2]}
            if self.transform:
                sample = self.transform(sample)
            return sample


if __name__ == '__main__':

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    g = Continuous2DWorldTransitions(os.path.join(project_root, 'data',
                                                  'input', 'test_dataset', 'train'),
                                     transform=ToTensor())

    data_loader = DataLoader(g, batch_size=1, shuffle=True, num_workers=1)
    print("Dataset size: {}".format(len(g)))
    for i_batch, sampled_batch in enumerate(data_loader):
        print(i_batch, sampled_batch['state'], sampled_batch['action'], sampled_batch['delta'])
        break
