import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def prepare_dataloader(batch_size=32):
    train_data_path = 'dataset_train/'
    val_data_path = 'dataset_valid/'
    test_data_path = 'dataset_test/'

    train_dataset = MyDataset(train_data_path)
    val_dataset = MyDataset(val_data_path)
    test_dataset = MyDataset(test_data_path)
    # Weight random sampler
    class_sample_count = np.array([len(np.where(train_dataset.future_values_da == t)[0]) for t in np.unique(train_dataset.future_values_da)])
    weight = 1. / class_sample_count
    sample_weight = np.array([weight[t] for t in train_dataset.future_values_da])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weight[:,0], len(sample_weight))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

class MyDataset(Dataset):
    def __init__(self, path):
        self.types_to_id = {'Receptive':0, 'Resistant to change':1, 'Open to change':2}
        self.past_values_da = np.load(path + 'past_values_da.npy')
        self.past_values_interlocutor = np.load(path + 'past_values_interlocutor.npy')
        self.past_time_features = np.load(path + 'past_time_features.npy')
        self.past_observed_mask = np.load(path + 'past_observed_mask.npy')
        self.static_categorical_features = np.load(path + 'static_categorical_features.npy')
        self.future_values_da = np.load(path + 'future_values_da.npy')
        self.future_values_interlocutor = np.load(path + 'future_values_interlocutor.npy')
        self.future_time_features = np.load(path + 'future_time_features.npy')
        self.attention_mask = np.load(path + 'attention_mask.npy')
        self.encoded_text = np.load(path + 'encoded_context.npy')

        self.data = [{
            'past_values_da': self.past_values_da[i],
            'past_values_interlocutor': self.past_values_interlocutor[i],
            'past_time_features': self.past_time_features[i],
            'past_observed_mask': self.past_observed_mask[i],
            'static_categorical_features': self.static_categorical_features[i],
            'future_values_da': self.future_values_da[i],
            'future_values_interlocutor': self.future_values_interlocutor[i],
            'future_time_features': self.future_time_features[i],
            'attention_mask': self.attention_mask[i],
            'text': torch.tensor(self.encoded_text[i]).float(),
        } for i in range(len(self.past_values_da))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        past_values_da = torch.tensor(row['past_values_da'])
        past_values_interlocutor = torch.tensor(row['past_values_interlocutor'])
        past_time_features = torch.tensor(row['past_time_features'])
        past_observed_mask = torch.tensor(row['past_observed_mask'])
        static_categorical_features = torch.tensor(row['static_categorical_features'])
        future_values_da = torch.tensor(row['future_values_da'])
        future_values_interlocutor = torch.tensor(row['future_values_interlocutor'])
        future_time_features = torch.tensor(row['future_time_features'])
        attention_mask = torch.tensor(row['attention_mask'])
        batch_text = row['text']

        return {
            'past_values_da': past_values_da,
            'past_values_interlocutor': past_values_interlocutor,
            'past_time_features': past_time_features,
            'past_observed_mask': past_observed_mask,
            'static_categorical_features': static_categorical_features,
            'future_values_da': future_values_da,
            'future_values_interlocutor': future_values_interlocutor,
            'future_time_features': future_time_features,
            'attention_mask': attention_mask,
            'text': batch_text
        }