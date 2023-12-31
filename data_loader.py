import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import (DataLoader, TensorDataset)
import pandas as pd
from sklearn.model_selection import train_test_split


class MyDataLoader():
    def __init__(self, dataset_path, tokenizer, max_length, batch_size):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.batch_size = batch_size

    def load_csv(self, dataset_path):
        df = pd.read_csv(dataset_path)
        # drop the row which has nan value
        df = df.dropna()
        df = df[df['tag'] != 'UNKNOW'].dropna()
        texts = df.post_text.values.tolist()
        labels = df.tag.values.tolist()

        label_0 = [i for i in labels if i == "AGENCY"]
        label_1 = [i for i in labels if i == "PROTENTIAL"]
        label_2 = [i for i in labels if i == "SHARING"]
        label_3 = [i for i in labels if i == "SPAM"]

        print('has {} label AGENCY in {} total label'.format(len(label_0), len(labels)))
        print('has {} label PROTENTIAL in {} total label'.format(len(label_1), len(labels)))
        print('has {} label SHARING in {} total label'.format(len(label_2), len(labels)))
        print('has {} label SPAM {} total label'.format(len(label_3), len(labels)))

        train_x, val_x, train_y, val_y = train_test_split(texts, labels, test_size=0.2, shuffle=True)

        # Khởi tạo trình mã hóa nhãn
        label_encoder = LabelEncoder()

        # Fit và chuyển đổi nhãn cho tập huấn luyện
        train_y = label_encoder.fit_transform(train_y)

        # Chuyển đổi nhãn cho tập validation
        val_y = label_encoder.transform(val_y)

        # In bảng ánh xạ giữa nhãn gốc và nhãn đã mã hóa
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("Bảng ánh xạ nhãn:", label_mapping)

        return train_x, val_x, train_y, val_y

    def dataloader(self):
        print("Loading dataloader...")
        train_x, val_x, train_y, val_y = self.load_csv(self.dataset_path)
        # print(train_x[3])


        tokenizer_data_train = self.tokenizer.batch_encode_plus(train_x,
                                                                add_special_tokens=True,
                                                                return_attention_mask=True,
                                                                pad_to_max_length=True,
                                                                max_length=self.max_length,
                                                                truncation=True,
                                                                return_tensors='pt')
        # print(tokenizer_data_train)
        tokenizer_data_val = self.tokenizer.batch_encode_plus(val_x,
                                                              add_special_tokens=True,
                                                              return_attention_mask=True,
                                                              pad_to_max_length=True,
                                                              max_length=self.max_length,
                                                              truncation=True,
                                                              return_tensors='pt')
        print(type(train_x))
        print("=====================================")
        print(type(train_y))
        # print(train_y.head())
        print(train_y)
        input_ids_train = tokenizer_data_train['input_ids']
        attention_masks_train = tokenizer_data_train['attention_mask']
        labels_train = torch.tensor(train_y)

        input_ids_val = tokenizer_data_val['input_ids']
        attention_masks_val = tokenizer_data_val['attention_mask']
        labels_val = torch.tensor(val_y)

        train_data = TensorDataset(
            input_ids_train, attention_masks_train, labels_train)
        val_data = TensorDataset(
            input_ids_val, attention_masks_val, labels_val)

        train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size)

        val_dataloader = DataLoader(
            val_data, batch_size=self.batch_size)
        return train_dataloader, val_dataloader
