import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)



class TripletFinetuneBlister(Dataset):
    def __init__(self, blister_dataset, current_pred_status):
        self.blister_dataset = blister_dataset
        self.transform = self.blister_dataset.transform
        
        self.train_paths = [item[0] for item in self.blister_dataset.imgs]
        self.train_labels = np.array([item[1] for item in self.blister_dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                    for label in self.labels_set}
        # additional
        self.current_pred_status = {s[0]: (s[1], s[2]) for s in current_pred_status}
        # print(self.current_pred_status)
        self.wrong_sample_pred_labels = {s[0]: s[1] for s in current_pred_status if s[1] != s[2]}
        print(len(self.wrong_sample_pred_labels))
    
    def __getitem__(self, index):
        # change index to image index
        index = list(self.current_pred_status.keys())[index]

        # incorrect sample
        if self.current_pred_status[index][0] != self.current_pred_status[index][1]:
            # 取 positive
            label_p = self.train_labels[index].item()
            img_p = Image.open(self.train_paths[index])
            # img_p.show()
            
            # 取 anchor (確保不會取到 positive)
            a_index = index
            while a_index == index:
                a_index = np.random.choice(self.label_to_indices[label_p])
            img_a = Image.open(self.train_paths[a_index])
            # img_a.show()
            
            # 取 negative
            label_n = self.wrong_sample_pred_labels[index] # 取得預測錯誤的當成負樣本
            n_index = np.random.choice(self.label_to_indices[label_n]) # 利用標記取到影像
            img_n = Image.open(self.train_paths[n_index])
            # img_n.show()
            
            if self.transform is not None:
                img_p = self.transform(img_p)
                img_a = self.transform(img_a)
                img_n = self.transform(img_n)

            # print('dataloader:', index, label_p, label_n)
            return (img_a, img_p, img_n), []
        # correct sample
        else: 
            # 取 positive
            label_a = self.train_labels[index].item()
            img_a = Image.open(self.train_paths[index])
            
            # 取 anchor (確保不會取到 positive)
            p_index = index
            while p_index == index:
                p_index = np.random.choice(self.label_to_indices[label_a])
            img_p = Image.open(self.train_paths[p_index])
            
            # 取 negative
            label_n = np.random.choice(list(self.labels_set - set([label_a]))) # 隨機取得負樣本標記
            n_index = np.random.choice(self.label_to_indices[label_n]) # 利用標記取到影像
            img_n = Image.open(self.train_paths[n_index])
            
            if self.transform is not None:
                img_a = self.transform(img_a)
                img_p = self.transform(img_p)
                img_n = self.transform(img_n)
            return (img_a, img_p, img_n), []
        
    def __len__(self):
        return len(self.current_pred_status)

class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            '''
            利用 label 把不同類別圖片的 index 都分出來
            label_to_indices = {
                0: array([img_index_number...])
                1: array([img_index_number...])
                ...
                9: array([img_index_number...])
            }
            '''
        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            # 跟上面一樣
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set} 

            random_state = np.random.RandomState(29)

            triplets = [[i, # index
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]), # positive index
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ]) # negative index
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            # 取一組圖像與標記
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            # 取一組正樣本
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            # 取一組負樣本
            negative_label = np.random.choice(list(self.labels_set - set([label1]))) # 隨機取得負樣本標記
            negative_index = np.random.choice(self.label_to_indices[negative_label]) # 利用標記取到影像
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
        


class Prototype_Dataset(Dataset):

    def __init__(self, dataset, class_num, prototype_index):
        
        self.transform = dataset.transform
        self.train_paths = [item[0] for item in dataset.imgs]
        self.train_labels = np.array([item[1] for item in dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                for label in self.labels_set}
        self.prototype_index = prototype_index
        self.class_num = class_num

    def __getitem__(self, index):

        image_index = self.label_to_indices[index]
        all_prototype_img = []
        
        for each_index in image_index[self.prototype_index]:
            all_prototype_img.append(self.transform(Image.open(self.train_paths[each_index])))
        
        return all_prototype_img

    def __len__(self):
        return self.class_num

class query_Dataset(Dataset):

    def __init__(self, dataset, class_num, query_index, add_class):
        
        self.transform = dataset.transform
        self.train_paths = [item[0] for item in dataset.imgs]
        self.train_labels = np.array([item[1] for item in dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                for label in self.labels_set}
        self.query_index = query_index
        self.class_num = class_num
        self.add_class = add_class

    def __getitem__(self, index):
        
        cls_num = index // len(self.query_index) + self.add_class
        idx_num = self.query_index[index % len(self.query_index)]
        image_index = self.label_to_indices[cls_num]
        return self.transform(Image.open(self.train_paths[image_index[idx_num]])), cls_num , idx_num

    def __len__(self):
        return self.class_num * len(self.query_index)


class new_sample_dataset(Dataset):

    def __init__(self, dataset, new_sample):
        
        self.imgs = dataset.imgs
        self.transform = dataset.transform
        self.new_sample = new_sample
        self.now_idx = 0

    def __getitem__(self, index):
        
        while not self.now_idx in self.new_sample:
            self.now_idx += 1
        self.now_idx += 1
        return self.dataset[self.now_idx - 1]

    def __len__(self):
        return len(self.new_sample)


class WrongTripletBlister(Dataset):
    def __init__(self, blister_dataset, class_num, wrong_idx, selected_idx_230):
        self.blister_dataset = blister_dataset
        self.transform = self.blister_dataset.transform

        self.train_paths = [item[0] for item in self.blister_dataset.imgs]
        self.train_labels = np.array([item[1] for item in self.blister_dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                    for label in self.labels_set}
        
        self.class_num = class_num
        self.wrong_idx = wrong_idx
        self.selected_idx_230 = selected_idx_230

        # print(self.train_labels)
        # print(self.labels_set)
        # print(self.label_to_indices)

    def __getitem__(self, index):


        # 取 anchor
        # label1 = self.train_labels[index].item()
        label1 = int(self.wrong_idx[index // 3][0])
        img_index = self.label_to_indices[label1][int(self.wrong_idx[index // 3][1])]
        img1 = Image.open(self.train_paths[img_index])
        # 取 positive (確保不會取到 anchor)
        positive_index = int(self.wrong_idx[index // 3][1])
        while positive_index == int(self.wrong_idx[index // 3][1]):
            positive_index = random.choice(self.selected_idx_230)
        positive_img_index = self.label_to_indices[label1][positive_index]
        img2 = Image.open(self.train_paths[positive_img_index])
        # 取 negative
        negative_label = int(self.wrong_idx[index // 3][2]) 
        negative_index = random.choice(self.selected_idx_230) # 利用標記取到影像

        negative_img_index = self.label_to_indices[negative_label][negative_index]
        img3 = Image.open(self.train_paths[negative_img_index])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []
    def __len__(self):
        return len(self.wrong_idx) * 3



class TripletBlister(Dataset):
    def __init__(self, blister_dataset, class_num, query_index):
        self.blister_dataset = blister_dataset
        self.transform = self.blister_dataset.transform

        self.train_paths = [item[0] for item in self.blister_dataset.imgs]
        self.train_labels = np.array([item[1] for item in self.blister_dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                    for label in self.labels_set}
        
        self.class_num = class_num
        self.query_index = query_index

        # print(self.train_labels)
        # print(self.labels_set)
        # print(self.label_to_indices)

    def __getitem__(self, index):


        # 取 anchor
        # label1 = self.train_labels[index].item()
        label1 = index // len(self.query_index)
        anchor_index = self.label_to_indices[label1][index % len(self.query_index)]
        img1 = Image.open(self.train_paths[anchor_index])
        # 取 positive (確保不會取到 anchor)
        positive_index = index % len(self.query_index)
        while positive_index == index % len(self.query_index):
            positive_index = np.random.choice(self.query_index)
        positive_index = self.label_to_indices[label1][positive_index]
        img2 = Image.open(self.train_paths[positive_index])
        # 取 negative
        negative_label = np.random.choice(list(self.labels_set - set([label1]))) # 隨機取得負樣本標記
        negative_index = np.random.choice(self.query_index) # 利用標記取到影像
        negative_index = self.label_to_indices[negative_label][negative_index]
        img3 = Image.open(self.train_paths[negative_index])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []
    def __len__(self):
        return self.class_num * len(self.query_index)


# class Triplet_query_train_Dataset(Dataset):

#     def __init__(self, dataset, class_num, prototype_index):
        
#         self.transform = dataset.transform
#         self.train_paths = [item[0] for item in dataset.imgs]
#         self.train_labels = np.array([item[1] for item in dataset.imgs])
#         self.labels_set = set(self.train_labels)
#         self.label_to_indices = {label: np.where(self.train_labels == label)[0]
#                                 for label in self.labels_set}
#         self.prototype_index = prototype_index
#         self.class_num = class_num

#     def __getitem__(self, index):
        
#         label1 = index // 13
#         idx_num = self.prototype_index[index % 13]
#         image_index = self.label_to_indices[label1]
#         img1 = Image.open(self.train_paths[image_index[idx_num]])

#         # 取 positive (確保不會取到 anchor)
#         positive_index = idx_num
#         while positive_index == idx_num:
#             positive_index = np.random.choice(self.prototype_index)
#         img2 = Image.open(self.train_paths[image_index[positive_index]])

#         # 取 negative
#         negative_label = np.random.choice(list(set(range(self.class_num)) - set([label1]))) # 隨機取得負樣本標記
#         negative_index = np.random.choice(list(set(self.prototype_index) - set([idx_num]))) # 利用標記取到影像
#         img3 = self.label_to_indices[negative_label][negative_index]
#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#             img3 = self.transform(img3)
#         return (img1, img2, img3), []

#     def __len__(self):
#         return self.class_num * len(self.prototype_index)

