import os
import sys
import torch
import random
import pickle
import config
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn import metrics
from torchvision.models import resnet50
from torchvision import transforms, datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_recall_fscore_support


sys.path.append(os.path.abspath('../'))
from triplet_utils.losses import TripletLoss
from triplet_utils.networks import TripletNet
from triplet_utils.dataset import TripletBlister
from triplet_utils.function import get_prototype_feature, test_epoch, train_epoch
from triplet_utils.dataset import Prototype_Dataset, query_Dataset, WrongTripletBlister


cuda = torch.cuda.is_available()
data_transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



#------------------------------------------------------------------------------
# B1 180 class training loader and test loader and prototype loader 
#------------------------------------------------------------------------------

# B1_180_set = datasets.ImageFolder(root=config.B1_180, transform=data_transform)
# random.seed(config.random_seed)
# B1_180_test_index = list(random.sample(range(config.B1_type_size), config.test_img_size))
# random.seed(config.random_seed)
# B1_180_prototype_index = list(random.sample(list(set(range(config.B1_type_size)) - set(B1_180_test_index)), config.prototype_size))
# B1_180_train_index =     list(random.sample(list(set(range(config.B1_type_size)) - set(B1_180_test_index)), config.train_img_size))

# B1_180_prototype_set = Prototype_Dataset(B1_180_set, 180, B1_180_prototype_index)
# B1_180_test_set = query_Dataset(B1_180_set, 180, B1_180_test_index, 0)
# B1_180_training_set = TripletBlister(B1_180_set, 180, B1_180_train_index)

# B1_180_prototype_loader = torch.utils.data.DataLoader(B1_180_prototype_set, batch_size=1, shuffle=False)
# B1_180_test_loader = torch.utils.data.DataLoader(B1_180_test_set, batch_size=8, shuffle=False)
# B1_180_train_loader = torch.utils.data.DataLoader(B1_180_training_set, batch_size=8, shuffle=True)


#------------------------------------------------------------------------------
# B1 50 class test loader and prototype loader 
#------------------------------------------------------------------------------

# B1_50_set = datasets.ImageFolder(root=config.B1_50, transform=data_transform)
# random.seed(config.random_seed)
# B1_50_test_index = list(random.sample(range(config.B1_type_size), config.test_img_size))
# random.seed(config.random_seed)
# B1_50_prototype_index = list(random.sample(list(set(range(config.B1_type_size)) - set(B1_50_test_index)), config.prototype_size))

# B1_50_prototype_set = Prototype_Dataset(B1_50_set, 50, B1_50_prototype_index)
# B1_50_test_set = query_Dataset(B1_50_set, 50, B1_50_test_index, 180)

# B1_50_prototype_loader = torch.utils.data.DataLoader(B1_50_prototype_set, batch_size=1, shuffle=False)
# B1_50_test_loader = torch.utils.data.DataLoader(B1_50_test_set, batch_size=8, shuffle=False)


#------------------------------------------------------------------------------
# B2 180 class tune loader and test loader and prototype loader 
#------------------------------------------------------------------------------

B2_180_set = datasets.ImageFolder(root=config.B2_180, transform=data_transform)
random.seed(config.random_seed)
B2_180_test_index = list(random.sample(range(config.B2_type_size), config.test_img_size))
random.seed(config.random_seed)
B2_180_prototype_index = list(random.sample(list(set(range(config.B2_type_size)) - set(B2_180_test_index)), config.prototype_size))
B2_180_tune_index =      list(random.sample(list(set(range(config.B2_type_size)) - set(B2_180_test_index)), config.tune_img_size))

B2_180_prototype_set = Prototype_Dataset(B2_180_set, 180, B2_180_prototype_index)
B2_180_test_set = query_Dataset(B2_180_set, 180, B2_180_test_index, 0)
B2_180_tune_set = TripletBlister(B2_180_set, 180, B2_180_tune_index)

B2_180_prototype_loader = torch.utils.data.DataLoader(B2_180_prototype_set, batch_size=1, shuffle=False)
B2_180_test_loader = torch.utils.data.DataLoader(B2_180_test_set, batch_size=8, shuffle=False)
B2_180_tune_loader = torch.utils.data.DataLoader(B2_180_tune_set, batch_size=6, shuffle=True)


#------------------------------------------------------------------------------
# B2 50 class test loader and prototype loader 
#------------------------------------------------------------------------------

B2_230_set = datasets.ImageFolder(root=config.B2_230, transform=data_transform)
random.seed(config.random_seed)
B2_230_shuffled = list(range(config.B2_type_size))
random.shuffle(B2_230_shuffled)

B2_230_prototype_set = Prototype_Dataset(B2_230_set, 230, B2_230_shuffled[:10])
B2_230_test_set_v1 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[10:20], 0)
B2_230_test_set_v2 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[20:30], 0)
B2_230_test_set_v3 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[30:40], 0)
B2_230_test_set_v4 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[40:50], 0)
B2_230_test_set_v5 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[50:60], 0)
B2_230_test_set_v6 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[60:70], 0)
B2_230_test_set_v7 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[70:80], 0)
B2_230_test_set_v8 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[80:90], 0)
B2_230_test_set_v9 =  query_Dataset(B2_230_set, 230, B2_230_shuffled[90:100], 0)
B2_230_test_set_v10 = query_Dataset(B2_230_set, 230, B2_230_shuffled[100:110], 0)
B2_230_test_set_v11 = query_Dataset(B2_230_set, 230, B2_230_shuffled[110:120], 0)
B2_230_test_set_v12 = query_Dataset(B2_230_set, 230, B2_230_shuffled[120:130], 0)
B2_230_test_set_v13 = query_Dataset(B2_230_set, 230, B2_230_shuffled[130:140], 0)
B2_230_test_set_v14 = query_Dataset(B2_230_set, 230, B2_230_shuffled[140:150], 0)
B2_230_test_set_v15 = query_Dataset(B2_230_set, 230, B2_230_shuffled[150:160], 0)
B2_230_test_set_v16 = query_Dataset(B2_230_set, 230, B2_230_shuffled[160:170], 0)
B2_230_test_set_v17 = query_Dataset(B2_230_set, 230, B2_230_shuffled[170:180], 0)
B2_230_test_set_v18 = query_Dataset(B2_230_set, 230, B2_230_shuffled[180:190], 0)
B2_230_test_set_v19 = query_Dataset(B2_230_set, 230, B2_230_shuffled[190:200], 0)
B2_230_test_set_v20 = query_Dataset(B2_230_set, 230, B2_230_shuffled[200:], 0)

B2_180_test_set_v1 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[10:20], 0)
B2_180_test_set_v2 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[20:30], 0)
B2_180_test_set_v3 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[30:40], 0)
B2_180_test_set_v4 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[40:50], 0)
B2_180_test_set_v5 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[50:60], 0)
B2_180_test_set_v6 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[60:70], 0)
B2_180_test_set_v7 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[70:80], 0)
B2_180_test_set_v8 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[80:90], 0)
B2_180_test_set_v9 =  query_Dataset(B2_230_set, 180, B2_230_shuffled[90:100], 0)
B2_180_test_set_v10 = query_Dataset(B2_230_set, 180, B2_230_shuffled[100:110], 0)
B2_180_test_set_v11 = query_Dataset(B2_230_set, 180, B2_230_shuffled[110:120], 0)
B2_180_test_set_v12 = query_Dataset(B2_230_set, 180, B2_230_shuffled[120:130], 0)
B2_180_test_set_v13 = query_Dataset(B2_230_set, 180, B2_230_shuffled[130:140], 0)
B2_180_test_set_v14 = query_Dataset(B2_230_set, 180, B2_230_shuffled[140:150], 0)
B2_180_test_set_v15 = query_Dataset(B2_230_set, 180, B2_230_shuffled[150:160], 0)
B2_180_test_set_v16 = query_Dataset(B2_230_set, 180, B2_230_shuffled[160:170], 0)
B2_180_test_set_v17 = query_Dataset(B2_230_set, 180, B2_230_shuffled[170:180], 0)
B2_180_test_set_v18 = query_Dataset(B2_230_set, 180, B2_230_shuffled[180:190], 0)
B2_180_test_set_v19 = query_Dataset(B2_230_set, 180, B2_230_shuffled[190:200], 0)
B2_180_test_set_v20 = query_Dataset(B2_230_set, 180, B2_230_shuffled[200:], 0)

B2_50_test_set_v1 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[10:20], 180)
B2_50_test_set_v2 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[20:30], 180)
B2_50_test_set_v3 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[30:40], 180)
B2_50_test_set_v4 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[40:50], 180)
B2_50_test_set_v5 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[50:60], 180)
B2_50_test_set_v6 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[60:70], 180)
B2_50_test_set_v7 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[70:80], 180)
B2_50_test_set_v8 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[80:90], 180)
B2_50_test_set_v9 =  query_Dataset(B2_230_set, 50, B2_230_shuffled[90:100], 180)
B2_50_test_set_v10 = query_Dataset(B2_230_set, 50, B2_230_shuffled[100:110], 180)
B2_50_test_set_v11 = query_Dataset(B2_230_set, 50, B2_230_shuffled[110:120], 180)
B2_50_test_set_v12 = query_Dataset(B2_230_set, 50, B2_230_shuffled[120:130], 180)
B2_50_test_set_v13 = query_Dataset(B2_230_set, 50, B2_230_shuffled[130:140], 180)
B2_50_test_set_v14 = query_Dataset(B2_230_set, 50, B2_230_shuffled[140:150], 180)
B2_50_test_set_v15 = query_Dataset(B2_230_set, 50, B2_230_shuffled[150:160], 180)
B2_50_test_set_v16 = query_Dataset(B2_230_set, 50, B2_230_shuffled[160:170], 180)
B2_50_test_set_v17 = query_Dataset(B2_230_set, 50, B2_230_shuffled[170:180], 180)
B2_50_test_set_v18 = query_Dataset(B2_230_set, 50, B2_230_shuffled[180:190], 180)
B2_50_test_set_v19 = query_Dataset(B2_230_set, 50, B2_230_shuffled[190:200], 180)
B2_50_test_set_v20 = query_Dataset(B2_230_set, 50, B2_230_shuffled[200:], 180)


B2_230_prototype_loader = torch.utils.data.DataLoader(B2_230_prototype_set, batch_size=1, shuffle=False)
B2_230_test_loader_v1 = torch.utils.data.DataLoader(B2_230_test_set_v1, batch_size=8, shuffle=False)
B2_230_test_loader_v2 = torch.utils.data.DataLoader(B2_230_test_set_v2, batch_size=8, shuffle=False)
B2_230_test_loader_v3 = torch.utils.data.DataLoader(B2_230_test_set_v3, batch_size=8, shuffle=False)
B2_230_test_loader_v4 = torch.utils.data.DataLoader(B2_230_test_set_v4, batch_size=8, shuffle=False)
B2_230_test_loader_v5 = torch.utils.data.DataLoader(B2_230_test_set_v5, batch_size=8, shuffle=False)
B2_230_test_loader_v6 = torch.utils.data.DataLoader(B2_230_test_set_v6, batch_size=8, shuffle=False)
B2_230_test_loader_v7 = torch.utils.data.DataLoader(B2_230_test_set_v7, batch_size=8, shuffle=False)
B2_230_test_loader_v8 = torch.utils.data.DataLoader(B2_230_test_set_v8, batch_size=8, shuffle=False)
B2_230_test_loader_v9 = torch.utils.data.DataLoader(B2_230_test_set_v9, batch_size=8, shuffle=False)
B2_230_test_loader_v10 = torch.utils.data.DataLoader(B2_230_test_set_v10, batch_size=8, shuffle=False)
B2_230_test_loader_v11 = torch.utils.data.DataLoader(B2_230_test_set_v11, batch_size=8, shuffle=False)
B2_230_test_loader_v12 = torch.utils.data.DataLoader(B2_230_test_set_v12, batch_size=8, shuffle=False)
B2_230_test_loader_v13 = torch.utils.data.DataLoader(B2_230_test_set_v13, batch_size=8, shuffle=False)
B2_230_test_loader_v14 = torch.utils.data.DataLoader(B2_230_test_set_v14, batch_size=8, shuffle=False)
B2_230_test_loader_v15 = torch.utils.data.DataLoader(B2_230_test_set_v15, batch_size=8, shuffle=False)
B2_230_test_loader_v16 = torch.utils.data.DataLoader(B2_230_test_set_v16, batch_size=8, shuffle=False)
B2_230_test_loader_v17 = torch.utils.data.DataLoader(B2_230_test_set_v17, batch_size=8, shuffle=False)
B2_230_test_loader_v18 = torch.utils.data.DataLoader(B2_230_test_set_v18, batch_size=8, shuffle=False)
B2_230_test_loader_v19 = torch.utils.data.DataLoader(B2_230_test_set_v19, batch_size=8, shuffle=False)
B2_230_test_loader_v20 = torch.utils.data.DataLoader(B2_230_test_set_v20, batch_size=8, shuffle=False)

B2_180_test_loader_v1 = torch.utils.data.DataLoader(B2_180_test_set_v1, batch_size=8, shuffle=False)
B2_180_test_loader_v2 = torch.utils.data.DataLoader(B2_180_test_set_v2, batch_size=8, shuffle=False)
B2_180_test_loader_v3 = torch.utils.data.DataLoader(B2_180_test_set_v3, batch_size=8, shuffle=False)
B2_180_test_loader_v4 = torch.utils.data.DataLoader(B2_180_test_set_v4, batch_size=8, shuffle=False)
B2_180_test_loader_v5 = torch.utils.data.DataLoader(B2_180_test_set_v5, batch_size=8, shuffle=False)
B2_180_test_loader_v6 = torch.utils.data.DataLoader(B2_180_test_set_v6, batch_size=8, shuffle=False)
B2_180_test_loader_v7 = torch.utils.data.DataLoader(B2_180_test_set_v7, batch_size=8, shuffle=False)
B2_180_test_loader_v8 = torch.utils.data.DataLoader(B2_180_test_set_v8, batch_size=8, shuffle=False)
B2_180_test_loader_v9 = torch.utils.data.DataLoader(B2_180_test_set_v9, batch_size=8, shuffle=False)
B2_180_test_loader_v10 = torch.utils.data.DataLoader(B2_180_test_set_v10, batch_size=8, shuffle=False)
B2_180_test_loader_v11 = torch.utils.data.DataLoader(B2_180_test_set_v11, batch_size=8, shuffle=False)
B2_180_test_loader_v12 = torch.utils.data.DataLoader(B2_180_test_set_v12, batch_size=8, shuffle=False)
B2_180_test_loader_v13 = torch.utils.data.DataLoader(B2_180_test_set_v13, batch_size=8, shuffle=False)
B2_180_test_loader_v14 = torch.utils.data.DataLoader(B2_180_test_set_v14, batch_size=8, shuffle=False)
B2_180_test_loader_v15 = torch.utils.data.DataLoader(B2_180_test_set_v15, batch_size=8, shuffle=False)
B2_180_test_loader_v16 = torch.utils.data.DataLoader(B2_180_test_set_v16, batch_size=8, shuffle=False)
B2_180_test_loader_v17 = torch.utils.data.DataLoader(B2_180_test_set_v17, batch_size=8, shuffle=False)
B2_180_test_loader_v18 = torch.utils.data.DataLoader(B2_180_test_set_v18, batch_size=8, shuffle=False)
B2_180_test_loader_v19 = torch.utils.data.DataLoader(B2_180_test_set_v19, batch_size=8, shuffle=False)
B2_180_test_loader_v20 = torch.utils.data.DataLoader(B2_180_test_set_v20, batch_size=8, shuffle=False)

B2_50_test_loader_v1 = torch.utils.data.DataLoader(B2_50_test_set_v1, batch_size=8, shuffle=False)
B2_50_test_loader_v2 = torch.utils.data.DataLoader(B2_50_test_set_v2, batch_size=8, shuffle=False)
B2_50_test_loader_v3 = torch.utils.data.DataLoader(B2_50_test_set_v3, batch_size=8, shuffle=False)
B2_50_test_loader_v4 = torch.utils.data.DataLoader(B2_50_test_set_v4, batch_size=8, shuffle=False)
B2_50_test_loader_v5 = torch.utils.data.DataLoader(B2_50_test_set_v5, batch_size=8, shuffle=False)
B2_50_test_loader_v6 = torch.utils.data.DataLoader(B2_50_test_set_v6, batch_size=8, shuffle=False)
B2_50_test_loader_v7 = torch.utils.data.DataLoader(B2_50_test_set_v7, batch_size=8, shuffle=False)
B2_50_test_loader_v8 = torch.utils.data.DataLoader(B2_50_test_set_v8, batch_size=8, shuffle=False)
B2_50_test_loader_v9 = torch.utils.data.DataLoader(B2_50_test_set_v9, batch_size=8, shuffle=False)
B2_50_test_loader_v10 = torch.utils.data.DataLoader(B2_50_test_set_v10, batch_size=8, shuffle=False)
B2_50_test_loader_v11 = torch.utils.data.DataLoader(B2_50_test_set_v11, batch_size=8, shuffle=False)
B2_50_test_loader_v12 = torch.utils.data.DataLoader(B2_50_test_set_v12, batch_size=8, shuffle=False)
B2_50_test_loader_v13 = torch.utils.data.DataLoader(B2_50_test_set_v13, batch_size=8, shuffle=False)
B2_50_test_loader_v14 = torch.utils.data.DataLoader(B2_50_test_set_v14, batch_size=8, shuffle=False)
B2_50_test_loader_v15 = torch.utils.data.DataLoader(B2_50_test_set_v15, batch_size=8, shuffle=False)
B2_50_test_loader_v16 = torch.utils.data.DataLoader(B2_50_test_set_v16, batch_size=8, shuffle=False)
B2_50_test_loader_v17 = torch.utils.data.DataLoader(B2_50_test_set_v17, batch_size=8, shuffle=False)
B2_50_test_loader_v18 = torch.utils.data.DataLoader(B2_50_test_set_v18, batch_size=8, shuffle=False)
B2_50_test_loader_v19 = torch.utils.data.DataLoader(B2_50_test_set_v19, batch_size=8, shuffle=False)
B2_50_test_loader_v20 = torch.utils.data.DataLoader(B2_50_test_set_v20, batch_size=8, shuffle=False)

test_list = [B2_230_test_loader_v1, B2_230_test_loader_v2, B2_230_test_loader_v3, B2_230_test_loader_v4, B2_230_test_loader_v5,
             B2_230_test_loader_v6, B2_230_test_loader_v7, B2_230_test_loader_v8, B2_230_test_loader_v9, B2_230_test_loader_v10,
             B2_230_test_loader_v11, B2_230_test_loader_v12, B2_230_test_loader_v13, B2_230_test_loader_v14, B2_230_test_loader_v15,
             B2_230_test_loader_v16, B2_230_test_loader_v17, B2_230_test_loader_v18, B2_230_test_loader_v19, B2_230_test_loader_v20]

test_list_180 = [B2_180_test_loader_v1, B2_180_test_loader_v2, B2_180_test_loader_v3, B2_180_test_loader_v4, B2_180_test_loader_v5,
                 B2_180_test_loader_v6, B2_180_test_loader_v7, B2_180_test_loader_v8, B2_180_test_loader_v9, B2_180_test_loader_v10,
                 B2_180_test_loader_v11, B2_180_test_loader_v12, B2_180_test_loader_v13, B2_180_test_loader_v14, B2_180_test_loader_v15,
                 B2_180_test_loader_v16, B2_180_test_loader_v17, B2_180_test_loader_v18, B2_180_test_loader_v19, B2_180_test_loader_v20]

test_list_50 = [B2_50_test_loader_v1, B2_50_test_loader_v2, B2_50_test_loader_v3, B2_50_test_loader_v4, B2_50_test_loader_v5,
                B2_50_test_loader_v6, B2_50_test_loader_v7, B2_50_test_loader_v8, B2_50_test_loader_v9, B2_50_test_loader_v10,
                B2_50_test_loader_v11, B2_50_test_loader_v12, B2_50_test_loader_v13, B2_50_test_loader_v14, B2_50_test_loader_v15,
                B2_50_test_loader_v16, B2_50_test_loader_v17, B2_50_test_loader_v18, B2_50_test_loader_v19, B2_50_test_loader_v20]


#---------------------------------------------------
#
# exp 2 Online Fine-Tuning
#
#----------------------------------------------------
#----------------------------------------------------
# make wrong 
#----------------------------------------------------

embedding_net = resnet50(pretrained = True).to(config.device)
embedding_net_feature_layer = torch.nn.Sequential(*list(embedding_net.children())[:-1])
model = TripletNet(embedding_net_feature_layer)
model = torch.load(config.exp1_result)
model.cuda()

# prototype process
print('processing prototype...')
prototype_230_feature = get_prototype_feature(B2_230_prototype_loader, model)

# test stage
print('\ntest B2_230_test_loader_v1')
accuracy_cosine, accuracy_euclidean, f1_cosine, f1_euclidean, _, euclidean_wrong= test_epoch(B2_230_test_loader_v1, model, prototype_230_feature)
print('Test set: F1 cosine: {:.4f}'.format(f1_cosine))
print('Test set: F1 euclidean: {:.4f}'.format(f1_euclidean))
print()

with open("../new_wrong_item/v1_wrong", "wb") as fp:
    pickle.dump(euclidean_wrong, fp)

with open("../new_wrong_item/v1_wrong", "rb") as fp:
    euclidean_wrong_v1 = pickle.load(fp)



#----------------------------------------------------
# B2 230 class 
#----------------------------------------------------

B2_230_set = datasets.ImageFolder(root=config.B2_230, transform=data_transform)
euclidean_wrong_v1_dataset = WrongTripletBlister(B2_230_set, 230, euclidean_wrong_v1, B2_230_shuffled[10:20])
euclidean_wrong_v1_loader = torch.utils.data.DataLoader(euclidean_wrong_v1_dataset, batch_size=8, shuffle=True)

# -------------------------------------------------
#   triplet exp2 tune
# -------------------------------------------------

best_f1_cosine = 0
n_epochs = config.epochs
loss_fn = TripletLoss(config.margin)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('training...')
for epoch in range(n_epochs):
    # Train stage
    train_loss = train_epoch(euclidean_wrong_v1_loader, model, loss_fn, optimizer, cuda)
    print('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

torch.save(model, config.new2_triplet_tune_wrong + 'v1.pt')



for i in range(1, 19):
    embedding_net = resnet50(pretrained = True).to(config.device)
    embedding_net_feature_layer = torch.nn.Sequential(*list(embedding_net.children())[:-1])
    model = TripletNet(embedding_net_feature_layer)
    model = torch.load(config.new2_triplet_tune_wrong + 'v{}.pt'.format(i))
    model.cuda()

    # prototype process
    print('processing prototype...')
    prototype_230_feature = get_prototype_feature(B2_230_prototype_loader, model)
    
    print('\ntest B2_230_test_loader_v{}'.format(i+1))
    accuracy_cosine, accuracy_euclidean, f1_cosine, f1_euclidean, _, euclidean_wrong= test_epoch(test_list[i], model, prototype_230_feature)
    print('Test set: F1 cosine: {:.4f}'.format(f1_cosine))
    print('Test set: F1 euclidean: {:.4f}'.format(f1_euclidean))
    print()

    with open("../new_wrong_item/v{}_wrong".format(i + 1), "wb") as fp:
        pickle.dump(euclidean_wrong, fp)

    with open("../new_wrong_item/v{}_wrong".format(i + 1), "rb") as fp:
        euclidean_wrong = pickle.load(fp)

    #----------------------------------------------------
    # B2 230 class 
    #----------------------------------------------------

    B2_230_set = datasets.ImageFolder(root=config.B2_230, transform=data_transform)
    euclidean_wrong_dataset = WrongTripletBlister(B2_230_set, 230, euclidean_wrong, B2_230_shuffled[10:(i+2)*10 ])
    euclidean_wrong_loader = torch.utils.data.DataLoader(euclidean_wrong_dataset, batch_size=6, shuffle=True)

    # -------------------------------------------------
    #   triplet exp2 tune
    # -------------------------------------------------

    best_f1_cosine = 0
    n_epochs = config.epochs
    loss_fn = TripletLoss(config.margin)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print('training...')
    for epoch in range(n_epochs):
        # Train stage
        train_loss = train_epoch(euclidean_wrong_loader, model, loss_fn, optimizer, cuda)
        print('Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

    torch.save(model, config.new2_triplet_tune_wrong + 'v{}.pt'.format(i + 1))




# -------------------------------------------------
#
# test  
#
#----------------------------------------------------

embedding_net = resnet50(pretrained = True).to(config.device)
embedding_net_feature_layer = torch.nn.Sequential(*list(embedding_net.children())[:-1])
model = TripletNet(embedding_net_feature_layer)
model = torch.load(config.exp1_result)
model.cuda()
# prototype process
print('processing prototype...')
prototype_230_feature = get_prototype_feature(B2_230_prototype_loader, model)

for i in range(0, 20):

    # test stage
    print('\ntest B2_180_test_loader_v{}'.format(i+1))
    accuracy_cosine, accuracy_euclidean, f1_cosine, f1_euclidean, _, euclidean_wrong= test_epoch(test_list_180[i], model, prototype_230_feature)
    print('Test set: F1 cosine: {:.4f}'.format(f1_cosine))
    print('Test set: F1 euclidean: {:.4f}'.format(f1_euclidean))
    print()

    # test stage
    print('\ntest B2_50_test_loader_v{}'.format(i+1))
    accuracy_cosine, accuracy_euclidean, f1_cosine, f1_euclidean, _, euclidean_wrong= test_epoch(test_list_50[i], model, prototype_230_feature)
    print('Test set: F1 cosine: {:.4f}'.format(f1_cosine))
    print('Test set: F1 euclidean: {:.4f}'.format(f1_euclidean))
    print()

