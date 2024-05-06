import torch

#-------
# device
#-------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#-----------
# parameter
#-----------
img_size = 412
batch_size = 1
val_batch_size = 1
epochs = 50
random_seed = 4999
B1_type_size = 105
B2_type_size = 210
prototype_size = 10
test_img_size = 10

train_img_size = 60
tune_img_size = 20

margin = 1.
lr = 1e-3

#------
# path
#------
exp1_result = '../exp1_result/triplet_exp1.pt'
new2_triplet_tune_wrong = '../new_tune_wrong_result/triplet_tune_wrong_'  # NEW 最新的

B1_180 = '../datasets/B1/180'
B1_50  = '../datasets/B1/50'

B2_180 = '../datasets/B2/180'
B2_50  = '../datasets/B2/50'

B2_230 = '../datasets/B2/230'












