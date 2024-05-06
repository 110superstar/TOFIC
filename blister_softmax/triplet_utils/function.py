import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

#----------------------------------------------------
# function
#----------------------------------------------------
def get_prototype_feature(prototype_loader, model):
    model.eval()
    with torch.no_grad():
        all_label_prototype_feature = {}
        for label, each_label_img in enumerate(tqdm(prototype_loader)):
            label_img_feature = []
            for each_img  in each_label_img:
                label_img_feature.append(model(each_img.cuda()).squeeze())
            all_label_prototype_feature[label] = sum(label_img_feature) / len(label_img_feature)
        return all_label_prototype_feature

def test_epoch(test_loader, model, prototype_feature):
    model.eval()
    pred_cosine = np.array([])
    pred_euclidean = np.array([])
    label = np.array([])

    with torch.no_grad():

        for batch_idx, (batch_data, batch_target, _) in enumerate(tqdm(test_loader)):
            for each_data, each_target in zip(batch_data, batch_target):
                data_feature = model(each_data.cuda().unsqueeze(0)).squeeze()
                max_cosine_similarity = 0
                min_euclidean_distance = 9999
                for label_name in prototype_feature:
                    
                    label_feature = prototype_feature[label_name]
                    label_cosine_similarity = cosine_similarity(data_feature.unsqueeze(0).cpu(), label_feature.unsqueeze(0).cpu()) #? nn cuda gpu
                    label_euclidean_distance = np.linalg.norm(data_feature.cpu()-label_feature.cpu())

                    if label_cosine_similarity > max_cosine_similarity:
                        max_label_cosine = label_name
                        max_cosine_similarity = label_cosine_similarity

                    if label_euclidean_distance < min_euclidean_distance:
                        min_label_euclidean = label_name
                        min_euclidean_distance = label_euclidean_distance

                pred_cosine = np.append(pred_cosine, max_label_cosine)
                pred_euclidean = np.append(pred_euclidean, min_label_euclidean)
                label = np.append(label, each_target)
    
    accuracy_euclidean = sum(pred_euclidean == label) / len(label)
    accuracy_cosine = sum(pred_cosine == label) / len(label)
    f1_cosine = metrics.f1_score(pred_cosine, label, average='macro')
    f1_euclidean = metrics.f1_score(pred_euclidean, label, average='macro')
    return accuracy_cosine, accuracy_euclidean, f1_cosine, f1_euclidean

def train_epoch(train_loader, model, loss_fn, optimizer, cuda):

    total_loss = 0
    model.train()
    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / (batch_idx + 1)
