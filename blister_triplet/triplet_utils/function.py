import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from statistics import mean
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
                # label_img_feature.append(model.get_embedding(each_img.cuda()).squeeze())
                label_img_feature.append(model.get_embedding(each_img).squeeze())
            all_label_prototype_feature[label] = sum(label_img_feature) / len(label_img_feature)
        return all_label_prototype_feature

def test_epoch(test_loader, model, prototype_feature):
    model.eval()
    pred_cosine = np.array([])
    pred_euclidean = np.array([])
    label = np.array([])

    with torch.no_grad():

        cosine_wrong = []
        euclidean_wrong = []
        for batch_idx, (batch_data, batch_target, b_idx) in enumerate(tqdm(test_loader)):
            for each_data, each_target, idx in zip(batch_data, batch_target, b_idx):
                data_feature = model.get_embedding(each_data.cuda().unsqueeze(0)).squeeze()
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

                if not max_label_cosine == each_target:
                    cosine_wrong.append([each_target, idx, max_label_cosine])
                if not min_label_euclidean == each_target:
                    euclidean_wrong.append([each_target, idx, min_label_euclidean])
                
                pred_cosine = np.append(pred_cosine, max_label_cosine)
                pred_euclidean = np.append(pred_euclidean, min_label_euclidean)
                label = np.append(label, each_target)
        
        
    accuracy_euclidean = sum(pred_euclidean == label) / len(label)
    accuracy_cosine = sum(pred_cosine == label) / len(label)
    consine_result = metrics.f1_score(pred_cosine, label, average=None)
    euclidean_result = metrics.f1_score(pred_euclidean, label, average=None)
    # consine_result = list(filter(lambda a: a !=0, metrics.f1_score(pred_cosine, label, average=None)))
    # euclidean_result = list(filter(lambda a: a !=0, metrics.f1_score(pred_euclidean, label, average=None)))
    if int(len(label) / 10) == 50:
        consine_result = consine_result[-50:]
        euclidean_result = euclidean_result[-50:]
    if int(len(label) / 10) == 180:
        consine_result = consine_result[:180]
        euclidean_result = euclidean_result[:180]
    f1_cosine = mean(consine_result)
    f1_euclidean = mean(euclidean_result)
    return accuracy_cosine, accuracy_euclidean, f1_cosine, f1_euclidean, cosine_wrong, euclidean_wrong

def train_epoch(train_loader, model, loss_fn, optimizer, cuda):

    model.train()
    model.cpu()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        
        if cuda:
            data = tuple(d.cpu() for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        loss_inputs = (o.squeeze() for o in outputs)
        loss_outputs = loss_fn(*loss_inputs)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / (batch_idx + 1)