from networks.RNN_classifier import UserEmbeddingExtractor
from dataset.dataset import NetFlowDatasetAssessment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import pickle


def compute_cosine_distance(embedding1, embedding2):
    norm1 = F.normalize(embedding1, p=2, dim=1)
    norm2 = F.normalize(embedding2, p=2, dim=1)

    cosine_similarity = torch.sum(norm1 * norm2, dim=1)

    cosine_distance = 1 - cosine_similarity

    return cosine_distance.item()

def get_results():
    warnings.filterwarnings("ignore", category=FutureWarning, message="'T' is deprecated and will be removed in a future version")

    device = 'cuda'
    embeddings_model = UserEmbeddingExtractor().to(device)

    checkpoint_path = "models/rnn/rnn_epoch=490.pth"  # Replace with your checkpoint path
    state_dict = torch.load(checkpoint_path)
    embeddings_model.load_state_dict(state_dict)

    train_dataset = NetFlowDatasetAssessment('dataset/train', examples_per_user=5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True,
    )

    results = []
    for contents_user1, contents_user2, targets in tqdm(train_dataset):
        # print(len(contents_user1))
        # print(len(contents_user2))
        # print(targets)
        for content_user1, content_user2, target in zip(contents_user1, contents_user2, targets):
            embedding_user1 = embeddings_model(content_user1)
            embedding_user2 = embeddings_model(content_user2)

            score = compute_cosine_distance(embedding_user1, embedding_user2)

            results.append({"score": score, "target": target})

    with open("results.pkl", "wb") as file:
        pickle.dump(results, file)

def compute_metrics(results_path, thresholds):
    with open(results_path, "rb") as file:
        results_list = pickle.load(file)

    correct = 0
    total = 0
    acc_list = []
    for threshold in thresholds:
        for result in results_list:
            if result['score'] >= threshold:
                predicted_target = 0 # same 
            else:
                predicted_target = 1
            if predicted_target == result['target']:
                correct += 1
            total += 1
        acc_list.append(correct/total)

    return acc_list




if __name__ == '__main__':

    get_results()

    # results_path = 'results.pkl'
    # thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 1]
    # acc_list = compute_metrics(results_path=results_path, thresholds=thresholds)
    # print(acc_list)