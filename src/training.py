from networks.RNN_classifier import UserEmbeddingExtractor, UserBooleanClassifier
from dataset.dataset import NetFlowDatasetClassification, NetFlowDatasetBooleanClassificator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="'T' is deprecated and will be removed in a future version")

def train_embedder():
    train_dataset = NetFlowDatasetClassification('dataset/train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True,
    )

    #lr = 1e-3
    #lr = 1e-4 # prost
    lr = 1e-5

    device = 'cuda'
    classifier_model = UserEmbeddingExtractor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=lr)

    # Training loop
    num_epochs = 500
    for epoch in range(num_epochs):
        total_loss = 0.0

        for content_user, targets in tqdm(train_dataset):
            one_hot_targets = F.one_hot(torch.tensor([targets]), num_classes=classifier_model.num_classes).float().to(device)

            batch_logits = classifier_model(content_user)

            loss = criterion(batch_logits, one_hot_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {(total_loss/len(train_dataset)):.4f}")
        if epoch % 10 == 0:
            torch.save(classifier_model.state_dict(), f"models/rnn/rnn_epoch={epoch}.pth")

def train_boolean_classificator(checkpoint_path=None):
    examples_per_user = 5
    #long_seq = 6 # half an hour
    #long_seq = 12 # one hour
    #long_seq = 144  # half a day
    long_seq = 288 # one day
    #long_seq = 2016 # one week

    short_seq = 6 # half an hout
    train_dataset = NetFlowDatasetBooleanClassificator('dataset/train', sequence_length_user1=long_seq, sequence_length_user2=short_seq, examples_per_user=examples_per_user)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=1, 
    #     shuffle=True,
    #     num_workers=4
    # )

    #lr = 1e-3
    lr = 1e-4 
    #lr = 1e-5
    #lr = 1e-6
    #lr = 1e-7

    device = 'cuda'
    #classifier_model = UserBooleanClassifier(input_dim=256, hidden_dim=128, lstm_layers=2, long_sequence_skip=10).to(device) # tried
    classifier_model = UserBooleanClassifier(input_dim=256, hidden_dim=128, lstm_layers=2, long_sequence_skip=12).to(device) # 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=lr) # **** CHANGE

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        classifier_model.load_state_dict(state_dict)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for user1_samples, user2_samples, targets in tqdm(train_dataset):

            logits_list = []
            targets_list = []
            for user1_sample, user2_sample, target in zip(user1_samples, user2_samples, targets):
                logits = classifier_model(user1_sample, user2_sample)
                logits_list.append(logits[0])
                target_onehot = F.one_hot(torch.tensor([target]), num_classes=2).float().to(device)
                targets_list.append(target_onehot[0])

            batch_logits = torch.stack(logits_list)
            one_hot_targets = torch.stack(targets_list)
            loss = criterion(batch_logits, torch.argmax(one_hot_targets, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(batch_logits, dim=1)
            true_classes = torch.argmax(one_hot_targets, dim=1)
            total_correct += (predictions == true_classes).sum().item()
            total_samples += true_classes.size(0)

            true_positive += ((predictions == 1) & (true_classes == 1)).sum().item()
            false_positive += ((predictions == 1) & (true_classes == 0)).sum().item()
            false_negative += ((predictions == 0) & (true_classes == 1)).sum().item()
            true_negative += ((predictions == 0) & (true_classes == 0)).sum().item()

        epoch_accuracy = total_correct / total_samples
        tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0

        print(f"Epoch Loss: {total_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}")
        print(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {(total_loss/(len(train_dataset) * examples_per_user)):.4f}")

        torch.save(classifier_model.state_dict(), f"models/rnn/rnn_epoch={epoch}_loss_{total_loss/(len(train_dataset) * examples_per_user)}_acc_{epoch_accuracy}.pth")


if __name__ == '__main__':

    # V1) The embedder approach
    #train_embedder()

    #checkpoint_path = r'models\rnn\rnn_0_probably_good.pth'
    checkpoint_path = None

    train_boolean_classificator(checkpoint_path)