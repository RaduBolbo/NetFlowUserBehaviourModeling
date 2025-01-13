import json
import socket
from typing import Tuple, Optional
import warnings
import os
import torch
import torch.nn as nn
import hashlib
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataset.dataset import NetFlowDatasetClassification
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.tokenizer.tokenizers import RNNAggregator, InputNormalizer
import torch.nn.functional as F


class UserEmbeddingExtractor(nn.Module):
    def __init__(self, device='cuda', num_classes=200, input_dim=256, hidden_dim=128, lstm_layers=2):
        super(UserEmbeddingExtractor, self).__init__()
        self.input_normalizer = InputNormalizer(10, device)
        self.rnn_aggregator = RNNAggregator(50, 256, 2).to(device)
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False).to(device)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        ) .to(device)

    def forward(self, content_user, return_embeddings=False):
        # PART 1: feature extraction
        feature_vectors = []
        #print('content_user: ', content_user)
        for moment in content_user:
            #print('moment: ', moment)
            outputs = []
            for event in moment["content"]:
                #print('event: ', event)
                outputs.append(self.input_normalizer(event, moment['day_of_week'], int(moment['hour'].split(':')[0])))
            #print(len(outputs))
            outputs = torch.stack(outputs)
            #print('outputs.shape: ', outputs.shape)
            aggregated_features = self.rnn_aggregator(outputs.float())
            #print('aggregated_features.shape: ', aggregated_features.shape)
            feature_vectors.append(aggregated_features)
        feature_vector = torch.stack(feature_vectors)

        # PART 2: classification
        feature_vector = feature_vector.unsqueeze(0)
        #batch_size = feature_vector.size(0)
        lstm_out, _ = self.lstm(feature_vector)  

        last_hidden_state = lstm_out[:, -1, :] 

        logits = self.fc(last_hidden_state)

        if return_embeddings: # embeddings are stored in the last hidden layer
            embeddings = self.fc[0](last_hidden_state)  # first FC layer
            embeddings = self.fc[1](embeddings)  # apply ReLU
            return embeddings
        else:
            logits = self.fc(last_hidden_state)
            return logits

# **** This works very good for sequences the same length
# class UserBooleanClassifier(nn.Module):
#     def __init__(self, device='cuda', input_dim=256, hidden_dim=128, lstm_layers=2):
#         super(UserBooleanClassifier, self).__init__()
#         self.input_normalizer = InputNormalizer(10, device)
#         self.rnn_aggregator = RNNAggregator(50, 256, 2).to(device)

#         self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False).to(device)
#         self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False).to(device)

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 2)  # boolobean classifier
#         ).to(device)

#     def extract_features(self, content_user):
#         feature_vectors = []
#         for moment in content_user:
#             outputs = []
#             for event in moment["content"]:
#                 outputs.append(self.input_normalizer(event, moment['day_of_week'], int(moment['hour'].split(':')[0])))
#             outputs = torch.stack(outputs)
#             aggregated_features = self.rnn_aggregator(outputs.float())
#             feature_vectors.append(aggregated_features)
#         feature_vector = torch.stack(feature_vectors)
#         return feature_vector

#     def forward(self, content_user1, content_user2):
#         print('len(content_user1): ', len(content_user1))
#         print('len(content_user2): ', len(content_user2))
#         feature_vector1 = self.extract_features(content_user1)
#         feature_vector2 = self.extract_features(content_user2)

#         print('feature_vector1.shape: ', feature_vector1.shape)
#         print('feature_vector2.shape: ', feature_vector2.shape)
#         feature_vector1 = feature_vector1.unsqueeze(0)  # Add batch dimension
#         feature_vector2 = feature_vector2.unsqueeze(0)  # Add batch dimension

#         lstm_out1, _ = self.lstm1(feature_vector1)
#         lstm_out2, _ = self.lstm2(feature_vector2)

#         last_hidden_state1 = lstm_out1[:, -1, :]
#         last_hidden_state2 = lstm_out2[:, -1, :]

#         combined_features = torch.cat([last_hidden_state1, last_hidden_state2], dim=1)

#         logits = self.fc(combined_features)
#         return logits


# class FCNNAggregator(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FCNNAggregator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, output_dim)

#     def forward(self, input_vectors):
#         x = self.fc1(input_vectors)
#         return x

class FCNNAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNNAggregator, self).__init__()
        self.fc1 = nn.Linear(input_dim, max(output_dim//2, 1))
        self.bn1 = nn.LayerNorm(max(output_dim//2, 1))
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(max(output_dim//2, 1), output_dim)
        self.bn2 = nn.LayerNorm(output_dim)

    def forward(self, input_vectors):
        x = self.fc1(input_vectors)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UserBooleanClassifier(nn.Module):
    #def __init__(self, device='cuda', input_dim=256, hidden_dim=128, lstm_layers=2, long_sequence_skip=10, aggregator_type='fcnn'):
    def __init__(self, device='cuda', input_dim=256, hidden_dim=6, lstm_layers=2, long_sequence_skip=10, aggregator_type='fcnn'):
        super(UserBooleanClassifier, self).__init__()
        self.input_normalizer_long_sequence = InputNormalizer(10, device)
        self.input_normalizer_short_sequence = InputNormalizer(10, device)
        self.aggregator_type = aggregator_type
        #self.aggregator_dim = 256 # **** intiial
        self.aggregator_dim = 4 # *** itris the input_dim
        self.aggregator_dim = 1 # *** TO DELETE
        if aggregator_type == 'rnn':
            #self.aggregator = RNNAggregator(50, 256, 2).to(device) # original sizes
            self.aggregator = RNNAggregator(50, self.aggregator_dim, 2).to(device)
        elif aggregator_type == 'fcnn':
            #self.aggregator = FCNNAggregator(50, 256).to(device) # original sizes
            self.aggregator = FCNNAggregator(self.aggregator_dim, self.aggregator_dim).to(device)
        self.long_sequence_skip = long_sequence_skip

        self.lstm_long_sequence = nn.LSTM(self.aggregator_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False).to(device)
        self.lstm_short_sequence = nn.LSTM(self.aggregator_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False).to(device)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.aggregator_dim),
            nn.LayerNorm(self.aggregator_dim),
            nn.LeakyReLU(),
            #nn.Dropout(0.5),
            nn.Linear(self.aggregator_dim, 2)  # boolobean classifier
        ).to(device)

    # def extract_features(self, content_user, normalizer, skip): # **** IDEE: EXTRACT FEATURES PE SARITE: PE O PERIOADA MAI LUNGA, DAR CU GAURI (fie gauri de cateva frameuri fie la fiecare x frameuri se iau doar cateva evenimente)
    #     feature_vectors = []
    #     for moment in content_user:
    #         outputs = []
    #         for event in moment["content"]:
    #             outputs.append(normalizer(event, moment['day_of_week'], int(moment['hour'].split(':')[0])))
    #         outputs = torch.stack(outputs)
    #         aggregated_features = self.rnn_aggregator(outputs.float())
    #         feature_vectors.append(aggregated_features)
    #     feature_vector = torch.stack(feature_vectors)
    #     return feature_vector

    def extract_features(self, content_user, normalizer, skip=0):
        feature_vectors = []
        
        interfaces = {}
        for idx, moment in enumerate(content_user):
            if skip > 0 and idx % (skip) != 0:
                continue
            
            outputs = []
            for event in moment["content"]:
                if event["input_interface"] not in interfaces:
                    interfaces[event["input_interface"]] = 1
                else:
                    interfaces[event["input_interface"]] += 1
                outputs.append(normalizer(event, moment['day_of_week'], int(moment['hour'].split(':')[0])))
            
            if self.aggregator_type == 'rnn':
                outputs = torch.stack(outputs) # stack them and then send them to the aggregator
                aggregated_features = self.aggregator(outputs.float())
            elif self.aggregator_type == 'fcnn':
                # V1) 
                # features = []
                # for output in outputs:
                #     features.append(self.aggregator(output.float())) # first send them to the agregator one at a time, then average them
                
                # aggregated_features = torch.mean(torch.stack(features), dim=0)
                
                # V2)
                aggregated_features = self.aggregator(torch.mean(torch.stack(outputs).float(), dim=0))
            
            feature_vectors.append(aggregated_features)
        print('interfaces: ', interfaces)

        feature_vector = torch.stack(feature_vectors)
        return feature_vector

    def forward(self, content_user1, content_user2):
        #print('len(content_user1): ', len(content_user1))
        #print('len(content_user2): ', len(content_user2))
        feature_vector1 = self.extract_features(content_user1, self.input_normalizer_long_sequence, skip=self.long_sequence_skip)
        feature_vector2 = self.extract_features(content_user2, self.input_normalizer_short_sequence, skip=0)

        #print('feature_vector1.shape: ', feature_vector1.shape)
        #print('feature_vector2.shape: ', feature_vector2.shape)
        feature_vector1 = feature_vector1.unsqueeze(0)  # Add batch dimension
        feature_vector2 = feature_vector2.unsqueeze(0)  # Add batch dimension

        lstm_out1, _ = self.lstm_long_sequence(feature_vector1)
        lstm_out2, _ = self.lstm_short_sequence(feature_vector2)

        #print('lstm_out1.shape: ', lstm_out1.shape)
        #print('lstm_out2.shape: ', lstm_out2.shape)

        last_hidden_state1 = lstm_out1[:, -1, :]
        last_hidden_state2 = lstm_out2[:, -1, :]

        #combined_features = torch.cat([last_hidden_state1, last_hidden_state2], dim=1)

        # normalize at concatenation
        combined_features = torch.cat([
            F.normalize(last_hidden_state1, p=2, dim=1),
            F.normalize(last_hidden_state2, p=2, dim=1)
        ], dim=1)

        logits = self.fc(combined_features)

        for name, param in self.named_parameters():
            if "embedding" in name:  # Replace "embedding" with the actual name of your embedding layer
                print(f"Gradients for {name}: {param.grad}")
            if "fc" in name:  # Replace "embedding" with the actual name of your embedding layer
                print(f"Gradients for {name}: {param.grad}")


        return logits

if __name__ == "__main__":

    device = 'cuda'
    train_dataset = NetFlowDatasetClassification('dataset/train')
    content_user, target = train_dataset[0]

    classifier_model = UserEmbeddingExtractor()

    prediction = classifier_model(content_user)
    print('prediction.shape: ', prediction.shape)





