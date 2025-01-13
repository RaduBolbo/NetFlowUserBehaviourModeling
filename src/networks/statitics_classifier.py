import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# class UserBooleanClassifierStatistics(nn.Module):
#     def __init__(self):
#         super(UserBooleanClassifierStatistics, self).__init__()

#         # Define the fully connected neural network for classification
#         self.classifier = nn.Sequential(
#             nn.Linear(10, 8),  # Updated to 10 features as input
#             nn.ReLU(),
#             nn.Linear(8, 2),
#         )

#     def extract_features(self, content_user1, content_user2):
#         def compute_top_k_overlap(list1, list2, k):
#             list1 = list(list1)
#             list2 = list(list2)
#             top_k_1 = [item for item, _ in list1[:k]]
#             top_k_2 = [item for item, _ in list2[:k]]
#             return len(set(top_k_1) & set(top_k_2)) / k

#         ip_count_user1 = pd.Series(
#             [entry['dest_ip'] for seq in content_user1 for entry in seq['content']]
#         ).value_counts()
#         ip_count_user2 = pd.Series(
#             [entry['dest_ip'] for seq in content_user2 for entry in seq['content']]
#         ).value_counts()
#         top_3_ip_overlap = compute_top_k_overlap(ip_count_user1.items(), ip_count_user2.items(), 3)
#         top_5_ip_overlap = compute_top_k_overlap(ip_count_user1.items(), ip_count_user2.items(), 5)
#         top_10_ip_overlap = compute_top_k_overlap(ip_count_user1.items(), ip_count_user2.items(), 10)

#         port_count_user1 = pd.Series(
#             [entry['source_port'] for seq in content_user1 for entry in seq['content']]
#         ).value_counts()
#         port_count_user2 = pd.Series(
#             [entry['source_port'] for seq in content_user2 for entry in seq['content']]
#         ).value_counts()
#         top_3_port_overlap = compute_top_k_overlap(port_count_user1.items(), port_count_user2.items(), 3)

#         interface_user1 = pd.Series(
#             [entry['input_interface'] for seq in content_user1 for entry in seq['content']]
#         ).value_counts()
#         interface_user2 = pd.Series(
#             [entry['input_interface'] for seq in content_user2 for entry in seq['content']]
#         ).value_counts()
#         top_2_interface_overlap = compute_top_k_overlap(interface_user1.items(), interface_user2.items(), 2)

#         protocol_user1 = pd.Series(
#             [entry['protocol'] for seq in content_user1 for entry in seq['content']]
#         ).value_counts(normalize=True)
#         protocol_user2 = pd.Series(
#             [entry['protocol'] for seq in content_user2 for entry in seq['content']]
#         ).value_counts(normalize=True)
#         protocol_similarity = 1 - np.abs(protocol_user1 - protocol_user2).sum()

#         user1_hourly_traffic = pd.Series(
#             [seq['hour'] for seq in content_user1]
#         ).value_counts()
#         user2_hourly_traffic = pd.Series(
#             [seq['hour'] for seq in content_user2]
#         ).value_counts()
#         traffic_similarity = np.abs(user1_hourly_traffic - user2_hourly_traffic).sum() / max(
#             user1_hourly_traffic.sum(), user2_hourly_traffic.sum()
#         )

#         user1_total_bytes = sum(entry['byte_count'] for seq in content_user1 for entry in seq['content'])
#         user2_total_bytes = sum(entry['byte_count'] for seq in content_user2 for entry in seq['content'])
#         byte_count_ratio = user2_total_bytes / user1_total_bytes if user1_total_bytes > 0 else 0

#         user1_unique_ips = len(ip_count_user1)
#         user2_unique_ips = len(ip_count_user2)
#         unique_ip_similarity = user2_unique_ips / user1_unique_ips if user1_unique_ips > 0 else 0

#         user1_average_packet_size = np.mean([entry['byte_count'] / entry['packet_count'] for seq in content_user1 for entry in seq['content'] if entry['packet_count'] > 0])
#         user2_average_packet_size = np.mean([entry['byte_count'] / entry['packet_count'] for seq in content_user2 for entry in seq['content'] if entry['packet_count'] > 0])
#         average_packet_size_similarity = 1 - abs(user1_average_packet_size - user2_average_packet_size) / max(user1_average_packet_size, user2_average_packet_size) if max(user1_average_packet_size, user2_average_packet_size) > 0 else 0

#         features = torch.tensor([
#             top_3_ip_overlap, top_5_ip_overlap, top_10_ip_overlap,
#             top_3_port_overlap, top_2_interface_overlap,
#             protocol_similarity, 1 - traffic_similarity,
#             byte_count_ratio, unique_ip_similarity,
#             average_packet_size_similarity
#         ], dtype=torch.float32)

#         return features

#     def forward(self, content_user1, content_user2):
#         features = self.extract_features(content_user1, content_user2)
#         output = self.classifier(features)
#         return output.unsqueeze(0)

class UserBooleanClassifierStatistics(nn.Module):
    def __init__(self):
        super(UserBooleanClassifierStatistics, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(7, 8),  # 7 features as input
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def extract_features(self, content_user1, content_user2):
        # print('content_user1: ', type(content_user1))
        # print('content_user2: ', type(content_user2))

        # print('len(content_user1): ', len(content_user1))
        # print('len(content_user2): ', len(content_user2))

        # print('type(content_user1[0]): ', type(content_user1[0]))
        # print('type(content_user2[0]): ', type(content_user2[0]))

        # print('content_user1[0].keys: ', content_user1[0].keys())
        # print('type(content_user1[0]["content"]): ', type(content_user1[0]['content']))
        # print('len(content_user1[0]["content"]): ', len(content_user1[0]['content']))
        # print('content_user1[0]["content"][0].keys(): ', content_user1[0]['content'][0].keys())

        def compute_top_k_overlap(list1, list2, k):
            list1 = list(list1)
            list2 = list(list2)
            top_k_1 = [item for item, _ in list1[:k]]
            top_k_2 = [item for item, _ in list2[:k]]
            return len(set(top_k_1) & set(top_k_2)) / k

        ip_count_user1 = pd.Series(
            [entry['dest_ip'] for seq in content_user1 for entry in seq['content']]
        ).value_counts()
        ip_count_user2 = pd.Series(
            [entry['dest_ip'] for seq in content_user2 for entry in seq['content']]
        ).value_counts()
        top_3_ip_overlap = compute_top_k_overlap(ip_count_user1.items(), ip_count_user2.items(), 3)
        top_5_ip_overlap = compute_top_k_overlap(ip_count_user1.items(), ip_count_user2.items(), 5)
        top_10_ip_overlap = compute_top_k_overlap(ip_count_user1.items(), ip_count_user2.items(), 10)

        port_count_user1 = pd.Series(
            [entry['source_port'] for seq in content_user1 for entry in seq['content']]
        ).value_counts()
        port_count_user2 = pd.Series(
            [entry['source_port'] for seq in content_user2 for entry in seq['content']]
        ).value_counts()
        top_3_port_overlap = compute_top_k_overlap(port_count_user1.items(), port_count_user2.items(), 3)

        interface_user1 = pd.Series(
            [entry['input_interface'] for seq in content_user1 for entry in seq['content']]
        ).value_counts()
        interface_user2 = pd.Series(
            [entry['input_interface'] for seq in content_user2 for entry in seq['content']]
        ).value_counts()
        top_2_interface_overlap = compute_top_k_overlap(interface_user1.items(), interface_user2.items(), 2)

        protocol_user1 = pd.Series(
            [entry['protocol'] for seq in content_user1 for entry in seq['content']]
        ).value_counts(normalize=True)
        protocol_user2 = pd.Series(
            [entry['protocol'] for seq in content_user2 for entry in seq['content']]
        ).value_counts(normalize=True)
        protocol_similarity = 1 - np.abs(protocol_user1 - protocol_user2).sum()

        user1_hourly_traffic = pd.Series(
            [seq['hour'] for seq in content_user1]
        ).value_counts()
        user2_hourly_traffic = pd.Series(
            [seq['hour'] for seq in content_user2]
        ).value_counts()
        traffic_similarity = np.abs(user1_hourly_traffic - user2_hourly_traffic).sum() / max(
            user1_hourly_traffic.sum(), user2_hourly_traffic.sum()
        )

        features = torch.tensor([
            top_3_ip_overlap, top_5_ip_overlap, top_10_ip_overlap,
            top_3_port_overlap, top_2_interface_overlap,
            protocol_similarity, 1 - traffic_similarity
        ], dtype=torch.float32)

        return features

    def forward(self, content_user1, content_user2):
        features = self.extract_features(content_user1, content_user2)
        output = self.classifier(features)
        #print('output.shape: ', output.shape)
        #print('output.squeeze(0).shape: ', output.unsqueeze(0).shape)
        return output.unsqueeze(0)

