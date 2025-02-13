import numpy as np
import os
import torch.nn as nn
import torch
import random
import pandas as pd


# class NetFlowDataset(torch.utils.data.Dataset):
#     def __init__(self, input_dir, long_sequence_length=2016, short_sequence_length=6):
#         self.user_paths = []
#         for filename in sorted(os.listdir(input_dir)):
#             filepath = os.path.join(input_dir, filename)
#             self.user_paths.append(filepath)
#         self.long_sequence_length = long_sequence_length
#         self.short_sequence_length = short_sequence_length

#     def __len__(self):
#         return len(self.user_paths)

#     def __getitem__(self, index):
#         same_user = random.choice([True, False])
#         user1 = self.user_paths[index]
#         if same_user:
#             user2 = user1
#             target = 0  # 0 means same
#         else:
#             possible_indices = [i for i in range(len(self.user_paths)) if i != index]
#             user2 = self.user_paths[random.choice(possible_indices)]
#             target = 1  # 1 means different users

#         content_user1 = self._generate_sequence(user1, self.long_sequence_length)
#         content_user2 = self._generate_sequence(user2, self.short_sequence_length)

#         return content_user1, content_user2, target

#     def _generate_sequence(self, filepath, sequence_length):
#         df = pd.read_csv(filepath)
#         if len(df) < sequence_length:
#             raise ValueError(f"File {filepath} does not have enough data to generate a sequence of length {sequence_length}.")

#         start_index = random.randint(0, len(df) - sequence_length)
#         sequence = df.iloc[start_index:start_index + sequence_length]

#         processed_sequence = []
#         for _, row in sequence.iterrows():
#             processed_event = {
#                 "date": f"{row['timestamp'].split()[0]}",
#                 "day_of_week": row["day_of_week"],
#                 "hour": f"{row['hour']}:{row['minute']}",
#                 "content": {
#                     "dest_ip": row["dest_ip"],
#                     "protocol": row["protocol"],
#                     "source_port": row["source_port"],
#                     "destination_port": row["destination_port"],
#                     "input_interface": row["input_interface"],
#                     "output_interface": row["output_interface"],
#                     "packet_count": row["packet_count"],
#                     "byte_count": row["byte_count"]
#                 }
#             }
#             processed_sequence.append(processed_event)

#         return processed_sequence

class NetFlowDatasetSequenceComparison(torch.utils.data.Dataset):
    def __init__(self, input_dir, long_sequence_length=2016, short_sequence_length=6):
        self.user_paths = []
        for filename in sorted(os.listdir(input_dir)):
            filepath = os.path.join(input_dir, filename)
            self.user_paths.append(filepath)
        self.long_sequence_length = long_sequence_length
        self.short_sequence_length = short_sequence_length

    def __len__(self):
        return len(self.user_paths)

    def __getitem__(self, index):
        same_user = random.choice([True, False])
        user1 = self.user_paths[index]
        if same_user:
            user2 = user1
            target = 0  # 0 means same
        else:
            possible_indices = [i for i in range(len(self.user_paths)) if i != index]
            user2 = self.user_paths[random.choice(possible_indices)]
            target = 1  # 1 means different users

        content_user1 = self._generate_sequence(user1, self.long_sequence_length)
        content_user2 = self._generate_sequence(user2, self.short_sequence_length)

        return content_user1, content_user2, target

    def _generate_sequence(self, filepath, sequence_length):
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["time_window"] = df["timestamp"].dt.floor("5T")
        unique_timestamps = df["time_window"].unique()

        if len(unique_timestamps) < sequence_length:
            raise ValueError(f"File {filepath} does not have enough unique timestamps to generate a sequence of length {sequence_length}.")

        start_index = random.randint(0, len(unique_timestamps) - sequence_length)
        selected_timestamps = unique_timestamps[start_index:start_index + sequence_length]
        sequence = df[df["time_window"].isin(selected_timestamps)]

        grouped_sequence = []
        for timestamp in selected_timestamps:
            events_in_window = sequence[sequence["time_window"] == timestamp]
            grouped_event = {
                "date": str(timestamp.date()),
                "day_of_week": events_in_window.iloc[0]["day_of_week"],
                "hour": f"{timestamp.hour}:{timestamp.minute}",
                "content": []
            }

            for _, row in events_in_window.iterrows():
                grouped_event["content"].append({
                    "dest_ip": row["dest_ip"],
                    "protocol": row["protocol"],
                    "source_port": row["source_port"],
                    "destination_port": row["destination_port"],
                    "input_interface": row["input_interface"],
                    "output_interface": row["output_interface"],
                    "packet_count": row["packet_count"],
                    "byte_count": row["byte_count"]
                })

            grouped_sequence.append(grouped_event)

        return grouped_sequence
    
class NetFlowDatasetClassification(torch.utils.data.Dataset):
    def __init__(self, input_dir, sequence_length=6):
        self.user_paths = []
        for filename in sorted(os.listdir(input_dir)):
            filepath = os.path.join(input_dir, filename)
            self.user_paths.append(filepath)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.user_paths)

    def __getitem__(self, index):
        user1 = self.user_paths[index] 

        content_user = self._generate_sequence(user1, self.sequence_length) # every epoch will draw a different chunk out of the file

        return content_user, index 

    def _generate_sequence(self, filepath, sequence_length):
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["time_window"] = df["timestamp"].dt.floor("5T")
        unique_timestamps = df["time_window"].unique()

        if len(unique_timestamps) < sequence_length:
            raise ValueError(f"File {filepath} does not have enough unique timestamps to generate a sequence of length {sequence_length}.")

        start_index = random.randint(0, len(unique_timestamps) - sequence_length)
        selected_timestamps = unique_timestamps[start_index:start_index + sequence_length]
        sequence = df[df["time_window"].isin(selected_timestamps)]

        grouped_sequence = []
        for timestamp in selected_timestamps:
            events_in_window = sequence[sequence["time_window"] == timestamp]
            grouped_event = {
                "date": str(timestamp.date()),
                "day_of_week": events_in_window.iloc[0]["day_of_week"],
                "hour": f"{timestamp.hour}:{timestamp.minute}",
                "content": []
            }

            for _, row in events_in_window.iterrows():
                grouped_event["content"].append({
                    "dest_ip": row["dest_ip"],
                    "protocol": row["protocol"],
                    "source_port": row["source_port"],
                    "destination_port": row["destination_port"],
                    "input_interface": row["input_interface"],
                    "output_interface": row["output_interface"],
                    "packet_count": row["packet_count"],
                    "byte_count": row["byte_count"]
                })

            grouped_sequence.append(grouped_event)

        return grouped_sequence

class NetFlowDatasetAssessment(NetFlowDatasetClassification):
    def __init__(self, input_dir, sequence_length=6, examples_per_user=1):
        super().__init__(input_dir, sequence_length)
        self.examples_per_user = examples_per_user

    def __getitem__(self, index):
        # 1st user
        user1_path = self.user_paths[index]

        user1_samples = [self._generate_sequence(user1_path, self.sequence_length) for _ in range(self.examples_per_user)]

        # decide for each example if the 2nd user will be the same or different
        user2_samples = []
        targets = []
        for _ in range(self.examples_per_user):
            same_user = random.choice([True, False])
            if same_user:
                user2_path = user1_path
                target = 0
            else:
                possible_indices = [i for i in range(len(self.user_paths)) if i != index]
                user2_path = self.user_paths[random.choice(possible_indices)]
                target = 1

            user2_samples.append(self._generate_sequence(user2_path, self.sequence_length))
            targets.append(target)

        return user1_samples, user2_samples, targets


class NetFlowDatasetBooleanClassificator(torch.utils.data.Dataset):
    def __init__(self, input_dir, sequence_length_user1=6, sequence_length_user2=6, examples_per_user=1):
        super().__init__()
        self.user_paths = []
        for filename in sorted(os.listdir(input_dir)):
            filepath = os.path.join(input_dir, filename)
            self.user_paths.append(filepath)
        self.sequence_length_user1 = sequence_length_user1
        self.sequence_length_user2 = sequence_length_user2
        self.examples_per_user = examples_per_user

    def __len__(self):
        return len(self.user_paths)

    def _generate_sequence(self, filepath, sequence_length):
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["time_window"] = df["timestamp"].dt.floor("5T")
        unique_timestamps = df["time_window"].unique()

        if len(unique_timestamps) < sequence_length:
            raise ValueError(f"File {filepath} does not have enough unique timestamps to generate a sequence of length {sequence_length}.")

        start_index = random.randint(0, len(unique_timestamps) - sequence_length)
        selected_timestamps = unique_timestamps[start_index:start_index + sequence_length]
        sequence = df[df["time_window"].isin(selected_timestamps)]

        grouped_sequence = []
        for timestamp in selected_timestamps:
            events_in_window = sequence[sequence["time_window"] == timestamp]
            grouped_event = {
                "date": str(timestamp.date()),
                "day_of_week": events_in_window.iloc[0]["day_of_week"],
                "hour": f"{timestamp.hour}:{timestamp.minute}",
                "content": []
            }

            for _, row in events_in_window.iterrows():
                grouped_event["content"].append({
                    "dest_ip": row["dest_ip"],
                    "protocol": row["protocol"],
                    "source_port": row["source_port"],
                    "destination_port": row["destination_port"],
                    "input_interface": row["input_interface"],
                    "output_interface": row["output_interface"],
                    "packet_count": row["packet_count"],
                    "byte_count": row["byte_count"]
                })

            grouped_sequence.append(grouped_event)

        return grouped_sequence

    def __getitem__(self, index):
        # 1st user
        user1_path = self.user_paths[index]
        user1_samples = [self._generate_sequence(user1_path, self.sequence_length_user1) for _ in range(self.examples_per_user)]

        # 2nd user
        user2_samples = []
        targets = []
        for _ in range(self.examples_per_user):
            same_user = random.choice([True, False])
            if same_user:
                user2_path = user1_path
                target = 0
            else:
                possible_indices = [i for i in range(len(self.user_paths)) if i != index]
                user2_path = self.user_paths[random.choice(possible_indices)]
                target = 1

            user2_samples.append(self._generate_sequence(user2_path, self.sequence_length_user2))
            targets.append(target)

        return user1_samples, user2_samples, targets


if __name__ == '__main__':
    train_dataset = NetFlowDataset('dataset/train')
    content_user1, content_user2, target = train_dataset[0]
    print(content_user1)
    print(content_user2)
    print(target)
            







