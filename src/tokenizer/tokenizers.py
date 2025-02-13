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
from tqdm import tqdm
from torch.utils.data import DataLoader



class IPTokenizer:
    filler_ip = 'filler IP'
    def __init__(self, cache_file_path: str):
        """
        Initialize the IPTokenizer with a cache file.
        Loads the IP -> Name table from the specified JSON file.
        """
        self.cache_file_path = cache_file_path
        try:
            with open(self.cache_file_path, 'r') as f:
                self.name_ip_cache = json.load(f)
        except FileNotFoundError:
            self.name_ip_cache = {}  # if file doens't exist, create empty file
        except json.JSONDecodeError:
            raise ValueError("The cache file is not a valid JSON.")

    def ip2name(self, ip: str) -> Optional[Tuple[str, list]]:
        """
        Resolve an IP address to its corresponding name and domains.
        If the IP is not in the cache, resolve it, add it to the cache, and return the result.
        """
        #print('ip: ', ip)
        #print('self.name_ip_cache: ', self.name_ip_cache)
        if ip in self.name_ip_cache:
            return self.name_ip_cache[ip] # if the ip is in the chache, just return the name from the dict
        
        try: # if the ip wa snot found, make a DNS interrogation
            # Reverse DNS lookup
            host_name = socket.gethostbyaddr(ip)[0] # get hostname
            #domains = [host_name] # get domains
            domain_parts = host_name.split('.')[-2:]  
            domains = '.'.join(domain_parts)
            self.name_ip_cache[ip] = domains  # add to chache
            return domains # return the name and domain, keeping 
        except (socket.herror, socket.gaierror):
            # Add unresolved IP with default None values to the cache
            self.name_ip_cache[ip] = (self.filler_ip)
            return None

    def update_table(self):
        """
        Update all entries in the table by resolving the IPs again.
        This ensures that the cache stays updated even if the IPs change over time.
        """
        for ip in list(self.name_ip_cache.keys()): # for each ip in the cache list
            try:
                host_name = socket.gethostbyaddr(ip)[0] # resolve
                #domains = [host_name]
                domain_parts = host_name.split('.')[-2:]  
                domains = '.'.join(domain_parts)
                self.name_ip_cache[ip] = domains # attribute it to the key
            except (socket.herror, socket.gaierror):
                continue  # keep the old value if resolvoing fails
        
        # save to .sjon
        self.save_cache()

    def save_cache(self):
        """
        Save the current cache to the JSON file.
        """
        with open(self.cache_file_path, 'w') as f:
            json.dump(self.name_ip_cache, f, indent=4)


class DNSNameEmbedding(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=5, hidden_dim=10):
        super(DNSNameEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, dns_name):
        char_embeddings = self.char_embedding(dns_name)
        _, hidden_state = self.rnn(char_embeddings)
        dns_embedding = self.linear(hidden_state.squeeze(0))

        return dns_embedding


# class PortTokenizer:
#     def __init__(self, ports=None, embedding_dim=10, file_path=None):
#         """
#         Initializes the PortTokenizer. Loads embeddings from the provided file if it exists;
#         otherwise, computes embeddings for the given ports and saves them to the file.
#         **** embedding here is actually a multidimesnional ID because it is not learned and could be replaced by a simple ID (and then there should be an UNKNOWNED IP ID)
#         """
#         self.embedding_dim = embedding_dim
#         self.file_path = file_path

#         # Validate input
#         if file_path and os.path.exists(file_path):
#             self.load_embeddings()
#         elif ports: # if no file is provided, compute the embeddings for the given ports
#             self.ports = ports
#             self.port_to_idx = {port: idx for idx, port in enumerate(self.ports)} # there is a need for port-index mapping
#             self.embedding_layer = nn.Embedding(len(self.ports), self.embedding_dim)
#             self.save_embeddings()
#         else:
#             raise ValueError("Either 'ports' or 'file_path' must be provided.")
        
#         self.embedding_layer.requires_grad_ = False

#     def _hash_unknowned_ports(self, port, scale=0.1):
#         """
#         Generates a deterministic embedding vector for a given port using its hash.
#         """
#         hash_value = hashlib.md5(str(port).encode()).hexdigest()
        
#         hash_as_ints = [int(hash_value[i:i+2], 16) for i in range(0, len(hash_value), 2)]

#         hash_as_floats = np.array(hash_as_ints[:self.embedding_dim]) / 255.0  # Scale to [0, 1]
#         hash_as_floats = (hash_as_floats * 2) - 1  # Scale to [-1, 1]
        
#         embedding = torch.tensor(hash_as_floats * scale, dtype=torch.float)
        
#         return embedding

#     def save_embeddings(self):
#         embeddings = {}
#         for port, idx in self.port_to_idx.items():
#             embeddings[port] = self.embedding_layer(torch.tensor(idx)).detach().tolist()

#         with open(self.file_path, 'w') as f:
#             json.dump(embeddings, f, indent=4)

#     def load_embeddings(self):
#         if not self.file_path:
#             raise ValueError("No file path provided for loading embeddings.")

#         with open(self.file_path, 'r') as f:
#             embeddings = json.load(f)

#         # Ports and their indices
#         self.ports = list(map(int, embeddings.keys()))
#         self.port_to_idx = {port: idx for idx, port in enumerate(self.ports)}

#         # Embedding layer
#         embedding_weights = torch.tensor([embeddings[str(port)] for port in self.ports])
#         self.embedding_layer = nn.Embedding.from_pretrained(embedding_weights)
#         print(f"Loaded embeddings from {self.file_path}")

#         self.embedding_layer.requires_grad_ = False

#     def get_embedding(self, port):
#         if port not in self.port_to_idx:
#             return self._hash_unknowned_ports(port)
#         idx = self.port_to_idx[port]
#         return self.embedding_layer(torch.tensor(idx))
    

class PortTokenizer:
    def __init__(self, file_path: str, embedding_dim: int = 10):
        self.embedding_dim = embedding_dim
        self.unknown_port_idx = 0

        with open(file_path, 'r') as f:
            ports = list(map(int, json.load(f).keys()))

        self.ports = ports
        self.port_to_idx = {port: idx + 1 for idx, port in enumerate(self.ports)}  # Reserve 0 for unknown
        self.embedding_layer = nn.Embedding(len(self.ports) + 1, self.embedding_dim)

    def get_embedding(self, port: int):
        idx = self.port_to_idx.get(port, self.unknown_port_idx)
        return self.embedding_layer(torch.tensor(idx))


def dns_to_char_indices(dns_name, char_to_idx):
    """
    Converts a DNS name into a list of character indices.
    """
    return torch.tensor([char_to_idx[char] for char in dns_name], dtype=torch.long)

class InterfaceEncoding:
    def __init__(self, num_interfaces, embedding_size):
        self.num_interfaces = num_interfaces
        self.embedding_size = embedding_size

        if embedding_size < num_interfaces:
            raise ValueError("Embedding size must be greater than or equal to the number of unique interfaces.")

    def encode(self, interface):
        one_hot = torch.eye(self.num_interfaces)[interface]
        padded = torch.nn.functional.pad(one_hot, (0, self.embedding_size - self.num_interfaces), value=0)
        return padded

    def batch_encode(self, interfaces):
        return torch.stack([self.encode(i) for i in interfaces])


class ProtocolEncoding:
    def __init__(self, protocols, embedding_size):
        self.protocols = protocols
        self.embedding_size = embedding_size
        self.protocol_to_index = {protocol: i for i, protocol in enumerate(protocols)}

        if embedding_size < len(protocols):
            raise ValueError("Embedding size must be greater than or equal to the number of unique protocols.")

    def encode(self, protocol):
        if protocol not in self.protocol_to_index:
            raise ValueError(f"Unknown protocol: {protocol}")

        one_hot = torch.eye(len(self.protocols))[self.protocol_to_index[protocol]]
        padded = torch.nn.functional.pad(one_hot, (0, self.embedding_size - len(self.protocols)), value=0)
        return padded

    def batch_encode(self, protocols):
        return torch.stack([self.encode(p) for p in protocols])

# class InputNormalizer(nn.Module):
#     '''
#     InputNormalizer normalizes the input data for one communication at a specific timestemp, tokeinzes it and then passes each input through an embedding layer such that it
#     finally obtains an aggregated result
#     '''
#     def __init__(self, embedding_size, device):
#         self.embedding_size = embedding_size
#         self.device = device

#     def forward(self, data):
#         '''
#         data will be a batched tensor with the fields in this order
#         dest_ip,protocol,source_port,destination_port,input_interface,output_interface,packet_count,byte_count,day_of_the_week,hour
#         '''
#         pass

class InputNormalizerNdim(nn.Module):
    '''
    This is intened to pe bart of the NN
    '''
    def __init__(self, embedding_size, device, cache_file_ip="resources/NameIpCacheFile.json", cache_file_ports="resources/port_embeddings.json"):
        super(InputNormalizerNdim, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.ip_tokenizer = IPTokenizer(cache_file_ip)
        self.name_tokeinzer = DNSNameEmbedding(vocab_size=256, embedding_dim=5, hidden_dim=10)
        self.port_tokenizer = PortTokenizer(file_path=cache_file_ports)
        self.protocol_encoder = ProtocolEncoding(protocols=['TCP', 'UDP', 'ICMP'], embedding_size=10)
        self.interface_encoder = InterfaceEncoding(num_interfaces=6, embedding_size=10)
        self.dayofmonth_encoder = ProtocolEncoding(protocols=[0, 1, 2, 3, 4, 5, 6], embedding_size=10)

        self.packet_bins = torch.linspace(0, 1000, steps=11, device=device)  # 10 bins
        self.byte_bins = torch.linspace(0, 1_500_000, steps=11, device=device)  # 10 bins

        self.char_to_idxs = {chr(i): i for i in range(256)}  # Simple ASCII mapping

    def one_hot_encode(self, value, bins):
        bin_index = torch.bucketize(torch.tensor([value], device=self.device), bins) - 1
        # one-hot encode the bin index
        one_hot = torch.zeros(len(bins) - 1, device=self.device)
        if 0 <= bin_index < len(bins) - 1:
            one_hot[bin_index] = 1.0
        return one_hot

    def forward(self, data, day_of_week):
        # Extract individual fields from the input data
        dest_name = data["dest_ip"]
        protocol = data["protocol"]
        source_port = data["source_port"]
        destination_port = data["destination_port"]
        input_interface = data["input_interface"]
        output_interface = data["output_interface"]
        packet_count = data["packet_count"]
        byte_count = data["byte_count"]

        #dest_name = self.ip_tokenizer.ip2name(dest_ip)
        #print('[name: ], name: ', name)
        dns_indices = dns_to_char_indices(dest_name, self.char_to_idxs)
        dns_embedding = self.name_tokeinzer(dns_indices)
        protocol_embedding = self.protocol_encoder.encode(protocol).to(self.device)
        source_port_embedding = self.port_tokenizer.get_embedding(source_port).to(self.device)
        destination_port_embedding = self.port_tokenizer.get_embedding(destination_port).to(self.device)
        input_interface_embedding = self.interface_encoder.encode(input_interface).to(self.device)
        output_interface_embedding = self.interface_encoder.encode(output_interface).to(self.device)
        day_of_week_embedding = self.dayofmonth_encoder.encode(day_of_week).to(self.device)

        packet_one_hot = self.one_hot_encode(packet_count, self.packet_bins)
        byte_one_hot = self.one_hot_encode(byte_count, self.byte_bins)

        # print(dns_embedding.shape)
        # print(protocol_embedding.shape)
        # print(source_port_embedding.shape)
        # print(destination_port_embedding.shape)
        # print(input_interface_embedding.shape)
        # print(output_interface_embedding.shape)
        # print(packet_one_hot.shape)
        # print(byte_one_hot.shape)
        # print(day_of_week_embedding.shape)

        # normalized_input = torch.cat([
        #     dns_embedding,
        #     protocol_embedding,
        #     source_port_embedding,
        #     destination_port_embedding,
        #     input_interface_embedding,
        #     output_interface_embedding,
        #     packet_one_hot,
        #     byte_one_hot,
        #     day_of_week_embedding
        # ], dim=0)

        normalized_input = torch.stack([
            dns_embedding,
            protocol_embedding,
            source_port_embedding,
            destination_port_embedding,
            input_interface_embedding,
            output_interface_embedding,
            packet_one_hot,
            byte_one_hot,
            day_of_week_embedding
        ])

        return normalized_input

class InputNormalizer(nn.Module):
    '''
    This is intened to pe bart of the NN
    '''
    def __init__(self, embedding_size, device, cache_file_ip="resources/NameIpCacheFile.json", cache_file_ports="resources/port_embeddings.json"):
        super(InputNormalizer, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.ip_tokenizer = IPTokenizer(cache_file_ip)
        self.name_tokeinzer = DNSNameEmbedding(vocab_size=256, embedding_dim=5, hidden_dim=10)
        self.port_tokenizer = PortTokenizer(file_path=cache_file_ports)
        self.protocol_encoder = ProtocolEncoding(protocols=['TCP', 'UDP', 'ICMP'], embedding_size=3)
        self.interface_encoder = InterfaceEncoding(num_interfaces=6, embedding_size=6)
        #self.dayofmonth_encoder = ProtocolEncoding(protocols=[0, 1, 2, 3, 4, 5, 6], embedding_size=10)
        self.device = device

        self.packet_bins = torch.linspace(0, 1000, steps=11, device=device)  # 10 bins
        self.byte_bins = torch.linspace(0, 1_500_000, steps=11, device=device)  # 10 bins

        self.char_to_idxs = {chr(i): i for i in range(256)}  # Simple ASCII mapping

    def one_hot_encode(self, value, bins):
        bin_index = torch.bucketize(torch.tensor([value], device=self.device), bins) - 1
        # one-hot encode the bin index
        one_hot = torch.zeros(len(bins) - 1, device=self.device)
        if 0 <= bin_index < len(bins) - 1:
            one_hot[bin_index] = 1.0
        return one_hot, bin_index/len(bins)

    def forward(self, data, day_of_week, hour):
        # Extract individual fields from the input data
        dest_name = data["dest_ip"]
        protocol = data["protocol"]
        source_port = data["source_port"]
        destination_port = data["destination_port"]
        input_interface = data["input_interface"]
        output_interface = data["output_interface"]
        packet_count = data["packet_count"]
        byte_count = data["byte_count"]

        #dest_name = self.ip_tokenizer.ip2name(dest_ip)
        #print('[name: ], name: ', name)
        dns_indices = dns_to_char_indices(dest_name, self.char_to_idxs).to(self.device)
        dns_embedding = self.name_tokeinzer(dns_indices).to(self.device)
        protocol_embedding = self.protocol_encoder.encode(protocol).to(self.device)
        source_port_embedding = self.port_tokenizer.get_embedding(source_port).to(self.device)
        destination_port_embedding = self.port_tokenizer.get_embedding(destination_port).to(self.device)
        input_interface_embedding = self.interface_encoder.encode(input_interface).to(self.device)
        output_interface_embedding = self.interface_encoder.encode(output_interface).to(self.device)
        #day_of_week_embedding = self.dayofmonth_encoder.encode(day_of_week).to(self.device)
        day_of_week_embedding = torch.tensor([day_of_week / 6]).to(self.device)
        hour_normalized = torch.tensor([hour / 24]).to(self.device)

        _, normalized_packet_count = self.one_hot_encode(packet_count, self.packet_bins)
        _, normalized_byte_count = self.one_hot_encode(byte_count, self.byte_bins)

        # print(dns_embedding.shape)
        # print(protocol_embedding.shape)
        # print(source_port_embedding.shape)
        # print(destination_port_embedding.shape)
        # print(input_interface_embedding.shape)
        # print(output_interface_embedding.shape)
        # print(normalized_packet_count.shape)
        # print(normalized_byte_count.shape)
        # print(day_of_week_embedding.shape)
        # print(hour_normalized.shape)

        normalized_input = torch.cat([
            dns_embedding,
            protocol_embedding,
            source_port_embedding,
            destination_port_embedding,
            input_interface_embedding,
            output_interface_embedding,
            normalized_packet_count.to(self.device),
            normalized_byte_count.to(self.device),
            day_of_week_embedding,
            hour_normalized,
            torch.tensor([0]).to(self.device)
        ], dim=0)

        #print('normalized_input.shape: ', normalized_input.shape)

        return normalized_input


class TransformerAggregator(nn.Module):
    '''
    This is the version 1 of the aggregator
    It has the downlide that keeps the same diemsnion as the input, so there will be probbaly too little space for the information to be represented
    '''
    def __init__(self, input_dim, num_heads, num_layers):
        super(TransformerAggregator, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, input_vectors):
        cls_token = self.cls_token.expand(1, -1, -1)  # [1, 1, input_dim]
        input_with_cls = torch.cat([cls_token, input_vectors.unsqueeze(0)], dim=1)

        encoded_vectors = self.transformer(input_with_cls)  # [1, num_vectors + 1, input_dim]

        aggregated_vector = encoded_vectors[:, 0, :]  # [1, input_dim]
        return aggregated_vector.squeeze(0)

# class RNNAggregatorGRU(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(RNNAggregator, self).__init__()
#         self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

#     def forward(self, input_vectors):
#         input_vectors = input_vectors.unsqueeze(0)  # [1, num_vectors, input_dim]
#         _, hidden_state = self.rnn(input_vectors)  # [1, hidden_dim]
#         return hidden_state.squeeze(0)  # [hidden_dim]

class RNNAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RNNAggregator, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, input_vectors):
        input_vectors = input_vectors.unsqueeze(0)  # [1, num_vectors, input_dim]
        _, (hidden_state, _) = self.rnn(input_vectors)  # [num_layers, 1, hidden_dim]
        return hidden_state[-1].squeeze(0)  # [hidden_dim]



if __name__ == "__main__":


    ##########
    # test everything
    ##########


    device = 'cuda'
    train_dataset = NetFlowDatasetClassification('dataset/train')
    transformer_aggregator = TransformerAggregator(50, 10, 256)
    rnn_aggregator = RNNAggregator(50, 256, 2)
    content_user, target = train_dataset[0]
    #print(content_user1)
    #print(content_user2)
    print('len(content_user1)', len(content_user))
    print(target)

    input_normalizer = InputNormalizer(10, 'cuda')
    rnn_aggregator.to('cuda')

    feature_vectors = []
    for moment in tqdm(content_user):
        #print('moment: ', moment)
        outputs = []
        for event in moment["content"]:
            outputs.append(input_normalizer(event, moment['day_of_week'], int(moment['hour'].split(':')[0])))
        #print(len(outputs))
        outputs = torch.stack(outputs)
        #print('outputs.shape: ', outputs.shape)
        aggregated_features = rnn_aggregator(outputs.float())
        #print('aggregated_features.shape: ', aggregated_features.shape)
        feature_vectors.append(aggregated_features)
    feature_vector = torch.stack(feature_vectors)
    print('feature_vector.shape: ', feature_vector.shape)


    '''
    # **** This is  a good example for unbatched inference
    device = 'cuda'
    train_dataset = NetFlowDataset('dataset/train')
    transformer_aggregator = TransformerAggregator(50, 10, 256)
    rnn_aggregator = RNNAggregator(50, 256, 2)
    content_user1, content_user2, target = train_dataset[0]
    #print(content_user1)
    #print(content_user2)
    print('len(content_user1)', len(content_user1))
    print('len(content_user2)', len(content_user2))
    print(target)

    input_normalizer = InputNormalizer(10, 'cuda')
    rnn_aggregator.to('cuda')

    feature_vectors = []
    for moment in tqdm(content_user1):
        #print('moment: ', moment)
        outputs = []
        for event in moment["content"]:
            outputs.append(input_normalizer(event, moment['day_of_week'], int(moment['hour'].split(':')[0])))
        #print(len(outputs))
        outputs = torch.stack(outputs)
        #print('outputs.shape: ', outputs.shape)
        aggregated_features = rnn_aggregator(outputs.float())
        #print('aggregated_features.shape: ', aggregated_features.shape)
        feature_vectors.append(aggregated_features)
    feature_vector = torch.stack(feature_vectors)
    print('feature_vector.shape: ', feature_vector.shape)

    exit()
    '''


    
    # this is a fucntional example
    '''
    output0 = input_normalizer(content_user1[0]["content"][0], content_user1[0]['day_of_week'], int(content_user1[0]['hour'].split(':')[0], ))
    output1 = input_normalizer(content_user1[1]["content"][0], content_user1[1]['day_of_week'], int(content_user1[1]['hour'].split(':')[0], ))

    print('output.shape: ', output0.shape)
    print('output: ', output0)
    print('output.shape: ', output1.shape)
    print('output: ', output1)

    print('torch.stack([output0, output1]: ', torch.stack([output0, output1]).shape)

    aggregated_features = transformer_aggregator(torch.stack([output0, output1]).float())
    print('aggregated_features.hhape', aggregated_features.shape)

    aggregated_features = rnn_aggregator(torch.stack([output0, output1]).float())
    print('aggregated_features.hhape', aggregated_features.shape)
    '''

    



    ##########
    # test interface and protocol encoding
    ##########

    # # interface encoding
    # interface_encoder = InterfaceEncoding(num_interfaces=6, embedding_size=10)
    # interfaces = [0, 1, 3, 5]
    # interface_embeddings = interface_encoder.batch_encode(interfaces)
    # print("Interface Encodings:")
    # print(interface_embeddings)

    # # protocol encoding
    # protocol_encoder = ProtocolEncoding(protocols=['TCP', 'UDP', 'ICMP'], embedding_size=10)
    # protocols = ['TCP', 'UDP', 'ICMP', 'TCP']
    # protocol_embeddings = protocol_encoder.batch_encode(protocols)
    # print("\nProtocol Encodings:")
    # print(protocol_embeddings)

    ##########
    # test DNS ip-name tokenizer
    ##########

    # cache_file = "resources/NameIpCacheFile.json"
    # tokenizer = IPTokenizer(cache_file)

    # ip = "8.8.8.8" # google
    # result = tokenizer.ip2name(ip)
    # print(f"Resolved {ip}: {result}")


    # embedding_dim = 10
    # hidden_dim = 32
    # dns_embedding_model = DNSNameEmbedding(vocab_size=256, embedding_dim=5, hidden_dim=10)

    # # Example DNS name
    # dns_name = "dns.google.com"
    # char_to_idx = {chr(i): i for i in range(256)}  # Simple ASCII mapping
    # dns_indices = dns_to_char_indices(dns_name, char_to_idx).unsqueeze(0)  # batch dimension

    # # Generate embedding
    # dns_embedding = dns_embedding_model(dns_indices)
    # print(f"DNS Embedding for '{dns_name}': {dns_embedding.squeeze().detach().numpy()}")

    ##########
    # test port tokenizer
    ##########

    # ports = [
    #     20, 21, 22, 23, 25, 53, 67, 68, 80, 110, 123, 135, 137, 138, 139, 143,
    #     161, 194, 389, 443, 465, 548, 587, 636, 902, 993, 995, 1080, 1433, 1434,
    #     1521, 1720, 1723, 1812, 1813, 1883, 2049, 25565, 27017, 31337, 3306, 3389,
    #     44444, 5060, 5432, 5800, 5900, 8080, 8443, 10000, 11211, 20000, 55555
    # ] # list of most common ports
    # tokenizer = PortTokenizer(ports=ports, embedding_dim=10, file_path="resources/port_embeddings.json")
    # print("Embedding port 80:", tokenizer.get_embedding(80))

    # tokenizer = PortTokenizer(file_path="resources/port_embeddings.json")
    # print("Embedding port 443:", tokenizer.get_embedding(443))
    # print("Embedding port 677889:", tokenizer.get_embedding(677889))
    # print("Embedding port 677889:", tokenizer.get_embedding(677889))

    ##########
    # tets IP tokenizer
    ##########

    # cache_file = "resources/NameIpCacheFile.json"
    # tokenizer = IPTokenizer(cache_file)
    
    # ip = "8.8.8.8" # google
    # result = tokenizer.ip2name(ip)
    # print(f"Resolved {ip}: {result}")

    # # 8.8.4.4 google dns
    # ip = "8.8.4.4" # gogle
    # result = tokenizer.ip2name(ip)
    # print(f"Resolved {ip}: {result}")

    # # 208.67.222.222 cisco open dns
    # ip = "208.67.222.222" # gogle
    # result = tokenizer.ip2name(ip)
    # print(f"Resolved {ip}: {result}")

    # # 1.1.1.1 clowdfalare
    # ip = "1.1.1.1" # gogle
    # result = tokenizer.ip2name(ip)
    # print(f"Resolved {ip}: {result}")

    # tokenizer.save_cache()
    # print("Cache saved.")
