import json
import socket
from typing import Tuple, Optional
import warnings
import os
import torch
import torch.nn as nn
import hashlib
import numpy as np


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


class PortTokenizer:
    def __init__(self, ports=None, embedding_dim=10, file_path=None):
        """
        Initializes the PortTokenizer. Loads embeddings from the provided file if it exists;
        otherwise, computes embeddings for the given ports and saves them to the file.
        """
        self.embedding_dim = embedding_dim
        self.file_path = file_path

        # Validate input
        if file_path and os.path.exists(file_path):
            self.load_embeddings()
        elif ports: # if no file is provided, compute the embeddings for the given ports
            self.ports = ports
            self.port_to_idx = {port: idx for idx, port in enumerate(self.ports)} # there is a need for port-index mapping
            self.embedding_layer = nn.Embedding(len(self.ports), self.embedding_dim)
            self.save_embeddings()
        else:
            raise ValueError("Either 'ports' or 'file_path' must be provided.")

    def _hash_unknowned_ports(self, port, scale=0.1):
        """
        Generates a deterministic embedding vector for a given port using its hash.
        """
        hash_value = hashlib.md5(str(port).encode()).hexdigest()
        
        hash_as_ints = [int(hash_value[i:i+2], 16) for i in range(0, len(hash_value), 2)]

        hash_as_floats = np.array(hash_as_ints[:self.embedding_dim]) / 255.0  # Scale to [0, 1]
        hash_as_floats = (hash_as_floats * 2) - 1  # Scale to [-1, 1]
        
        embedding = torch.tensor(hash_as_floats * scale, dtype=torch.float)
        
        return embedding

    def save_embeddings(self):
        embeddings = {}
        for port, idx in self.port_to_idx.items():
            embeddings[port] = self.embedding_layer(torch.tensor(idx)).detach().tolist()

        with open(self.file_path, 'w') as f:
            json.dump(embeddings, f, indent=4)

    def load_embeddings(self):
        if not self.file_path:
            raise ValueError("No file path provided for loading embeddings.")

        with open(self.file_path, 'r') as f:
            embeddings = json.load(f)

        # Ports and their indices
        self.ports = list(map(int, embeddings.keys()))
        self.port_to_idx = {port: idx for idx, port in enumerate(self.ports)}

        # Embedding layer
        embedding_weights = torch.tensor([embeddings[str(port)] for port in self.ports])
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_weights)
        print(f"Loaded embeddings from {self.file_path}")

    def get_embedding(self, port):
        if port not in self.port_to_idx:
            return self._hash_unknowned_ports(port)
        idx = self.port_to_idx[port]
        return self.embedding_layer(torch.tensor(idx))
    

if __name__ == "__main__":

    ##########
    # test port tokenizer
    ##########

    ports = [
        20, 21, 22, 23, 25, 53, 67, 68, 80, 110, 123, 135, 137, 138, 139, 143,
        161, 194, 389, 443, 465, 548, 587, 636, 902, 993, 995, 1080, 1433, 1434,
        1521, 1720, 1723, 1812, 1813, 1883, 2049, 25565, 27017, 31337, 3306, 3389,
        44444, 5060, 5432, 5800, 5900, 8080, 8443, 10000, 11211, 20000, 55555
    ] # list of most common ports
    tokenizer = PortTokenizer(ports=ports, embedding_dim=10, file_path="resources/port_embeddings.json")
    print("Embedding port 80:", tokenizer.get_embedding(80))

    tokenizer = PortTokenizer(file_path="resources/port_embeddings.json")
    print("Embedding port 443:", tokenizer.get_embedding(443))
    print("Embedding port 677889:", tokenizer.get_embedding(677889))
    print("Embedding port 677889:", tokenizer.get_embedding(677889))

    ##########
    # tets IP tokenizer
    ##########

    # cache_file = "resources/NameIpCacheFile.json"
    # tokenizer = IPTokenizer(cache_file)
    
    # ip = "8.8.8.8" # gogle
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
