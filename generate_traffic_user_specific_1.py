import pandas as pd
import random
from datetime import datetime, timedelta

# Define constants
NUM_USERS = 10  # Number of users (IPs) to simulate
SIMULATION_DAYS = 90  # Number of days (approx. 3 months)
TRAFFIC_EVENTS_PER_DAY = 100  # Average traffic events per user per day

# Define user characteristics
user_profiles = {
    f"192.168.1.{i}": {
        "http_ratio": random.uniform(0.4, 0.7),  # HTTP traffic proportion
        "ftp_ratio": random.uniform(0.1, 0.3),   # FTP traffic proportion
        "ssh_ratio": random.uniform(0.1, 0.2)    # SSH traffic proportion
    }
    for i in range(1, NUM_USERS + 1)
}

# Define traffic types
traffic_types = ["HTTP", "FTP", "SSH", "OTHER"]

# Generate traffic dataset
def generate_traffic(user_ip, day):
    traffic_data = []
    for _ in range(TRAFFIC_EVENTS_PER_DAY):
        event_time = day + timedelta(seconds=random.randint(0, 86399))  # Random second in the day

        # Determine traffic type based on user profile
        profile = user_profiles[user_ip]
        traffic_type = random.choices(
            traffic_types,
            weights=[profile["http_ratio"], profile["ftp_ratio"], profile["ssh_ratio"], 1 - sum(profile.values())],
            k=1
        )[0]

        # Simulate packet size and destination port based on traffic type
        if traffic_type == "HTTP":
            port = 80
            packet_size = random.randint(300, 1500)  # Simulate HTTP packet sizes
        elif traffic_type == "FTP":
            port = 21
            packet_size = random.randint(500, 2000)
        elif traffic_type == "SSH":
            port = 22
            packet_size = random.randint(100, 500)
        else:
            port = random.choice([53, 443, 8080])  # DNS, HTTPS, other services
            packet_size = random.randint(50, 1500)

        # Destination IP (randomized within a private range for example)
        dest_ip = f"10.0.0.{random.randint(1, 254)}"

        # Log traffic event
        traffic_data.append({
            "timestamp": event_time,
            "src_ip": user_ip,
            "dest_ip": dest_ip,
            "port": port,
            "protocol": traffic_type,
            "packet_size": packet_size
        })

    return traffic_data

# Main simulation
all_traffic = []
start_date = datetime.now() - timedelta(days=SIMULATION_DAYS)
for day_offset in range(SIMULATION_DAYS):
    current_day = start_date + timedelta(days=day_offset)
    for user_ip in user_profiles:
        user_traffic = generate_traffic(user_ip, current_day)
        all_traffic.extend(user_traffic)

# Save to CSV
df = pd.DataFrame(all_traffic)
df.to_csv("synthetic_network_traffic.csv", index=False)

print("Synthetic network traffic dataset generated as 'synthetic_network_traffic.csv'")
