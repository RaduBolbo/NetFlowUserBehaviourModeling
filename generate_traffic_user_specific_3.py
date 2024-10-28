import pandas as pd
import random
from datetime import datetime, timedelta

# Define constants
NUM_USERS = 10  # Number of users (IPs) to simulate
SIMULATION_DAYS = 90  # Number of days (approx. 3 months)
TRAFFIC_EVENTS_PER_DAY = 100  # Average traffic events per user per day

# Define user profiles with basic features
user_profiles = {
    f"192.168.1.{i}": {
        "http_ratio": random.uniform(0.4, 0.7),  # HTTP traffic proportion
        "ftp_ratio": random.uniform(0.1, 0.3),   # FTP traffic proportion
        "ssh_ratio": random.uniform(0.05, 0.15),  # SSH traffic proportion
        "peak_hours": (random.randint(8, 11), random.randint(17, 20)),  # Active hours
        "avg_bandwidth": random.uniform(100, 500),  # Avg bandwidth in KB per day
        "location": random.choice(["New York", "San Francisco", "London", "Berlin"]),
        "weekly_activity_cycle": random.choice(["Weekdays", "Weekend-heavy"]),
    }
    for i in range(1, NUM_USERS + 1)
}

# Define traffic types
traffic_types = ["HTTP", "FTP", "SSH", "OTHER"]

# Generate traffic dataset without abstract categories
def generate_traffic(user_ip, day):
    traffic_data = []
    profile = user_profiles[user_ip]
    
    for _ in range(TRAFFIC_EVENTS_PER_DAY):
        # Set time based on weekly and daily patterns
        if profile["weekly_activity_cycle"] == "Weekdays" and day.weekday() >= 5:
            continue  # Skip weekends for Weekdays-only users
        if profile["weekly_activity_cycle"] == "Weekend-heavy" and day.weekday() < 5 and random.random() < 0.5:
            continue  # Reduced activity on weekdays for Weekend-heavy users
            
        hour = random.choices(
            range(24),
            weights=[3 if profile["peak_hours"][0] <= h <= profile["peak_hours"][1] else 1 for h in range(24)],
            k=1
        )[0]
        event_time = day + timedelta(hours=hour, minutes=random.randint(0, 59), seconds=random.randint(0, 59))
        
        # Determine traffic type and packet size based on protocol
        traffic_type = random.choices(
            traffic_types,
            weights=[profile["http_ratio"], profile["ftp_ratio"], profile["ssh_ratio"], 1 - (profile["http_ratio"] + profile["ftp_ratio"] + profile["ssh_ratio"])],
            k=1
        )[0]
        
        if traffic_type == "HTTP":
            port = 80
            packet_size = random.randint(300, 1500)
        elif traffic_type == "FTP":
            port = 21
            packet_size = random.randint(500, 2000)
        elif traffic_type == "SSH":
            port = 22
            packet_size = random.randint(100, 500)
        else:
            port = random.choice([53, 443, 8080])  # Other common service ports
            packet_size = random.randint(50, 1500)
        
        # Consistent location without anomalies
        location = profile["location"]
        
        # Adjust packet size by user's average bandwidth usage
        packet_size = int(packet_size * profile["avg_bandwidth"] / 100)

        # Log traffic event with simplified features for real traffic simulation
        traffic_data.append({
            "timestamp": event_time,
            "src_ip": user_ip,
            "dest_ip": f"10.0.0.{random.randint(1, 254)}",
            "port": port,
            "protocol": traffic_type,
            "packet_size": packet_size,
            "location": location
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
df.to_csv("user_specific_traffic.csv", index=False)

print("User-specific traffic dataset generated as 'user_specific_traffic.csv'")
