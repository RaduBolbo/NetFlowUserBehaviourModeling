import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from tqdm import tqdm

# Parameters
num_users = 400
num_days = 90
time_window = 5  # in minutes
protocols = ["TCP", "UDP", "ICMP"]
start_date = datetime(2024, 1, 1)

# Load common IPs and ports
with open("resources/NameIpCacheFile.json", "r") as f:
    common_ips = json.load(f)  # format: {"name": "ip"}
    common_ip_list = list(common_ips.values())

with open("resources/port_embeddings.json", "r") as f:
    ports_data = json.load(f)  # format: {"port_number": "port_name"}
    ports_list = list(ports_data.keys())
    ports_list = [int(port) for port in ports_list]

def generate_activity_pattern():
    base_inactive = list(range(0, 6)) + list(range(22, 24))
    base_high = list(range(9, 18))  # Working hours
    base_medium = list(range(7, 9)) + list(range(18, 20))
    base_low = list(range(6, 7)) + list(range(20, 22))

    inactive = sorted(list(set(base_inactive + random.sample(base_low, k=random.randint(1, 2)))))
    high = sorted(list(set(base_high + random.sample(base_medium, k=random.randint(2, 4)))))
    medium = sorted([hour for hour in base_medium if hour not in high] + random.sample(base_low, k=random.randint(0, 2)))
    low = sorted([hour for hour in base_low if hour not in high + medium] + random.sample(base_inactive, k=random.randint(0, 1)))

    return {
        "inactive": inactive,
        "low": low,
        "medium": medium,
        "high": high
    }

# Create users with more logical separation
users = []
for user_id in range(num_users):
    favorite_dest_ips = random.sample(common_ip_list, 5)  
    preferred_ports = random.sample(ports_list, 5) 
    preferred_dest_ports = random.sample(ports_list, 5)
    protocol_distribution = [random.uniform(0.4, 0.8), random.uniform(0.1, 0.6), random.uniform(0.05, 0.3)]
    protocol_distribution = [p / sum(protocol_distribution) for p in protocol_distribution] 
    primary_interface = random.randint(1, 5) 
    secondary_interface = random.choice([i for i in range(1, 6) if i != primary_interface])

    users.append({
        "user_id": f"{user_id}",
        "favorite_dest_ips": favorite_dest_ips,
        "preferred_ports": preferred_ports,
        "protocol_distribution": protocol_distribution,
        "activity_pattern": generate_activity_pattern(),
        "primary_interface": primary_interface,
        "secondary_interface": secondary_interface
    })

def generate_activity_trend(duration, trend_type="flat"):
    if trend_type == "flat":
        return [random.randint(1, 5)] * duration
    elif trend_type == "rise_and_fall":
        peak = random.randint(5, 10)
        half_duration = duration // 2
        rise = list(range(1, peak))[:half_duration]
        fall = list(range(peak, 0, -1))[:half_duration]
        return rise + fall
    elif trend_type == "fluctuating":
        base = random.randint(3, 8)
        return [max(1, base + random.randint(-2, 2)) for _ in range(duration)]

# Generate data for each user
records = []
for user_idx, user in tqdm(enumerate(users), total=num_users):
    user_records = []
    user_id = user["user_id"]
    dest_ips = user["favorite_dest_ips"]
    preferred_ports = user["preferred_ports"]
    protocol_probs = user["protocol_distribution"]
    activity_pattern = user["activity_pattern"]
    primary_interface = user["primary_interface"]
    secondary_interface = user["secondary_interface"]

    current_index = 0
    ongoing_activities = []

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()

        for hour in range(24):
            if hour in activity_pattern["inactive"]:
                continue
            activity_level = (
                1 if hour in activity_pattern["low"]
                else 2 if hour in activity_pattern["medium"]
                else 3
            )

            for minute in range(0, 60, time_window):
                timestamp = current_date + timedelta(hours=hour, minutes=minute)

                ongoing_activities = [
                    act for act in ongoing_activities
                    if act["end_time"] > timestamp or random.random() > 0.2
                ]

                if random.random() < 0.3 * activity_level:
                    duration = random.randint(3, 15)
                    trend_type = random.choice(["flat", "rise_and_fall", "fluctuating"])
                    flow_trend = generate_activity_trend(duration, trend_type)

                    ongoing_activities.append({
                        "dest_ip": random.choices(dest_ips, k=1)[0],
                        "protocol": random.choices(protocols, protocol_probs)[0],
                        "end_time": timestamp + timedelta(minutes=duration * time_window),
                        "trend": flow_trend,
                        "trend_index": 0
                    })

                for activity in ongoing_activities:
                    dest_ip = activity["dest_ip"]
                    protocol = activity["protocol"]
                    flow_count = activity["trend"][activity["trend_index"]]
                    activity["trend_index"] = min(activity["trend_index"] + 1, len(activity["trend"]) - 1)

                    for _ in range(flow_count):
                        source_port = random.choices(preferred_ports, k=1)[0]
                        destination_port = random.choices(preferred_dest_ports, k=1)[0]
                        packet_count = random.randint(50, 1000)
                        byte_count = packet_count * random.randint(64, 1500)
                        interface_choice = (
                            primary_interface if random.random() < 0.8 else secondary_interface
                        )
                        user_records.append({
                            "index": current_index,
                            "user_id": user_id,
                            "dest_ip": dest_ip,
                            "protocol": protocol,
                            "source_port": source_port,
                            "destination_port": destination_port,
                            "input_interface": interface_choice,
                            "output_interface": interface_choice,
                            "packet_count": packet_count,
                            "byte_count": byte_count,
                            "timestamp": timestamp,
                            "day_of_week": day_of_week,
                            "hour": hour,
                            "minute": minute,
                            "second": 0
                        })
                        current_index += 1

    df = pd.DataFrame(user_records)
    df.to_csv(f'dataset/output{user_idx}.csv', index=False)