import random
import pandas as pd
from datetime import datetime, timedelta
import json
from tqdm import tqdm

num_users = 400
num_days = 90
time_window = 5
protocols = ["TCP", "UDP", "ICMP"]
start_date = datetime(2024, 1, 1)

with open("resources/NameIpCacheFile.json", "r") as f:
    common_ips = json.load(f)
    common_ip_list = list(common_ips.values())

with open("resources/port_embeddings.json", "r") as f:
    ports_data = json.load(f)
    ports_list = [int(port) for port in ports_data.keys()]

def generate_activity_pattern():
    base_inactive = list(range(0, 6)) + list(range(22, 24))
    base_high = list(range(9, 18))
    base_medium = list(range(7, 9)) + list(range(18, 20))
    base_low = list(range(6, 7)) + list(range(20, 22))

    inactive = sorted(set(base_inactive + random.sample(base_low, k=random.randint(1, 2))))
    high = sorted(set(base_high + random.sample(base_medium, k=random.randint(2, 4))))
    medium = sorted([h for h in base_medium if h not in high] + random.sample(base_low, k=random.randint(0, 2)))
    low = sorted([h for h in base_low if h not in high + medium] + random.sample(base_inactive, k=random.randint(0, 1)))

    return {
        "inactive": inactive,
        "low": low,
        "medium": medium,
        "high": high
    }

def generate_activity_trend(duration, trend_type):
    if trend_type == "rise_and_fall":
        peak = random.randint(5, 10)
        rise = list(range(1, peak))[:duration // 2]
        fall = list(range(peak, 0, -1))[:duration // 2]
        return rise + fall
    elif trend_type == "constant":
        return [random.randint(3, 8)] * duration
    elif trend_type == "fluctuant":
        base = random.randint(3, 8)
        return [max(1, base + random.randint(-2, 2)) for _ in range(duration)]

users = []
for user_id in range(num_users):
    protocol_distribution = [random.uniform(0.4, 0.8), random.uniform(0.1, 0.6), random.uniform(0.05, 0.3)]
    protocol_distribution = [p / sum(protocol_distribution) for p in protocol_distribution]

    users.append({
        "user_id": str(user_id),
        "favorite_dest_ips": random.sample(common_ip_list, random.randint(3, 5)),
        "preferred_ports": random.sample(ports_list, random.randint(3, 5)),
        "protocol_distribution": protocol_distribution,
        "activity_pattern": generate_activity_pattern()
    })

for user in tqdm(users):
    user_records = []
    ongoing_activities = set()

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        activity_pattern = user["activity_pattern"]

        for hour in range(24):
            if hour in activity_pattern["inactive"]:
                continue

            activity_level = (
                1 if hour in activity_pattern["low"] else
                2 if hour in activity_pattern["medium"] else
                3
            )

            for minute in range(0, 60, time_window):
                timestamp = current_date + timedelta(hours=hour, minutes=minute)

                if random.random() > 0.3 * activity_level:
                    continue

                num_activities = random.randint(1, activity_level)

                for _ in range(num_activities):
                    dest_ip = random.choices(
                        user["favorite_dest_ips"] + common_ip_list,
                        [0.7] * len(user["favorite_dest_ips"]) + [0.3] * len(common_ip_list)
                    )[0]
                    protocol = random.choices(protocols, user["protocol_distribution"])[0]
                    source_port = random.choice(user["preferred_ports"])
                    destination_port = source_port  # Ensure the same port is used for source and destination
                    input_interface = random.randint(1, 5)
                    output_interface = input_interface  # Ensure the same interface is used for input and output
                    trend_type = random.choice(["rise_and_fall", "constant", "fluctuant"])
                    duration = random.randint(1, 5)
                    trend = generate_activity_trend(duration, trend_type)

                    user_records.append({
                        "index": len(user_records),
                        "user_id": user["user_id"],
                        "dest_ip": dest_ip,
                        "protocol": protocol,
                        "source_port": source_port,
                        "destination_port": destination_port,
                        "input_interface": input_interface,
                        "output_interface": output_interface,
                        "packet_count": sum(trend),
                        "byte_count": sum([p * random.randint(64, 1500) for p in trend]),
                        "timestamp": timestamp,
                        "day_of_week": timestamp.weekday(),
                        "hour": timestamp.hour,
                        "minute": timestamp.minute,
                        "second": 0
                    })

    df = pd.DataFrame(user_records)
    df.to_csv(f'dataset/user_{user["user_id"]}_activity.csv', index=False)
