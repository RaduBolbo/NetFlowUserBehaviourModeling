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
protocols = ["TCP", "UDP", "ICMP"]  # TCP, UDP, ICMP
start_date = datetime(2024, 1, 1)

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

users = []
from_where = 81
for user_id in range(num_users):
    preferred_ports = random.sample(ports_list, 7)
    users.append({
        "user_id": f"{user_id}",  # User index as a string
        "favorite_dest_ips": random.sample(common_ip_list, 8),  # Sample 5 IPs from the common IP list
        "preferred_ports": preferred_ports,
        "protocol_distribution": [0.6, 0.3, 0.1],  # Probabilities for TCP, UDP, ICMP
        "activity_pattern": generate_activity_pattern(),
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


records = []

for user_idx, user in tqdm(enumerate(users)):
    if user_idx < from_where:
        continue
    user_records = []
    user_id = user["user_id"]
    dest_ips = user["favorite_dest_ips"]
    preferred_ports = user["preferred_ports"]
    protocol_probs = user["protocol_distribution"]
    activity_pattern = user["activity_pattern"]
    current_index = 0

    ongoing_activities = []

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()

        for hour in range(24):
            if hour in activity_pattern["inactive"]:
                continue
            elif hour in activity_pattern["low"]:
                activity_level = 1
            elif hour in activity_pattern["medium"]:
                activity_level = 2
            elif hour in activity_pattern["high"]:
                activity_level = 3

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
                        "dest_ip": random.choices(dest_ips + common_ip_list,
                                                  [0.7 if ip in dest_ips else 0.3 for ip in dest_ips + common_ip_list])[0],
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
                        source_port = random.choices(preferred_ports + ports_list,
                                                     [0.8 if port in preferred_ports else 0.2 for port in preferred_ports + ports_list])[0]
                        destination_port = random.choices(preferred_ports + ports_list,
                                                          [0.8 if port in preferred_ports else 0.2 for port in preferred_ports + ports_list])[0]
                        packet_count = random.randint(50, 1000)
                        byte_count = packet_count * random.randint(64, 1500)
                        input_interface = random.randint(1, 5)
                        output_interface = random.randint(1, 5)

                        user_records.append({
                            "index": current_index,
                            "user_id": user_id,
                            "dest_ip": dest_ip,
                            "protocol": protocol,
                            "source_port": source_port,
                            "destination_port": destination_port,
                            "input_interface": input_interface,
                            "output_interface": output_interface,
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


# df = pd.DataFrame(records)

# print(df.head())

# df.to_csv('dataset/output{}.csv', index=False)







# import random
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import json

# # Parameters
# num_users = 10
# num_days = 90
# time_window = 5  # in minutes
# protocols = [6, 17, 1]  # TCP, UDP, ICMP
# start_date = datetime(2024, 1, 1)

# with open("resources/NameIpCacheFile.json", "r") as f:
#     common_ips = json.load(f)  # format: {"name": "ip"}
#     common_ip_list = list(common_ips.values())

# def generate_activity_pattern():
#     base_inactive = list(range(0, 6)) + list(range(22, 24))
#     base_high = list(range(9, 18))  # Working hours
#     base_medium = list(range(7, 9)) + list(range(18, 20))
#     base_low = list(range(6, 7)) + list(range(20, 22))

#     inactive = sorted(list(set(base_inactive + random.sample(base_low, k=random.randint(1, 2)))))
#     high = sorted(list(set(base_high + random.sample(base_medium, k=random.randint(2, 4)))))
#     medium = sorted([hour for hour in base_medium if hour not in high] + random.sample(base_low, k=random.randint(0, 2)))
#     low = sorted([hour for hour in base_low if hour not in high + medium] + random.sample(base_inactive, k=random.randint(0, 1)))

#     return {
#         "inactive": inactive,
#         "low": low,
#         "medium": medium,
#         "high": high
#     }

# users = []
# for user_id in range(num_users):
#     users.append({
#         "user_id": f"{user_id}",  # User index as a string
#         "favorite_dest_ips": random.sample(common_ip_list, 8),  # Sample 5 IPs from the common IP list
#         "protocol_distribution": [0.6, 0.3, 0.1],  # Probabilities for TCP, UDP, ICMP
#         # Activity patterns: INACTIVE, LOW, MEDIUM, HIGH
#         "activity_pattern": generate_activity_pattern(),
#     })


# def generate_activity_trend(duration, trend_type="flat"):
#     if trend_type == "flat":
#         return [random.randint(1, 5)] * duration
#     elif trend_type == "rise_and_fall":
#         peak = random.randint(5, 10)
#         half_duration = duration // 2
#         rise = list(range(1, peak))[:half_duration]
#         fall = list(range(peak, 0, -1))[:half_duration]
#         return rise + fall
#     elif trend_type == "fluctuating":
#         base = random.randint(3, 8)
#         return [max(1, base + random.randint(-2, 2)) for _ in range(duration)]


# records = []

# for user in users:
#     user_id = user["user_id"]
#     dest_ips = user["favorite_dest_ips"]
#     protocol_probs = user["protocol_distribution"]
#     activity_pattern = user["activity_pattern"]
#     current_index = 0  # index for the flow for each user

#     ongoing_activities = []

#     for day in range(num_days):
#         current_date = start_date + timedelta(days=day)
#         day_of_week = current_date.weekday()

#         for hour in range(24):
#             if hour in activity_pattern["inactive"]:
#                 continue # skip the inactive hours. People need sleep
#             elif hour in activity_pattern["low"]:
#                 activity_level = 1
#             elif hour in activity_pattern["medium"]:
#                 activity_level = 2
#             elif hour in activity_pattern["high"]:
#                 activity_level = 3

#             for minute in range(0, 60, time_window):
#                 timestamp = current_date + timedelta(hours=hour, minutes=minute)

#                 ongoing_activities = [
#                     act for act in ongoing_activities
#                     if act["end_time"] > timestamp or random.random() > 0.2
#                 ]

#                 if random.random() < 0.3 * activity_level:
#                     duration = random.randint(3, 15)
#                     trend_type = random.choice(["flat", "rise_and_fall", "fluctuating"])
#                     flow_trend = generate_activity_trend(duration, trend_type)

#                     ongoing_activities.append({
#                         "dest_ip": random.choices(dest_ips + common_ip_list,
#                                                   [0.7 if ip in dest_ips else 0.3 for ip in dest_ips + common_ip_list])[0],
#                         "protocol": random.choices(protocols, protocol_probs)[0],
#                         "end_time": timestamp + timedelta(minutes=duration * time_window),
#                         "trend": flow_trend,
#                         "trend_index": 0
#                     })

#                 # Generate flows for ongoing activities
#                 for activity in ongoing_activities:
#                     dest_ip = activity["dest_ip"]
#                     protocol = activity["protocol"]
#                     flow_count = activity["trend"][activity["trend_index"]]
#                     activity["trend_index"] = min(activity["trend_index"] + 1, len(activity["trend"]) - 1)

#                     for _ in range(flow_count):
#                         packet_count = random.randint(50, 1000)
#                         byte_count = packet_count * random.randint(64, 1500)
#                         input_interface = random.randint(1, 5)
#                         output_interface = random.randint(1, 5)

#                         records.append({
#                             "index": current_index,
#                             "user_id": user_id,
#                             "dest_ip": dest_ip,
#                             "protocol": protocol,
#                             "input_interface": input_interface,
#                             "output_interface": output_interface,
#                             "packet_count": packet_count,
#                             "byte_count": byte_count,
#                             "timestamp": timestamp,
#                             "day_of_week": day_of_week,
#                             "hour": hour,
#                             "minute": minute,
#                             "second": 0
#                         })
#                         current_index += 1

# # Convert to DataFrame for analysis
# df = pd.DataFrame(records)

# print(df.head())

# df.to_csv('output.csv', index=False)














'''
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
num_users = 10
num_days = 90
time_window = 5  # in minutes
protocols = [6, 17, 1]  # TCP, UDP, ICMP
start_date = datetime(2024, 1, 1)

users = []
for user_id in range(num_users):
    users.append({
        "user_id": f"{user_id}", # **** acutually I wanted indexes for users, nbot IPs
        "favorite_dest_ips": [f"8.8.8.{i}" for i in random.sample(range(1, 255), 5)], # **** here I want to take these IP-s form a list pof very common IPs that are commonly used: google, microsoft, wapp and a lost of others, but those IPs will be read froma  different JSON file NameIpCacheFile.json, where ther will be a dict in the format "name": "ip"
        "protocol_distribution": [0.6, 0.3, 0.1],  # Probabilities for TCP, UDP, ICMP
        "activity_pattern": {
            "weekday": [random.randint(7, 9), random.randint(17, 20)], # **** here I would have liked not to have rigid active/inactive hours. instead, I want to have INACTIVE hours, LIGHT ACTIVITY hours, MEDIUM and HIGH activity hours. (Sure, in the night the users will likely be ianctive. it has to eb realist)
            "weekend": [random.randint(10, 12), random.randint(14, 16)]
        }
    })

def generate_activity_trend(duration, trend_type="flat"):
    if trend_type == "flat":
        return [random.randint(1, 5)] * duration
    elif trend_type == "rise_and_fall":
        peak = random.randint(5, 10)
        return list(range(1, peak)) + list(range(peak, 0, -1)) # **** why doesn't rise_and_fall use durration? Is this ok? Check again maybe it is good maybe not
    elif trend_type == "fluctuating":
        base = random.randint(3, 8)
        return [max(1, base + random.randint(-2, 2)) for _ in range(duration)]

records = []

for user in users:
    user_id = user["user_id"]
    dest_ips = user["favorite_dest_ips"]
    protocol_probs = user["protocol_distribution"]
    activity_pattern = user["activity_pattern"]
    current_index = 0 # **** this should have been here because it is the index for the flow as part of th  sequence for each user: each user with his index.

    ongoing_activities = []

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()
        hours_active = activity_pattern["weekday"] if day_of_week < 5 else activity_pattern["weekend"]

        for hour in range(24): # **** also, I would like not to record the INACTIVE hours at all. Just active hours.
            if hour not in hours_active:
                continue  # Skip inactive hours

            for minute in range(0, 60, time_window):
                timestamp = current_date + timedelta(hours=hour, minutes=minute)

                ongoing_activities = [
                    act for act in ongoing_activities
                    if act["end_time"] > timestamp or random.random() > 0.2
                ]

                if random.random() < 0.3:
                    duration = random.randint(3, 15)
                    trend_type = random.choice(["flat", "rise_and_fall", "fluctuating"])
                    flow_trend = generate_activity_trend(duration, trend_type)

                    ongoing_activities.append({ # **** I think this should also depend on the multiple LOW ACTIVITY/ MEDIUM ACTIVITY / HIGH activity thing.
                        "dest_ip": random.choice(dest_ips), # **** these are only the favourite dest IPs. Other IPs can also be called. Also, we may consider a TOP of favourite dest IPs, so that not every favourite IP has equal chances
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
                        packet_count = random.randint(50, 1000)
                        byte_count = packet_count * random.randint(64, 1500)
                        input_interface = random.randint(1, 5)
                        output_interface = random.randint(1, 5)

                        records.append({
                            "index": current_index,
                            "user_id": user_id,
                            "dest_ip": dest_ip,
                            "protocol": protocol,
                            "input_interface": input_interface,
                            "output_interface": output_interface,
                            "packet_count": packet_count,
                            "byte_count": byte_count,
                            "timestamp": timestamp,
                            "day_of_week": day_of_week,
                            "hour": hour,
                            "minute": minute,
                            "second": 0 
                        })
                        current_index += 1

df = pd.DataFrame(records)

print(df.head())

'''
