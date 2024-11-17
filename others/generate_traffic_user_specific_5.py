import pandas as pd
import random
from datetime import datetime, timedelta

NUM_USERS = 10  # Number of users (IPs) to simulate
SIMULATION_DAYS = 90  # Number of days (approx. 3 months)
TRAFFIC_EVENTS_PER_DAY = 100  # Average traffic events per user per day

user_profiles = {
    f"192.168.1.{i}": {
        "http_ratio": random.uniform(0.4, 0.7),  # HTTP traffic proportion
        "ftp_ratio": random.uniform(0.1, 0.3),   # FTP traffic proportion
        "ssh_ratio": random.uniform(0.05, 0.15),  # SSH traffic proportion
        "peak_hours": (random.randint(8, 11), random.randint(17, 20)),  # Active hours
        "avg_bandwidth": random.uniform(100, 500),  # Avg bandwidth in KB per day
        "device_type": random.choice(["Laptop", "Desktop", "Smartphone", "Tablet"]),
        "location": random.choice(["New York", "San Francisco", "London", "Berlin"]),
        "weekly_activity_cycle": random.choice(["Weekdays", "Weekend-heavy"]),
        "application_preference": random.choice(["Video Streaming", "Web Browsing", "File Transfer", "Gaming"]),
        "packet_size_variance": random.uniform(0.5, 1.5),  # Controls packet size variation
        "connection_reliability": random.uniform(0.9, 1.0),  # Probability of packet success
        "network_speed": random.uniform(10, 100)  # Network speed in Mbps
    }
    for i in range(1, NUM_USERS + 1)
}

traffic_types = ["HTTP", "DNS", "FTP", "SMTP", "IMAP", "POP3", "DHCP", "SNMP", "Telnet", "SSH", "TLS/SSL", "RTP", "SMB"]

def generate_traffic(user_ip, day):
    traffic_data = []
    profile = user_profiles[user_ip]
    
    for _ in range(TRAFFIC_EVENTS_PER_DAY):
        if random.random() > profile["connection_reliability"]:
            continue
            
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
        
        weights = [
            profile["http_ratio"],      # HTTP
            0.1,                        # DNS - arbitrary default weight
            profile["ftp_ratio"],       # FTP
            0.05,                       # SMTP - arbitrary default weight
            0.02,                       # IMAP - arbitrary default weight
            0.02,                       # POP3 - arbitrary default weight
            0.03,                       # DHCP - arbitrary default weight
            0.03,                       # SNMP - arbitrary default weight
            0.01,                       # Telnet - arbitrary default weight
            profile["ssh_ratio"],       # SSH
            0.05,                       # TLS/SSL - arbitrary default weight
            0.02,                       # RTP - arbitrary default weight
            0.02                        # SMB - arbitrary default weight
        ]
        
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]

        traffic_type = random.choices(
            traffic_types,
            weights=normalized_weights,
            k=1
        )[0]

        port = 0
        packet_size = int(random.randint(100, 1500) * profile["packet_size_variance"])  # Varying packet size
        highest_layer = traffic_type
        application_protocol = traffic_type

        if traffic_type == "HTTP":
            port = 80
            packet_size = int(random.randint(300, 1500) * profile["packet_size_variance"])
            highest_layer = "HTTP"
            application_protocol = "HTTP"
            http_host = f"host{random.randint(1, 100)}.com"
            http_request_uri = f"/page{random.randint(1, 100)}"
            http_user_agent = f"{profile['device_type']}-Browser"
        elif traffic_type == "DNS":
            port = 53
            packet_size = int(random.randint(60, 500) * profile["packet_size_variance"])
            highest_layer = "DNS"
            application_protocol = "DNS"
            dns_query = f"query{random.randint(1, 100)}.com"
            dns_query_type = random.choice(["A", "AAAA", "MX", "CNAME"])
        elif traffic_type == "FTP":
            port = 21
            packet_size = int(random.randint(500, 2000) * profile["packet_size_variance"])
            highest_layer = "FTP"
            application_protocol = "FTP"
            ftp_request = "RETR"
        elif traffic_type == "SMTP":
            port = 25
            packet_size = int(random.randint(500, 1500) * profile["packet_size_variance"])
            highest_layer = "SMTP"
            application_protocol = "SMTP"
            smtp_mail_from = f"user{random.randint(1, 100)}@example.com"
            smtp_rcpt_to = f"user{random.randint(1, 100)}@example.com"
            smtp_subject = f"Subject {random.randint(1, 100)}"
        elif traffic_type == "SSH":
            port = 22
            packet_size = int(random.randint(100, 500) * profile["packet_size_variance"])
            highest_layer = "SSH"
            application_protocol = "SSH"
            ssh_version = "SSH-2.0-OpenSSH_8.2p1"
        elif traffic_type == "TLS/SSL":
            port = 443
            packet_size = int(random.randint(100, 2000) * profile["packet_size_variance"])
            highest_layer = "TLS/SSL"
            application_protocol = "TLS/SSL"
            tls_handshake_type = random.choice(["ClientHello", "ServerHello"])
            tls_record_version = random.choice(["TLS 1.2", "TLS 1.3"])
            tls_cipher_suite = "TLS_AES_128_GCM_SHA256"
            tls_session_id = f"session{random.randint(1, 100)}"

        dest_ip = f"10.0.0.{random.randint(1, 254)}"
        
        traffic_data.append({
            "timestamp": event_time,
            "src_ip": user_ip,
            "dest_ip": dest_ip,
            "src_port": port,
            "dst_port": random.randint(10000, 60000),  # Randomized client-side port
            "protocol": traffic_type,
            "packet_size": packet_size,
            "highest_layer": highest_layer,
            "application_protocol": application_protocol,
            "http_host": locals().get("http_host"),
            "http_request_uri": locals().get("http_request_uri"),
            "http_user_agent": locals().get("http_user_agent"),
            "dns_query": locals().get("dns_query"),
            "dns_query_type": locals().get("dns_query_type"),
            "ftp_request": locals().get("ftp_request"),
            "smtp_mail_from": locals().get("smtp_mail_from"),
            "smtp_rcpt_to": locals().get("smtp_rcpt_to"),
            "smtp_subject": locals().get("smtp_subject"),
            "ssh_version": locals().get("ssh_version"),
            "tls_handshake_type": locals().get("tls_handshake_type"),
            "tls_record_version": locals().get("tls_record_version"),
            "tls_cipher_suite": locals().get("tls_cipher_suite"),
            "tls_session_id": locals().get("tls_session_id")
        })

    return traffic_data

all_traffic = []
start_date = datetime.now() - timedelta(days=SIMULATION_DAYS)

for day_offset in range(SIMULATION_DAYS):
    current_day = start_date + timedelta(days=day_offset)
    for user_ip in user_profiles:
        user_traffic = generate_traffic(user_ip, current_day)
        all_traffic.extend(user_traffic)

df = pd.DataFrame(all_traffic)
df.to_csv("pyshark_compatible_traffic.csv", index=False)

print("User-specific traffic dataset generated as 'pyshark_compatible_traffic.csv'")
