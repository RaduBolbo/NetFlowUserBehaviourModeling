import pyshark

INTERFACE = 'enp5s0'


def analyze_packet(packet):
    """Function to process each captured packet and store its details."""
    packet_info = {}
    try:
        # Basic packet metadata
        packet_info["timestamp"] = packet.sniff_time
        if 'ip' in packet:
            packet_info["src_ip"] = packet.ip.src
            packet_info["dst_ip"] = packet.ip.dst

        # Transport Layer Information (TCP/UDP)
        if 'tcp' in packet:
            packet_info["protocol"] = "TCP"
            packet_info["src_port"] = packet.tcp.srcport
            packet_info["dst_port"] = packet.tcp.dstport
            packet_info["seq_num"] = packet.tcp.seq
            packet_info["window_size"] = packet.tcp.window_size
        elif 'udp' in packet:
            packet_info["protocol"] = "UDP"
            packet_info["src_port"] = packet.udp.srcport
            packet_info["dst_port"] = packet.udp.dstport
        elif 'ssl' in packet or 'tls' in packet:
            packet_info["protocol"] = "TLS/SSL"
            # Capture TLS-specific fields if available
            packet_info["tls_handshake_type"] = getattr(packet.ssl, 'handshake_type', None)
            packet_info["tls_record_version"] = getattr(packet.ssl, 'record_version', None)
            packet_info["tls_cipher_suite"] = getattr(packet.ssl, 'cipher_suite', None)
            packet_info["tls_session_id"] = getattr(packet.ssl, 'session_id', None)

        # Application Layer Protocols and Important Metadata
        if 'http' in packet:
            packet_info["application_protocol"] = "HTTP"
            packet_info["http_host"] = getattr(packet.http, 'host', None)
            packet_info["http_request_uri"] = getattr(packet.http, 'request_uri', None)
            packet_info["http_user_agent"] = getattr(packet.http, 'user_agent', None)
        elif 'dns' in packet:
            packet_info["application_protocol"] = "DNS"
            packet_info["dns_query"] = getattr(packet.dns, 'qry_name', None)
            packet_info["dns_query_type"] = getattr(packet.dns, 'qry_type', None)
        elif 'ftp' in packet:
            packet_info["application_protocol"] = "FTP"
            packet_info["ftp_request"] = getattr(packet.ftp, 'request', None)
        elif 'smtp' in packet:
            packet_info["application_protocol"] = "SMTP"
            packet_info["smtp_mail_from"] = getattr(packet.smtp, 'mail_from', None)
            packet_info["smtp_rcpt_to"] = getattr(packet.smtp, 'rcpt_to', None)
            packet_info["smtp_subject"] = getattr(packet.smtp, 'subject', None)
        elif 'imap' in packet:
            packet_info["application_protocol"] = "IMAP"
            packet_info["imap_request"] = getattr(packet.imap, 'request', None)
        elif 'pop' in packet:
            packet_info["application_protocol"] = "POP3"
            packet_info["pop_request"] = getattr(packet.pop, 'request', None)
        elif 'dhcp' in packet:
            packet_info["application_protocol"] = "DHCP"
            packet_info["dhcp_client_id"] = getattr(packet.dhcp, 'client_id', None)
            packet_info["dhcp_hostname"] = getattr(packet.dhcp, 'hostname', None)
        elif 'snmp' in packet:
            packet_info["application_protocol"] = "SNMP"
            packet_info["snmp_oid"] = getattr(packet.snmp, 'oid', None)
            packet_info["snmp_value"] = getattr(packet.snmp, 'value', None)
        elif 'telnet' in packet:
            packet_info["application_protocol"] = "Telnet"
            packet_info["telnet_data"] = getattr(packet.telnet, 'data', None)
        elif 'ssh' in packet:
            packet_info["application_protocol"] = "SSH"
            packet_info["ssh_version"] = getattr(packet.ssh, 'version', None)
        elif 'ssl' in packet:
            packet_info["application_protocol"] = "TLS/SSL"
            packet_info["tls_handshake_type"] = getattr(packet.ssl, 'handshake_type', None)
        elif 'rtp' in packet:
            packet_info["application_protocol"] = "RTP"
            packet_info["rtp_ssrc"] = getattr(packet.rtp, 'ssrc', None)
            packet_info["rtp_payload_type"] = getattr(packet.rtp, 'payload_type', None)
        elif 'smb' in packet:
            packet_info["application_protocol"] = "SMB"
            packet_info["smb_command"] = getattr(packet.smb, 'command', None)
            packet_info["smb_filename"] = getattr(packet.smb, 'filename', None)

        # General packet size and protocol layers
        packet_info["packet_length"] = packet.length
        packet_info["highest_layer"] = packet.highest_layer

        # Only print if the highest layer is an application layer
        if packet_info.get("application_protocol"):
            print("\nApplication Layer Packet Detected:")
            for key, value in packet_info.items():
                print(f"{key}: {value}")
            print("-" * 40)

        return packet_info

    except AttributeError as e:
        # Handle packets that may not have all expected attributes
        print(f"Encountered a packet with missing fields: {e}")
        return None

def start_capture(interface):
    print(f"Starting capture on interface {interface}...")
    capture = pyshark.LiveCapture(interface=interface)

    try:
        for packet in capture.sniff_continuously():
            analyze_packet(packet)
    
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        capture.close()

if __name__ == "__main__":
    start_capture(INTERFACE)

