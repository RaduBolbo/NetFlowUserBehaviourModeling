import pyshark

INTERFACE = 'enp5s0'

def analyze_packet(packet):
    """Function to process each captured packet and print its details."""
    try:
        # Basic packet metadata
        print("\nPacket captured:")
        print(f"Timestamp: {packet.sniff_time}")
        if 'ip' in packet:
            print(f"Source IP: {packet.ip.src}")
            print(f"Destination IP: {packet.ip.dst}")

        # Transport Layer Information (TCP/UDP)
        if 'tcp' in packet:
            print(f"Protocol: TCP")
            print(f"Source Port: {packet.tcp.srcport}")
            print(f"Destination Port: {packet.tcp.dstport}")
            print(f"Sequence Number: {packet.tcp.seq}")
            print(f"Window Size: {packet.tcp.window_size}")
        elif 'udp' in packet:
            print(f"Protocol: UDP")
            print(f"Source Port: {packet.udp.srcport}")
            print(f"Destination Port: {packet.udp.dstport}")

        # Application Layer Protocols and Important Metadata
        if 'http' in packet:
            print("Application Protocol: HTTP")
            if hasattr(packet.http, 'host'):
                print(f"HTTP Host: {packet.http.host}")
            if hasattr(packet.http, 'request_uri'):
                print(f"HTTP Request URI: {packet.http.request_uri}")
            if hasattr(packet.http, 'user_agent'):
                print(f"User Agent: {packet.http.user_agent}")
        elif 'dns' in packet:
            print("Application Protocol: DNS")
            if hasattr(packet.dns, 'qry_name'):
                print(f"DNS Query: {packet.dns.qry_name}")
            if hasattr(packet.dns, 'qry_type'):
                print(f"DNS Query Type: {packet.dns.qry_type}")
        elif 'ftp' in packet:
            print("Application Protocol: FTP")
            if hasattr(packet.ftp, 'request'):
                print(f"FTP Request: {packet.ftp.request}")
        elif 'smtp' in packet:
            print("Application Protocol: SMTP")
            if hasattr(packet.smtp, 'mail_from'):
                print(f"SMTP Mail From: {packet.smtp.mail_from}")
            if hasattr(packet.smtp, 'rcpt_to'):
                print(f"SMTP Recipient To: {packet.smtp.rcpt_to}")
            if hasattr(packet.smtp, 'subject'):
                print(f"SMTP Subject: {packet.smtp.subject}")
        elif 'imap' in packet:
            print("Application Protocol: IMAP")
            if hasattr(packet.imap, 'request'):
                print(f"IMAP Request: {packet.imap.request}")
        elif 'pop' in packet:
            print("Application Protocol: POP3")
            if hasattr(packet.pop, 'request'):
                print(f"POP3 Request: {packet.pop.request}")
        elif 'dhcp' in packet:
            print("Application Protocol: DHCP")
            if hasattr(packet.dhcp, 'client_id'):
                print(f"DHCP Client ID: {packet.dhcp.client_id}")
            if hasattr(packet.dhcp, 'hostname'):
                print(f"DHCP Hostname: {packet.dhcp.hostname}")
        elif 'snmp' in packet:
            print("Application Protocol: SNMP")
            if hasattr(packet.snmp, 'oid'):
                print(f"SNMP OID: {packet.snmp.oid}")
            if hasattr(packet.snmp, 'value'):
                print(f"SNMP Value: {packet.snmp.value}")
        elif 'telnet' in packet:
            print("Application Protocol: Telnet")
            if hasattr(packet.telnet, 'data'):
                print(f"Telnet Data: {packet.telnet.data}")
        elif 'ssh' in packet:
            print("Application Protocol: SSH")
            if hasattr(packet.ssh, 'version'):
                print(f"SSH Version: {packet.ssh.version}")
        elif 'ssl' in packet:
            print("Application Protocol: TLS/SSL")
            if hasattr(packet.ssl, 'handshake_type'):
                print(f"TLS Handshake Type: {packet.ssl.handshake_type}")
        elif 'rtp' in packet:
            print("Application Protocol: RTP")
            if hasattr(packet.rtp, 'ssrc'):
                print(f"RTP SSRC: {packet.rtp.ssrc}")
            if hasattr(packet.rtp, 'payload_type'):
                print(f"RTP Payload Type: {packet.rtp.payload_type}")
        elif 'smb' in packet:
            print("Application Protocol: SMB")
            if hasattr(packet.smb, 'command'):
                print(f"SMB Command: {packet.smb.command}")
            if hasattr(packet.smb, 'filename'):
                print(f"SMB Filename: {packet.smb.filename}")

        # General packet size and protocol layers
        print(f"Packet Length: {packet.length}")
        print(f"Highest Layer Detected: {packet.highest_layer}")
        print("-" * 40)

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
