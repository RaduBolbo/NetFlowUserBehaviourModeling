import netflow
import socket

# Set up the UDP socket to listen for NetFlow v9 data on port 2055
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 2055

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for NetFlow packets on {UDP_IP}:{UDP_PORT}...")

while True:
    # Receive data from the NetFlow exporter
    payload, addr = sock.recvfrom(4096)  # Buffer size of 4096 bytes

    # Parse the NetFlow v9 packet
    try:
        packet = netflow.parse_packet(payload)
        print(f"Received NetFlow v9 packet from {addr}")
        
        # Access packet fields
        print(f"Version: {packet.header.version}")
        print(f"Records: {len(packet.flows)}")
        
        for flow in packet.flows:
            # Display information about each flow
            print(f"Source IP: {flow.SRCADDR}")
            print(f"Destination IP: {flow.DSTADDR}")
            print(f"Source Port: {flow.SRCPORT}")
            print(f"Destination Port: {flow.DSTPORT}")
            print(f"Protocol: {flow.PROTO}")
            print(f"Bytes: {flow.DOctets}")
            print(f"Packets: {flow.DPkts}")

            # If available, display MAC addresses (if supported by the exporter)
            if hasattr(flow, 'SRC_MAC'):
                print(f"Source MAC: {flow.SRC_MAC}")
            if hasattr(flow, 'DST_MAC'):
                print(f"Destination MAC: {flow.DST_MAC}")

    except Exception as e:
        print(f"Error parsing packet: {e}")
