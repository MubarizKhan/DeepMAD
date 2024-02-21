from scapy.all import sniff, wrpcap

def capture_packets(interface='wlp0s20f3', packet_count=10, file_name='captured_packets.pcap'):
    """
    Captures a specified number of packets on the given network interface and saves them to a PCAP file.

    :param interface: The network interface to capture packets from (e.g., 'eth0').
    :param packet_count: The number of packets to capture.
    :param file_name: The name of the file to save the captured packets to.
    """
    print(f"Capturing {packet_count} packets on interface {interface}...")
    packets = sniff(iface=interface, count=packet_count)
    print(f"Saving captured packets to {file_name}...")
    wrpcap(file_name, packets)
    print("Done.")

# Example usage
capture_packets(interface='wlp0s20f3', packet_count=10, file_name='captured_packets.pcap')
