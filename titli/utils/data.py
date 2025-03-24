"""
This file contains all the util function for data processing
"""
from scapy.all import PcapReader

def dataloader(pcap_path, fe):
    packets = PcapReader(pcap_path)

    for pkt in packets:
        # extract parsed packet values
        traffic_vector = fe.get_traffic_vector(pkt)

        # extract features by updating the states
        features = fe.update(traffic_vector)

        yield features
