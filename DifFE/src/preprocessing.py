import random

import torch
import torch.nn as nn

from constants import MAX_BITS_SIZE

try:
    import scapy.all as scapy
except Exception:
    scapy = None

random.seed(42)


class IPAddressEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super(IPAddressEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.octet_embedding = nn.Embedding(256, embedding_dim)
        self.packet = None

    def set_packet(self, packet):
        self.packet = packet

    def embed_ip(self, ip):
        octets = [int(octet) for octet in ip.split('.')]
        octets_tensor = torch.tensor(octets, dtype=torch.long)
        embedded_octets = self.octet_embedding(octets_tensor)
        return embedded_octets.view(-1)

    def ip_to_embedding_tensor(self, adv):
        if scapy is None:
            raise RuntimeError("scapy is required for packet feature extraction but could not be imported.")
        if self.packet and self.packet.haslayer(scapy.IP):
            src_ip = self.packet[scapy.IP].src
            dst_ip = self.packet[scapy.IP].dst
            if adv:
                src_ip = '.'.join(src_ip.split('.')[:-1] + [str(random.randint(0, 255))])
                dst_ip = '.'.join(dst_ip.split('.')[:-1] + [str(random.randint(0, 255))])
            src_ip_embedded = self.embed_ip(src_ip)
            dst_ip_embedded = self.embed_ip(dst_ip)
            return torch.cat((src_ip_embedded, dst_ip_embedded))
        else:
            return torch.zeros(8 * self.embedding_dim, dtype=torch.float)


class FeatureRepresentation:
    def __init__(self):
        self.packet = None
        self.prev_packet = None

    def _extract_timestamp(self, adv=False, get_integer=False):
        current_time = float(self.packet.time)
        prev_time = float(self.prev_packet.time)
        int_diff = current_time - prev_time

        if adv:
            perturbation = random.uniform(0.01, 0.1)
            if random.choice([True, False]):
                int_diff += perturbation
            else:
                int_diff -= perturbation
                if int_diff < 0:
                    int_diff = -int_diff

        if get_integer:
            return torch.tensor([int_diff])

        diff = int(max(int_diff * 1000000, 0))
        timestamp_bits = bin(diff)[2:]
        if len(timestamp_bits) < 32:
            timestamp_bits = timestamp_bits + "0" * (32 - len(timestamp_bits))
        return torch.tensor([int(bit) for bit in timestamp_bits])

    def _extract_packet_size(self, get_integer=False):
        packet_size = len(self.packet)
        if get_integer:
            normalized_size = (packet_size - 64) / (1518 - 64)
            return torch.tensor([normalized_size])
        packet_size_bits = format(packet_size, f'0{MAX_BITS_SIZE}b')
        return torch.tensor([int(bit) for bit in packet_size_bits])

    def mac_to_normalized_tensor(self):
        if scapy is None:
            raise RuntimeError("scapy is required for packet feature extraction but could not be imported.")
        if self.packet.haslayer(scapy.Ether):
            src_mac = self.packet[scapy.Ether].src
            dst_mac = self.packet[scapy.Ether].dst
            src_octets = [int(octet, 16) / 255.0 for octet in src_mac.split(':')]
            dst_octets = [int(octet, 16) / 255.0 for octet in dst_mac.split(':')]
            return torch.cat((
                torch.tensor(src_octets, dtype=torch.float32),
                torch.tensor(dst_octets, dtype=torch.float32),
            ))
        return torch.full((12,), 0.0, dtype=torch.float32)

    def port_to_normalized_tensor(self):
        if scapy is None:
            raise RuntimeError("scapy is required for packet feature extraction but could not be imported.")
        if self.packet.haslayer(scapy.IP):
            sport = dport = 1055
            if self.packet.haslayer(scapy.TCP):
                sport = self.packet[scapy.TCP].sport
                dport = self.packet[scapy.TCP].dport
            elif self.packet.haslayer(scapy.UDP):
                sport = self.packet[scapy.UDP].sport
                dport = self.packet[scapy.UDP].dport
            return torch.tensor([sport / 65535.0, dport / 65535.0], dtype=torch.float32)
        return torch.full((2,), 0.0, dtype=torch.float32)

    def get_int_embedded_representation(self, packet, prev_packet, adv):
        self.packet = packet
        self.prev_packet = prev_packet
        try:
            embedder = IPAddressEmbedder(embedding_dim=16)
            embedder.set_packet(self.packet)
            timestamp_tensor = self._extract_timestamp(adv, get_integer=True)
            mac_tensor = self.mac_to_normalized_tensor()
            ip_tensor = embedder.ip_to_embedding_tensor(adv=False)
            port_tensor = self.port_to_normalized_tensor()
            packet_size_tensor = self._extract_packet_size(get_integer=True)
            return torch.cat((timestamp_tensor, mac_tensor, ip_tensor, port_tensor, packet_size_tensor))
        except Exception as e:
            print(f"Exception occured: {e}")
            return None
