try:
    from scapy.utils import PcapReader
except Exception:
    PcapReader = None

from torch.utils.data import Dataset
from preprocessing import FeatureRepresentation
import torch
import bisect


class PcapWindowedDataset_lipschitz(Dataset):
    """
    Returns sliding windows of packet embeddings (clean + adversarial IAT) with AfterImage targets.
    Each sample is (x_window, x_adv_window, target_features).
    """

    def __init__(self, pcap_file, features, window_size, max_iterations=None, transform=None):
        if PcapReader is None:
            raise RuntimeError("scapy is required for reading PCAPs but could not be imported.")
        self.transform = transform
        self.window_size = window_size
        self.features = features.reset_index(drop=True) if features is not None else None

        packets_reader = PcapReader(pcap_file)
        prev_reader = PcapReader(pcap_file)

        if max_iterations is None:
            max_iterations = sum(1 for _ in PcapReader(pcap_file))
        self.max_iterations = max_iterations

        print("PCAP contains ", self.max_iterations, " packets")
        embedded_packets = []
        embedded_packets_adv = []
        fr = FeatureRepresentation()
        for i in range(self.max_iterations):
            if i == 0:
                packet = next(packets_reader)
                pkt_tensor = fr.get_int_embedded_representation(packet, packet, adv=False)
                pkt_tensor_adv = fr.get_int_embedded_representation(packet, packet, adv=True)
            else:
                packet = next(packets_reader)
                prev_packet = next(prev_reader)
                pkt_tensor = fr.get_int_embedded_representation(packet, prev_packet, adv=False)
                pkt_tensor_adv = fr.get_int_embedded_representation(packet, prev_packet, adv=True)

            if self.transform:
                pkt_tensor = self.transform(pkt_tensor)
                pkt_tensor_adv = self.transform(pkt_tensor_adv)

            embedded_packets.append(pkt_tensor.detach())
            embedded_packets_adv.append(pkt_tensor_adv.detach())

        self.embedded_packets = torch.stack(embedded_packets).detach()
        self.embedded_packets_adv = torch.stack(embedded_packets_adv).detach()

    def __len__(self):
        return self.max_iterations

    def __getitem__(self, index):
        def _window(buf):
            if index < self.window_size:
                x = buf[0:index + 1]
                pad = torch.zeros(self.window_size - x.size(0), x.size(1))
                return torch.cat([pad, x], dim=0)
            return buf[index - self.window_size + 1:index + 1]

        x = _window(self.embedded_packets)
        x_adv = _window(self.embedded_packets_adv)

        y = torch.tensor([], dtype=torch.float32)
        if self.features is not None:
            y = torch.tensor(self.features.iloc[index].values, dtype=torch.float32)

        return x, x_adv, y


class PcapTimeWindowedDataset_lipschitz(Dataset):
    """
    Variable-length time-based windowing for the Transformer feature extractor.

    For each packet i, returns all packets j with (t_i - t_j) <= t_max,
    additionally capped at the n_max most recent packets.

    Returns: (x, x_adv, t_rel, target)
        x, x_adv : (L_i, input_size) float — packet embeddings (L_i variable)
        t_rel    : (L_i,) float — timestamps relative to window start (seconds, monotonic)
        target   : (output_size,) float
    Use `time_windowed_collate` as the DataLoader collate_fn.
    """

    def __init__(self, pcap_file, features, t_max=10.0, n_max=512, max_iterations=None, transform=None):
        if PcapReader is None:
            raise RuntimeError("scapy is required for reading PCAPs but could not be imported.")
        self.transform = transform
        self.t_max = float(t_max)
        self.n_max = int(n_max)
        self.features = features.reset_index(drop=True) if features is not None else None

        packets_reader = PcapReader(pcap_file)
        prev_reader = PcapReader(pcap_file)

        if max_iterations is None:
            max_iterations = sum(1 for _ in PcapReader(pcap_file))
        self.max_iterations = max_iterations

        print("PCAP contains ", self.max_iterations, " packets")
        embedded_packets = []
        embedded_packets_adv = []
        cum_times = []
        cum = 0.0
        fr = FeatureRepresentation()
        for i in range(self.max_iterations):
            if i == 0:
                packet = next(packets_reader)
                pkt_tensor = fr.get_int_embedded_representation(packet, packet, adv=False)
                pkt_tensor_adv = fr.get_int_embedded_representation(packet, packet, adv=True)
                iat = 0.0
            else:
                packet = next(packets_reader)
                prev_packet = next(prev_reader)
                pkt_tensor = fr.get_int_embedded_representation(packet, prev_packet, adv=False)
                pkt_tensor_adv = fr.get_int_embedded_representation(packet, prev_packet, adv=True)
                iat = max(float(packet.time) - float(prev_packet.time), 0.0)
            cum += iat
            cum_times.append(cum)

            if self.transform:
                pkt_tensor = self.transform(pkt_tensor)
                pkt_tensor_adv = self.transform(pkt_tensor_adv)

            embedded_packets.append(pkt_tensor.detach())
            embedded_packets_adv.append(pkt_tensor_adv.detach())

        self.embedded_packets = torch.stack(embedded_packets).detach()
        self.embedded_packets_adv = torch.stack(embedded_packets_adv).detach()
        self.cum_times = cum_times  # plain list for bisect

    def __len__(self):
        return self.max_iterations

    def __getitem__(self, index):
        t_end = self.cum_times[index]
        threshold = t_end - self.t_max
        # first j with cum_times[j] >= threshold
        j = bisect.bisect_left(self.cum_times, threshold, 0, index + 1)
        start = max(j, index - self.n_max + 1)

        x = self.embedded_packets[start:index + 1]
        x_adv = self.embedded_packets_adv[start:index + 1]
        t_rel = torch.tensor(
            [self.cum_times[k] - self.cum_times[start] for k in range(start, index + 1)],
            dtype=torch.float32,
        )

        y = torch.tensor([], dtype=torch.float32)
        if self.features is not None:
            y = torch.tensor(self.features.iloc[index].values, dtype=torch.float32)

        return x, x_adv, t_rel, y


def time_windowed_collate(batch):
    """Right-aligned padding: newest packet at position max_len-1.

    Returns (x_pad, x_adv_pad, t_pad, key_padding_mask, y)
        x_pad, x_adv_pad : (B, L, input_size)
        t_pad            : (B, L)
        key_padding_mask : (B, L) bool, True where padded
        y                : (B, output_size)
    """
    xs, xs_adv, ts, ys = zip(*batch)
    lens = [x.size(0) for x in xs]
    max_len = max(lens)
    B = len(xs)
    feat_dim = xs[0].size(1)

    x_pad = torch.zeros(B, max_len, feat_dim, dtype=xs[0].dtype)
    x_adv_pad = torch.zeros(B, max_len, feat_dim, dtype=xs_adv[0].dtype)
    t_pad = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.ones(B, max_len, dtype=torch.bool)

    for i, (x, x_adv, t, L) in enumerate(zip(xs, xs_adv, ts, lens)):
        x_pad[i, max_len - L:, :] = x
        x_adv_pad[i, max_len - L:, :] = x_adv
        t_pad[i, max_len - L:] = t
        mask[i, max_len - L:] = False

    if ys[0].numel() > 0:
        y = torch.stack(list(ys))
    else:
        y = torch.tensor([], dtype=torch.float32)
    return x_pad, x_adv_pad, t_pad, mask, y
