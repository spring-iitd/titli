import time
import random

from torch import Tensor
from typing import Any
from scapy.all import PcapReader

from fe.after_image import AfterImage, NetStat

PCAP_PATH = "weekday.pcap"
state = NetStat()
fe = AfterImage(state=state)

def model(features: Tensor) -> int:

    rand_bin = random.randint(0, 1)

    return rand_bin

def perturb_timestamp(packet: Any) -> Any:
    packet.time = packet.time + 0.5

    return packet

carry = 0
for i, packet in enumerate(PcapReader(PCAP_PATH)):

    start = time.time()
    perturbed_timestamps = []

    # step 1: extract parsed packet values
    packet.time += carry
    traffic_vector = fe.get_traffic_vector(packet)
    print(traffic_vector)
    import pdb; pdb.set_trace()

    # step 2: FAKE update the feature extractor
    features = fe.peek([traffic_vector])

    # step 2.1: check for evasion
    anomaly_score = model(features)
    if not anomaly_score: # 0 is normal, 1 is anomaly
        # print('Evasion detected')

        # step 2.2: update the feature extractor with the original packet
        _ = fe.update(traffic_vector)
        continue

    # step 3: perturbation loop
    max_steps = 3
    for step in range(max_steps):

        # step 3.1: add a perturbation to the timestamp
        modified_packet = perturb_timestamp(packet)
        carry += 0.5

        # step 3.2: extract updated parsed packet values
        modified_traffic_vector = fe.get_traffic_vector(modified_packet)

        # step 3.2: FAKE update the feature extractor
        modified_features = fe.peek([modified_traffic_vector])

        # step 3.3: check for evasion
        anomaly_score = model(modified_features)

        if not anomaly_score: # 0 is normal, 1 is anomaly
            # print('Evasion detected at step', step)
            break

    # step 4: update the feature extractor with the original packet
    features = fe.update(modified_traffic_vector)

    print(f"Time to process packet {i+1}: {time.time() - start}")
