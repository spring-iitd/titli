import torch
import numpy as np
import matplotlib.pyplot as plt

from scapy.all import *

from titli.fe import AfterImage, NetStat
from titli.ids import TorchKitNET
from titli.utils import RMSELoss

# Read the pcap file using scapy
PCAP_PATH = "../PANDA/PANDA/data/benign/weekday.pcap"
packets = PcapReader(PCAP_PATH)

# define packet parser and feature extractor
state = NetStat()
fe = AfterImage(state=state)

clusters = [
    [29, 24, 23, 22, 20, 21, 28, 27, 25, 26],
    [61, 62, 60, 63, 64],
    [8, 9, 7, 5, 6],
    [38, 39, 72, 73, 74, 70, 71, 37, 35, 36],
    [4, 34],
    [48, 49, 82, 83, 84, 80, 81, 47, 45, 46],
    [13, 14, 12, 10, 11], [33], [19, 69], [3],
    [18, 2, 17, 15, 16, 0, 1],
    [68, 32, 67, 65, 66, 30, 31],
    [53, 54, 43, 44],
    [90, 91, 92, 93, 94, 58, 59, 57, 55, 56],
    [42, 40, 41],
    [78, 79, 77, 75, 76],
    [87, 88, 89, 85, 86, 52, 50, 51],
    [96, 95, 97, 98, 99]
]

# load the model
torch_model = TorchKitNET(clusters, norms_path="./artifacts/kitsune_norm_params.pkl")
torch_model.load_state_dict(
        torch.load("./artifacts/kitsune.pth")
    )
torch_model.eval()
anomaly_scores = []
criterion = RMSELoss()
threshold = 0.17
packets = PcapReader(PCAP_PATH)

print("Loaded the model in eval mode!!!")
for i, packet in enumerate(packets):
    
    # extract parsed packet values
    traffic_vector = fe.get_traffic_vector(packet)

    # extract features by updating the states
    features = fe.update(traffic_vector)

    outputs, tails = torch_model(features)
    loss = criterion(outputs, tails) # average loss over the batch

    if loss.data == 0:
        loss.data = torch.tensor(1e-2)
    anomaly_score = loss.data
    anomaly_scores.append(anomaly_score)

_ = plt.get_cmap('Set3')
        
_, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)
x_val = np.arange(len(anomaly_scores))

try:
    ax1.scatter(x_val, anomaly_scores, s=1, c='#00008B')
except:
    ax1.scatter(x_val, anomaly_scores, s=1, alpha=1.0, c='#FF8C00')

ax1.axhline(y=threshold, color='b', linestyle='-')
ax1.set_yscale("log")
ax1.set_title("Anomaly Scores from Kitsune Execution Phase")
ax1.set_ylabel("RMSE (log scaled)")
ax1.set_xlabel("packet index")

# Show or save the plot
plt.legend(["Anomaly Score", "Threshold"])
plt.savefig(f"{PCAP_PATH.split('/')[-1][:-5]}_re.png")
plt.close()
