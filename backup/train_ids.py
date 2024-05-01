import pickle
import numpy as np

from scapy.all import *

from walle.fe import AfterImage, NetStat
from walle.ids import KitNET

import matplotlib.pyplot as plt

# Read the pcap file using scapy
PCAP_PATH = "../PANDA/PANDA/data/benign/weekday.pcap"
packets = PcapReader(PCAP_PATH)

# define packet parser and feature extractor
state = NetStat()
fe = AfterImage(state=state)

num_packets = len(rdpcap(PCAP_PATH))
params = {
    # the pcap, pcapng, or tsv file to process.
    "path": PCAP_PATH,
    "packet_limit": np.Inf,  # the number of packets to process,
    # KitNET params:
    # maximum size for any autoencoder in the ensemble layer
    "maxAE": 10,
    # the number of instances taken to learn the feature mapping (the ensemble's architecture)
    "FMgrace": np.floor(0.2 * num_packets),
    # FMgrace+ADgrace<=num samples in normal traffic
    "ADgrace": np.floor(0.8 * num_packets),
    "normalize": True,
    "num_features": 100,
    "model_path": "./artifacts/kitsune.pkl"
}

# define the model
model = KitNET(params["num_features"], params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75, normalize=params["normalize"],
               model_path=params["model_path"])

# parse, extract and process each packet
for i, pkt in enumerate(packets):
    
    # extract parsed packet values
    traffic_vector = fe.get_traffic_vector(pkt)

    # extract features by updating the states
    features = fe.update(traffic_vector)

    model.process(features)

# save both pkl and pth(torch) models
model.get_torch_model()
with open(params["model_path"], "wb") as of:
    pickle.dump(model, of)

print(model.v)

#### Threshold Calculation ####
# Read the pcap file using scapy
PCAP_PATH = "../PANDA/PANDA/data/benign/weekday.pcap"
packets = PcapReader(PCAP_PATH)

# define packet parser and feature extractor
state = NetStat()
fe = AfterImage(state=state)

rmse_array = []

for i, pkt in enumerate(packets):
    traffic_vector = fe.get_traffic_vector(pkt)
    features = fe.update(traffic_vector)
    rmse_array.append(model.execute(features))

# threshold is min(mean+3std, max)
benignSample = np.log(rmse_array)
mean = np.mean(benignSample)
std = np.std(benignSample)
threshold_std = np.exp(mean + 3 * std)
threshold_max = max(rmse_array)
threshold = min(threshold_max, threshold_std)

print(f"Threshold: {threshold}")

_ = plt.get_cmap('Set3')

_, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)
x_val = np.arange(len(rmse_array))

try:
    ax1.scatter(x_val, rmse_array, s=1, c='#00008B')
except:
    ax1.scatter(x_val, rmse_array, s=1, alpha=1.0, c='#FF8C00')

ax1.axhline(y=threshold, color='r', linestyle='-')
ax1.set_yscale("log")
ax1.set_title("Anomaly Scores from Kitsune Execution Phase")
ax1.set_ylabel("RMSE (log scaled)")
ax1.set_xlabel("packet index")

# Show or save the plot
plt.legend(["Anomaly Score", "Threshold"])
plt.savefig(f"{PCAP_PATH.split('/')[-1][:-5]}_raw_re.png")
plt.close()
