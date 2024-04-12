import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scapy.all import *

from walle.fe import AfterImage, NetStat
from walle.ids import KitNET, TorchKitNET
from walle.utils import RMSELoss

# Read the pcap file using scapy
PCAP_PATH = "weekday.pcap"
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
    # the number of instances used to train the anomaly detector (ensemble itself)
    # FMgrace+ADgrace<=num samples in normal traffic
    "ADgrace": np.floor(0.8 * num_packets),
    # directory of kitsune
    # "model_path": kitsune_path,
    "normalize": True,
    "num_features": 100,
    "model_path": "kitsune.pkl"
}

# define the model
model = KitNET(params["num_features"], params["maxAE"], params["FMgrace"],
               params["ADgrace"], 0.1, 0.75, normalize=params["normalize"])

# parse, extract and process each packet
for i, packet in enumerate(packets):
    
    # extract parsed packet values
    traffic_vector = fe.get_traffic_vector(packet)

    # extract features by updating the states
    features = fe.update(traffic_vector)

    model.process(features)

# save both pkl and pth(torch) models
model.get_torch_model()
with open(params["model_path"], "wb") as of:
    pickle.dump(model, of)

# load the model
torch_model = TorchKitNET(model.v)
torch_model.load_state_dict(
        torch.load(f"kitsune.pth")
    )
torch_model.eval()
anomaly_scores = []
criterion = RMSELoss()
threshold = 0.27
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

ax1.axhline(y=threshold, color='r', linestyle='-')
ax1.set_yscale("log")
ax1.set_title("Anomaly Scores from Kitsune Execution Phase")
ax1.set_ylabel("RMSE (log scaled)")
ax1.set_xlabel("packet index")

# Show or save the plot
plt.legend(["Threshold", "Anomaly Score"])
plt.savefig(f"{PCAP_PATH.split('/')[-1][:-5]}_re.png")
plt.close()
