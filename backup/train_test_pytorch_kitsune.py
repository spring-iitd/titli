from walle.ids import PyTorchKitsune

model = PyTorchKitsune(num_features=100)
threshold = model.train(pcap_path="../PANDA/PANDA/data/benign/weekday.pcap")
print(threshold)
model.infer(pcap_path="../PANDA/PANDA/data/benign/weekday.pcap")
model.plot_kitsune(threshold)