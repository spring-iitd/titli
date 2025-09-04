import os
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



def create_directories(dataset_name):
    # Define the base path for artifacts
    base_path = os.path.join("./artifacts", dataset_name)
    
    # Define the subdirectories to be created
    subdirectories = ["plots", "models", "objects"]
    
    # Iterate through the subdirectories and create them if they don't exist
    for subdirectory in subdirectories:
        path = os.path.join(base_path, subdirectory)
        os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Create the subdirectories inside 'plots' for specific plot types
    plots_path = os.path.join(base_path, "plots")
    plot_subdirectories = ["roc", "anomaly", "confusion_matrix"]
    for plot_subdir in plot_subdirectories:
        plot_path = os.path.join(plots_path, plot_subdir)
        os.makedirs(plot_path, exist_ok=True)
    
    # Create the subdirectories inside 'objects' for metrics and reconstruction_error
    objects_path = os.path.join(base_path, "objects")
    object_subdirectories = ["metrics"]
    for object_subdir in object_subdirectories:
        object_path = os.path.join(objects_path, object_subdir)
        os.makedirs(object_path, exist_ok=True)

    print(f"Folder structure created for dataset: {dataset_name}")

