"""Base feature extractor for network traffic analysis."""

import json
import pickle
from abc import ABC, abstractmethod
from io import TextIOWrapper
from pathlib import Path

import numpy as np
from scapy.all import PcapReader

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Path, TextIOWrapper, and numpy types."""
    
    def default(self, obj):
        """Convert non-serializable objects to JSON-serializable format.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, TextIOWrapper):
            return obj.name
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

class BaseTrafficFeatureExtractor(ABC):
    """Abstract base class for network traffic feature extraction.
    
    This class provides the framework for extracting features from network traffic
    captured in PCAP files. Subclasses must implement the abstract methods to define
    specific feature extraction logic.
    
    Attributes:
        file_path (str): Path to the input PCAP file
        state: Optional pre-existing state to continue from previous extraction
        feature_file: Output file handle for extracted features
        meta_file: Output file handle for traffic vector metadata
        count (int): Number of packets successfully processed
        skipped (int): Number of packets skipped during processing
    """
    
    def __init__(self, file_path, dataset_name=None, state=None, **kwargs):
        """Initialize the feature extractor.
        
        Args:
            file_path (str): Path to the PCAP file to process
            dataset_name (str, optional): Name of the dataset (deprecated, not used)
            state (NetStat, optional): Pre-existing state to continue from. If None,
                starts fresh extraction
            **kwargs: Additional arguments for subclass customization
        """
        self.file_path = file_path
        self.state = state

    @abstractmethod
    def update(self, traffic_vector):
        """Update the feature extractor with a new traffic vector.
        
        This method processes a traffic vector and updates the internal state
        of the feature extractor, returning the computed features.

        Args:
            traffic_vector (np.ndarray): Traffic vector extracted from packet(s)
            
        Returns:
            np.ndarray: Extracted features corresponding to the traffic vector
        """
        pass

    @abstractmethod
    def peek(self, traffic_vectors):
        """Simulate feature extraction without updating internal state.
        
        This method performs a "dry run" of feature extraction without modifying
        the extractor's state. Useful for adversarial attacks or what-if analysis.

        Args:
            traffic_vectors (list): List of traffic vectors to process
            
        Returns:
            list: List of features corresponding to each traffic vector
        """
        pass

    @abstractmethod
    def get_traffic_vector(self, packet):
        """Extract traffic vector from a raw network packet.

        Args:
            packet (scapy.packet.Packet): Input packet to process
            
        Returns:
            np.ndarray or None: Extracted traffic vector, or None if packet should be skipped
        """
        pass

    def setup(self, output_path=None):
        """Set up the feature extractor for processing.
        
        Opens the input PCAP file, creates output CSV files for features and metadata,
        and initializes processing counters and state management flags.
        
        Args:
            output_path (str or Path, optional): Custom path for the output feature file.
                If None, creates the feature file in the same directory as the input PCAP
                with a .csv extension. The metadata file will be created with a '_meta.csv'
                suffix in the same directory.
                
        Side Effects:
            - Opens input PCAP file for reading
            - Creates and opens feature and metadata CSV files for writing
            - Initializes count, skipped counters to 0
            - Sets state management flags based on whether pre-existing state was provided
        """
        self.path = Path(self.file_path)
        
        if output_path is not None:
            feature_file = Path(output_path)
            meta_file = feature_file.parent / (feature_file.stem + "_meta.csv")
        else:
            feature_file = self.path.with_suffix(".csv")
            meta_file = self.path.parent / (self.path.stem + "_meta.csv")

        self.feature_file = open(feature_file, "w")
        self.meta_file = open(meta_file, "w")
        self.feature_file.write(",".join(self.get_headers()) + "\n")
        self.meta_file.write(",".join(self.get_meta_headers()) + "\n")

        self.count = 0
        self.skipped = 0

        self.input_pcap = PcapReader(str(self.path))

        if self.state is not None:
            self.reset_state = False
            self.save_state = False
            self.offset_timestamp = True
        else:
            self.reset_state = True
            self.save_state = True
            self.offset_timestamp = False
        self.offset_time = None

    @abstractmethod
    def get_headers(self):
        """Get the column names for the feature CSV file.
        
        Returns:
            list[str]: List of feature column names
        """
        pass

    @abstractmethod
    def get_meta_headers(self):
        """Get the column names for the metadata/traffic vector CSV file.
        
        Returns:
            list[str]: List of metadata column names
        """
        pass

    def teardown(self):
        """Clean up resources and finalize feature extraction.
        
        Closes all open files (PCAP input, feature output, metadata output),
        prints processing statistics, and saves the extractor state if configured.
        
        Side Effects:
            - Closes all open file handles
            - Prints processing statistics (skipped, processed, written counts)
            - Saves state to 'state.pkl' in the PCAP directory if save_state is True
        """
        self.meta_file.close()
        self.feature_file.close()
        self.input_pcap.close()
        
        print(
            f"skipped: {self.skipped} processed: {self.count+self.skipped} written: {self.count}"
        )

        if self.save_state:
            state_path = self.path.parent / "state.pkl"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "wb") as pf:
                pickle.dump(self.state, pf)

    @abstractmethod
    def extract_features(self):
        """Main entry point for feature extraction from PCAP file.
        
        This method should implement the complete feature extraction pipeline:
        reading packets from the input PCAP, extracting traffic vectors,
        computing features, and writing results to output files.
        
        Must call setup() before and teardown() after processing.
        """
        pass
