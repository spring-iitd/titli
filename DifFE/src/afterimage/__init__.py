"""AfterImage incremental-statistics feature extractor (Kitsune).

Usage:
    from afterimage import FE
    fe = FE("/path/to/capture.pcap")
    headers = fe.nstat.getNetStatHeaders()
    while True:
        v = fe.get_next_vector()
        if not v:
            break
        # v is a list of 100 floats
"""
from .feature_extractor import FE

__all__ = ["FE"]
