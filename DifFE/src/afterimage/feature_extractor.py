"""AfterImage feature extractor: pcap → per-packet feature vectors."""
import csv
import os
import os.path
import platform
import subprocess
import sys

import numpy as np
from scapy.all import *

from . import net_stat as ns


class FE:
    def __init__(self, file_path, limit=np.inf):
        self.path = file_path
        self.limit = limit
        self.parse_type = None
        self.curPacketIndx = 0
        self.tsvin = None
        self.scapyin = None

        self.__prep__()

        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return r'C:\Program Files\Wireshark\tshark.exe'
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def __prep__(self):
        if not os.path.isfile(self.path):
            print("File: " + self.path + " does not exist")
            raise Exception()

        type = self.path.split('.')[-1]

        self._tshark = self._get_tshark_path()

        if type == "tsv":
            self.parse_type = "tsv"
        elif type == "pcap" or type == 'pcapng':
            if os.path.isfile(self._tshark):
                self.pcap2tsv_with_tshark()
                self.path += ".tsv"
                self.parse_type = "tsv"
            else:
                print("tshark not found. Trying scapy...")
                self.parse_type = "scapy"
        else:
            print("File: " + self.path + " is not a tsv or pcap file")
            raise Exception()

        if self.parse_type == "tsv":
            maxInt = sys.maxsize
            decrement = True
            while decrement:
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt / 10)
                    decrement = True

            print("counting lines in file...")
            num_lines = sum(1 for line in open(self.path))
            print("There are " + str(num_lines) + " Packets.")
            self.limit = min(self.limit, num_lines-1)
            self.tsvinf = open(self.path, 'rt', encoding="utf8")
            self.tsvin = csv.reader(self.tsvinf, delimiter='\t')
            row = self.tsvin.__next__()  # move iterator past header

        else:  # scapy
            print("Reading PCAP file via Scapy...")
            self.scapyin = rdpcap(self.path)
            self.limit = len(self.scapyin)
            print("Loaded " + str(len(self.scapyin)) + " Packets.")

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            if self.parse_type == 'tsv':
                self.tsvinf.close()
            return []

        if self.parse_type == "tsv":
            row = self.tsvin.__next__()
            IPtype = np.nan
            timestamp = row[0]
            framelen = row[1]
            srcIP = ''
            dstIP = ''
            if row[4] != '':  # IPv4
                srcIP = row[4]
                dstIP = row[5]
                IPtype = 0
            elif row[17] != '':  # ipv6
                srcIP = row[17]
                dstIP = row[18]
                IPtype = 1
            srcproto = row[6] + row[8]
            dstproto = row[7] + row[9]
            srcMAC = row[2]
            dstMAC = row[3]
            if srcproto == '':  # L2/L1 level protocol
                if row[12] != '':  # ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = row[14]
                    dstIP = row[16]
                    IPtype = 0
                elif row[10] != '':  # ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':
                    srcIP = row[2]
                    dstIP = row[3]

        elif self.parse_type == "scapy":
            packet = self.scapyin[self.curPacketIndx]
            IPtype = np.nan
            timestamp = packet.time
            framelen = len(packet)
            if packet.haslayer(IP):
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                IPtype = 0
            elif packet.haslayer(IPv6):
                srcIP = packet[IPv6].src
                dstIP = packet[IPv6].dst
                IPtype = 1
            else:
                srcIP = ''
                dstIP = ''

            if packet.haslayer(TCP):
                srcproto = str(packet[TCP].sport)
                dstproto = str(packet[TCP].dport)
            elif packet.haslayer(UDP):
                srcproto = str(packet[UDP].sport)
                dstproto = str(packet[UDP].dport)
            else:
                srcproto = ''
                dstproto = ''

            srcMAC = packet.src
            dstMAC = packet.dst
            if srcproto == '':
                if packet.haslayer(ARP):
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = packet[ARP].psrc
                    dstIP = packet[ARP].pdst
                    IPtype = 0
                elif packet.haslayer(ICMP):
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':
                    srcIP = packet.src
                    dstIP = packet.dst
        else:
            return []

        self.curPacketIndx += 1

        try:
            return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                             int(framelen), float(timestamp))
        except Exception as e:
            print(e)
            return []

    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = ("-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst "
                  "-e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code "
                  "-e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 "
                  "-e ipv6.src -e ipv6.dst")
        cmd = ('"' + self._tshark + '" -r ' + self.path + ' -T fields ' + fields +
               ' -E header=y -E occurrence=f > ' + self.path + ".tsv")
        subprocess.call(cmd, shell=True)
        print("tshark parsing complete. File saved as: " + self.path + ".tsv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
