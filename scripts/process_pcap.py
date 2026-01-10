from scapy.all import rdpcap, IP, TCP, UDP, ICMP
import pandas as pd
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

def process_pcap(input_file, output_file, label=0):
    logger.info(f"Reading PCAP file: {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return

    try:
        packets = rdpcap(input_file)
    except Exception as e:
        logger.error(f"Failed to read PCAP: {e}")
        return

    logger.info(f"Loaded {len(packets)} packets. Extracting features...")
    
    data = []
    
    for pkt in packets:
        if IP not in pkt:
            continue
            
        row = {
            'timestamp': float(pkt.time),
            'src_ip': pkt[IP].src,
            'dst_ip': pkt[IP].dst,
            'protocol': 'OTHER',
            'length': len(pkt),
            'flags': None,
            'ttl': pkt[IP].ttl,
            'label': label
        }
        
        # Determine Protocol and Specific Fields
        if TCP in pkt:
            row['protocol'] = 'TCP'
            # pkt[TCP].flags is an object, convert to string
            row['flags'] = str(pkt[TCP].flags)
        elif UDP in pkt:
            row['protocol'] = 'UDP'
        elif ICMP in pkt:
            row['protocol'] = 'ICMP'
        else:
            # Map number to protocol name if needed, or keep OTHER
            row['protocol'] = str(pkt[IP].proto)
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Ensure Output Directory Exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    logger.info(f"Saving {len(df)} rows to {output_file}")
    df.to_csv(output_file, index=False)
    logger.info("Processing complete.")

if __name__ == "__main__":
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Convert PCAP to CSV for IDS Training")
    parser.add_argument("--input", required=True, help="Path to input .pcap file")
    parser.add_argument("--output", required=True, help="Path to output .csv file")
    parser.add_argument("--label", type=int, default=0, help="Label for this file (0=Benign, 1=Attack)")
    
    args = parser.parse_args()
    
    process_pcap(args.input, args.output, args.label)
