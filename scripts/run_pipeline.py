import time
import json
import os
import argparse
import pandas as pd
import numpy as np
import sys
from collections import defaultdict
from typing import Dict, List, Any

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Scapy for packet capture
from scapy.all import sniff, rdpcap, IP, TCP, UDP, ICMP, Ether

# Project Modules
from utils.logger import setup_logging, get_logger
from utils.serialization import load_object
from models.fusion import DecisionFusion
from feature_extraction.extractor import IDSFeatureExtractor
from preprocessing.preprocessor import IDSPreprocessor

logger = get_logger(__name__)

class FlowAggregator:
    """
    Aggregates packets into flows based on 5-tuple.
    Simple window-based aggregation (e.g. 5 seconds).
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.active_flows = defaultdict(lambda: {
            "start_time": 0, "last_time": 0, 
            "src_ip": "", "dst_ip": "", 
            "protocol": "", 
            "packets": 0, "bytes": 0, 
            "flags": set()
        })
        self.last_flush = time.time()

    def process_packet(self, packet) -> None:
        if not packet.haslayer(IP):
            return

        ip = packet[IP]
        src = ip.src
        dst = ip.dst
        proto_num = ip.proto
        
        # Map Protocol Number to Name
        proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
        proto = proto_map.get(proto_num, "OTHER")
        
        if packet.haslayer(TCP):
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif packet.haslayer(UDP):
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        else:
            sport = 0
            dport = 0
            
        # Flow Key: 5-tuple (src, dst, proto, sport, dport)
        flow_key = (src, dst, proto, sport, dport)
        
        flow = self.active_flows[flow_key]
        if flow["packets"] == 0:
            flow["start_time"] = packet.time
            flow["src_ip"] = src
            flow["dst_ip"] = dst
            flow["protocol"] = proto
            flow["src_port"] = sport
            flow["dst_port"] = dport
        
        flow["last_time"] = packet.time
        flow["packets"] += 1
        flow["bytes"] += len(packet)
        
        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            # Scapy flags are complex objects, simplify to string chars
            # S=SYN, A=ACK, F=FIN, R=RST, P=PSH, U=URG
            if 'S' in flags: flow["flags"].add("SYN")
            if 'A' in flags: flow["flags"].add("ACK")
            if 'F' in flags: flow["flags"].add("FIN")
            if 'R' in flags: flow["flags"].add("RST")
            if 'P' in flags: flow["flags"].add("PSH")

    def flush(self) -> List[Dict]:
        """
        Returns list of flow dicts ready for DataFrame conversion
        and clears active flows.
        """
        flows = []
        for key, f in self.active_flows.items():
            duration = f["last_time"] - f["start_time"]
            # Formatting for model input
            # Flags: OneHot style or just a representative string?
            # Model expects 'flags' column (string categorical)
            # We pick the most "significant" flag for simplifiction or join them
            flag_str = "normal"
            if "RST" in f["flags"]: flag_str = "RST"
            elif "SYN" in f["flags"] and "ACK" not in f["flags"]: flag_str = "SYN"
            elif "FIN" in f["flags"]: flag_str = "FIN"
            elif "ACK" in f["flags"]: flag_str = "ACK"
            
            flows.append({
                "timestamp": f["start_time"],
                "src_ip": f["src_ip"],
                "dst_ip": f["dst_ip"],
                "protocol": f["protocol"], # Preprocessor expects 'protocol', not 'proto'
                "proto": f["protocol"], # Keep 'proto' for compatibility if needed, duplicates are fine in dict but DataFrame needs correct columns for Transformer
                "length": f["bytes"], 
                "flags": flag_str,
                "duration": duration,
                # --- Missing Features (Filled for Demo Compatibility) ---
                "ts": f["start_time"],
                "service": "http" if f["protocol"] == "TCP" else "dns", 
                "conn_state": "SF" if "FIN" in f["flags"] else "S0", 
                "orig_pkts": f["packets"],
                "resp_pkts": 0, 
                "pkts_total": f["packets"],
                "orig_bytes": f["bytes"],
                "resp_bytes": 0,
                "bytes_total": f["bytes"],
                "pkts_per_sec": f["packets"] / (duration + 1e-6),
                "bytes_per_sec": f["bytes"] / (duration + 1e-6),
                "src_port": f["src_port"], 
                "dst_port": f["dst_port"],
                "iat_mean": 0, "iat_std": 0,
                "syn_count": 1 if "SYN" in f["flags"] else 0,
                "ack_count": 1 if "ACK" in f["flags"] else 0,
                "rst_count": 1 if "RST" in f["flags"] else 0,
                "risk_hint": 0 
            })
        
        self.active_flows.clear()
        return flows

class Pipeline:
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = config_path
        self._load_artifacts()
        self.fusion = DecisionFusion()
        self.aggregator = FlowAggregator(window_size=5)

    def _load_artifacts(self):
        logger.info("Loading Models...")
        base = "models/artifacts"
        
        # Load XGBoost (Advanced)
        adv_path = os.path.join(base, "model_advanced.joblib")
        if os.path.exists(adv_path):
            self.model_xgb = load_object(adv_path)
            logger.info("Loaded XGBoost: Advanced")
        else:
            self.model_xgb = load_object(os.path.join(base, "model.joblib"))
            logger.info("Loaded XGBoost: Basic")

        # Load Isolation Forest
        anom_path = "models/anomaly/if_model.joblib"
        if os.path.exists(anom_path):
            self.model_if = load_object(anom_path)
            self.if_preprocessor = load_object("models/anomaly/if_preprocessor.joblib")
            logger.info("Loaded IsolationForest")
        else:
            logger.warning("Isolation Forest not found. Anomaly Detection disabled.")
            self.model_if = None

        # Load Common Transforms
        self.extractor = load_object(os.path.join(base, "extractor.joblib"))
        self.preprocessor = load_object(os.path.join(base, "preprocessor.joblib"))

    def run(self, source="pcap", pcap_file=None, iface=None):
        logger.info(f"Starting Pipeline (Source: {source})...")
        
        def packet_callback(pkt):
            self.aggregator.process_packet(pkt)
            
            # Check flush aggregation
            # Simulating batch processing relative to wall time or packet count
            # Here we just flush every 100 packets for demo or time based
            if len(self.aggregator.active_flows) >= 10: 
                self.process_batch()
        
        # Capture Loop
        try:
            if source == "pcap":
                if not pcap_file: raise ValueError("PCAP file required")
                logger.info(f"Replaying {pcap_file}...")
                scapy_pkts = rdpcap(pcap_file)
                for pkt in scapy_pkts:
                    packet_callback(pkt)
                # Flush remaining
                self.process_batch()
            elif source == "live":
                logger.info(f"Listening on {iface}...")
                sniff(iface=iface, prn=packet_callback, store=0)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user.")
            
    def process_batch(self):
        flows = self.aggregator.flush()
        if not flows:
            return

        df = pd.DataFrame(flows)
        logger.info(f"Processing Batch: {len(df)} flows")
        
        # 1. Pipeline Transformations (XGBoost)
        try:
            feat_x = self.extractor.transform(df)
            proc_x = self.preprocessor.transform(feat_x)
            
            # XGB Prediction
            # Align features if model has feature selection enabled
            if hasattr(self.model_xgb, "feature_names_in_"):
                 expected_cols = self.model_xgb.feature_names_in_
                 # Ensure strict alignment (reorder/select)
                 # Missing columns = 0, Extra columns = Drop
                 proc_x = proc_x.reindex(columns=expected_cols, fill_value=0)
            
            xgb_preds = self.model_xgb.predict(proc_x)
            xgb_probas = self.model_xgb.predict_proba(proc_x)[:, 1]
            
            # 2. Anomaly Detection
            anom_scores = np.zeros(len(df))
            if self.model_if:
                # Reuse extractor output, but maybe need IF specific preprocessor?
                # train_anomaly used ids_preprocessor? 
                # Wait, train_anomaly saved 'if_preprocessor.joblib', which might be same but fitted differently?
                # Let's use if_preprocessor if available
                # Note: Extractor is shared standard.
                if_proc_x = self.if_preprocessor.transform(feat_x)
                anom_scores = self.model_if.decision_function(if_proc_x)
            
            # 3. Fusion & Alerting
            alerts = []
            for i in range(len(df)):
                res = self.fusion.fuse(
                    xgb_label=xgb_preds[i],
                    xgb_proba=xgb_probas[i],
                    anomaly_score=anom_scores[i]
                )
                
                # Log ALL flows for dashboard visualization testing
                # In production, use ["HIGH", "CRITICAL"]
                if True: # res['alert_level'] in ["HIGH", "CRITICAL"]:
                    alert = {
                        "timestamp": str(df.iloc[i]['timestamp']),
                        "src": df.iloc[i]['src_ip'],
                        "dst": df.iloc[i]['dst_ip'],
                        "src_port": int(df.iloc[i]['src_port']),
                        "dst_port": int(df.iloc[i]['dst_port']),
                        "proto": df.iloc[i]['protocol'],
                        "xgb_prob": float(xgb_probas[i]),
                        "anomaly_score": float(anom_scores[i]),
                        **res
                    }
                    alerts.append(alert)
                    logger.warning(f"ALERT: {json.dumps(alert)}")
            
            # Log to file
            if alerts:
                os.makedirs("logs", exist_ok=True)
                with open("logs/alerts.jsonl", "a") as f:
                    for a in alerts:
                        f.write(json.dumps(a) + "\n")
                        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pcap", "live"], default="pcap")
    parser.add_argument("--source", help="Path to PCAP or Interface Name")
    args = parser.parse_args()
    
    pipeline = Pipeline()
    
    # Generate mock PCAP if testing and no file provided
    if args.mode == "pcap" and (not args.source or not os.path.exists(args.source)):
        logger.info("No PCAP provided. Generating mock PCAP for testing...")
        from scapy.all import wrpcap, Ether, IP, TCP
        pkts = []
        # Normal
        for _ in range(50):
            pkts.append(Ether()/IP(src="192.168.1.5", dst="10.0.0.5")/TCP(flags="S"))
        # Attack
        for _ in range(10):
            pkts.append(Ether()/IP(src="1.1.1.1", dst="10.0.0.5")/TCP(flags="S"))
        
        args.source = "test_traffic.pcap"
        wrpcap(args.source, pkts)
        
    pipeline.run(source=args.mode, pcap_file=args.source if args.mode=="pcap" else None, iface=args.source if args.mode=="live" else None)
