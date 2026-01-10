import time
import random
import pandas as pd
from typing import Generator
from utils.logger import get_logger

logger = get_logger(__name__)

class StreamLoader:
    """
    Simulates real-time packet capture or reading from a Message Queue (Kafka/RabbitMQ).
    """
    def __init__(self, source: str = "mock"):
        self.source = source
    
    def stream_packets(self) -> Generator[pd.DataFrame, None, None]:
        """
        Yields single-row DataFrames simulating incoming packets.
        """
        logger.info(f"Starting Stream Loader from source: {self.source}")
        while True:
            # Simulate latency
            time.sleep(1)
            
            # Mock Data
            mock_data = {
                "timestamp": [time.time()],
                "src_ip": [f"192.168.1.{random.randint(1, 255)}"],
                "dst_ip": ["10.0.0.1"],
                "protocol": [random.choice(["TCP", "UDP", "ICMP"])],
                "length": [random.randint(64, 1500)],
                "flags": [random.choice(["SYN", "ACK", "FIN", "RST"])]
            }
            yield pd.DataFrame(mock_data)
