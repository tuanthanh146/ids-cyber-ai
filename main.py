import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="AI IDS System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the IDS model")
    train_parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    train_parser.add_argument("--data", help="Path to training data (CSV)")
    train_parser.add_argument("--output", help="Directory to save artifacts")
    train_parser.add_argument("--demo", action="store_true", help="Run with synthetic data")
    
    # Serve Command
    serve_parser = subparsers.add_parser("serve", help="Start the Inference API")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    # Dashboard Command
    dash_parser = subparsers.add_parser("dashboard", help="Start the Streamlit Dashboard")

    # Process PCAP Command
    process_parser = subparsers.add_parser("process", help="Convert PCAP to CSV")
    process_parser.add_argument("--input", required=True, help="Path to input .pcap file")
    process_parser.add_argument("--output", required=True, help="Path to output .csv file")
    process_parser.add_argument("--label", type=int, default=0, help="Label for this file (0=Benign, 1=Attack)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from scripts.train import train_pipeline
        train_pipeline(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
            demo_mode=args.demo
        )
        
    elif args.command == "serve":
        import uvicorn
        # We refer to the file inference.api and the app object `app`
        uvicorn.run("inference.api:app", host=args.host, port=args.port, reload=True)
        
    elif args.command == "dashboard":
        os.system("streamlit run dashboard/dashboard_app_v3.py")

    elif args.command == "process":
        from scripts.process_pcap import process_pcap
        process_pcap(args.input, args.output, args.label)
        
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()
