from .export import run_export
from .train import run_train
from .predict import run_predict
from .cluster import run_cluster
import argparse

def run_pipeline(args):
    print("--- ML Pipeline: Export ---", flush=True)
    args.force = False
    run_export(args)
    
    print("--- ML Pipeline: Train ---", flush=True)
    run_train(args)
    
    print("--- ML Pipeline: Predict ---", flush=True)
    run_predict(args)
    
    print("--- ML Pipeline: Cluster ---", flush=True)
    args.size = 0
    run_cluster(args)
    
    print("--- ML Pipeline: Complete ---", flush=True)
