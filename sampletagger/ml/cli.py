import argparse
import sys
from .export import run_export
from .train import run_train
from .predict import run_predict
from .report import run_report
from .pipeline import run_pipeline
from .clap import run_clap_embed, run_clap_eval
def main():
    ap = argparse.ArgumentParser(description="Machine Learning pipeline for sample-tagger")
    subparsers = ap.add_subparsers(dest="command", required=True)
    
    # Export
    p_exp = subparsers.add_parser("export", help="Export embeddings and labels to .npz")
    p_exp.add_argument("db", help="Path to samples.db")
    p_exp.add_argument("--force", action="store_true", help="Force re-export even if cached")
    
    # Train
    p_train = subparsers.add_parser("train", help="Train the classifier head")
    p_train.add_argument("db", help="Path to samples.db")
    
    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict and write back to DB")
    p_pred.add_argument("db", help="Path to samples.db")
    
    # Report
    p_rep = subparsers.add_parser("report", help="Evaluate accuracy and coverage")
    p_rep.add_argument("db", help="Path to samples.db")

    # Cluster

    # Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full pipeline (export, train, predict)")
    p_pipe.add_argument("db", help="Path to samples.db")

    # CLAP pilot / full
    p_ce = subparsers.add_parser("clap-embed", help="Embed val/train files with CLAP")
    p_ce.add_argument("db", help="Path to samples.db")
    p_ce.add_argument("--full", action="store_true", help="Embed all valid files in the DB")
    
    p_cv = subparsers.add_parser("clap-eval", help="Evaluate CLAP zero-shot vs trained")
    p_cv.add_argument("db", help="Path to samples.db")

    args = ap.parse_args()

    if args.command == "export":
        run_export(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "report":
        run_report(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "clap-embed":
        run_clap_embed(args)
    elif args.command == "clap-eval":
        run_clap_eval(args)
    else:
        ap.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
