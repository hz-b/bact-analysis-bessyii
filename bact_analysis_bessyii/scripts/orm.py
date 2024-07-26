from bact_analysis_bessyii.orm import app
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="postprocess bba measurement data")
    parser.add_argument("--uid", metavar="uid", required=True)
    parser.add_argument("--read-from-file", metavar="read_from_file", default=False)
    parser.add_argument("--catalog-name", metavar="catalog_name", default="heavy_local")
    args = parser.parse_args()
    app.main(args.uid, read_from_file=args.read_from_file, catalog_name=args.catalog_name)
