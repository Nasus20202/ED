#!/bin/bash

OUTPUT_DIR="data"
echo "Downloading data files to $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR
curl "https://fil.rednik.top/nextcloud/s/wb48fjoQ2YkTJEt/download/flights.csv" --output $OUTPUT_DIR/flights.csv
curl "https://fil.rednik.top/nextcloud/s/n5KoosjCySPxEAZ/download/airports.csv" --output $OUTPUT_DIR/airports.csv
curl "https://fil.rednik.top/nextcloud/s/QZzsQL9bKf9M2Rd/download/airlines.csv" --output $OUTPUT_DIR/airlines.csv
