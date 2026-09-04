#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

print_help() {
    echo ""
    echo "DynamicEventSSL Dataset Manager"
    echo ""
    echo "Usage:"
    echo ""
    echo "  ./scripts/dataset_manager.sh download <dataset> <subset>"
    echo "  ./scripts/dataset_manager.sh inspect <dataset>"
    echo "  ./scripts/dataset_manager.sh process <dataset>"
    echo "  ./scripts/dataset_manager.sh verify <dataset>"
    echo "  ./scripts/dataset_manager.sh clean <dataset>"
    echo ""
    echo "Examples:"
    echo ""
    echo "  ./scripts/dataset_manager.sh download mvsec indoor1"
    echo "  ./scripts/dataset_manager.sh download mvsec all"
    echo "  ./scripts/dataset_manager.sh inspect mvsec"
    echo "  ./scripts/dataset_manager.sh process mvsec"
    echo "  ./scripts/dataset_manager.sh verify mvsec"
    echo "  ./scripts/dataset_manager.sh clean mvsec"
    echo ""
}

if [ $# -lt 1 ]; then
    print_help
    exit 0
fi

COMMAND=$1

case "$COMMAND" in

download)

python "$SCRIPT_DIR/download_mvsec.py" \
    --subset "$3"

;;

inspect)

python "$SCRIPT_DIR/inspect_mvsec.py"

;;

process)

python "$SCRIPT_DIR/convert_mvsec.py"

;;

verify)

python "$SCRIPT_DIR/verify_mvsec.py"

;;

clean)

python "$SCRIPT_DIR/clean_mvsec.py"

;;

help)

print_help

;;

*)

echo "Unknown command."

print_help

;;

esac