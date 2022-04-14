devcnt=$1
scrname=$2
shift
shift
CUDA_VISIBLE_DEVICES="$devcnt" /home/federico.pedeni/internship_venv/bin/python3 "$scrname" $@