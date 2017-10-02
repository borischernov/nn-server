# nn-server

nn-server is a simple HTTP-based server for deployment of (pre-trained) Caffe networks.

Usage:
> python3 ./nn_server.py config.yml

Sample config.yml is included.

Requests are done via HTTP POST. Request path corresponds to network name in config file. Input data (image) is passed as raw POST data. Server returns output blob data as flattened JSON-encoded array.

Currently nn-server only supports images as input data.

Request example with curl:
> curl --data-binary "@/path/to/image.jpg" http://127.0.0.1:8000/googlenet

