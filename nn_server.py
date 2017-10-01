'''
Server for deployment of caffe-based neural networks
'''
import sys
import json
import yaml
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import io
import os
import caffe
import tempfile

import utils
from network import Network

# Load configuration file
config = {
	"server": {
		"host": "localhost",
		"port": 8000,
		"mode": "cpu"
	}
}
config_file = sys.argv[1]
with open(config_file) as f:
	utils.dict_merge(config, yaml.load(f))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if config["server"]["mode"] == "gpu":
	caffe.set_mode_gpu()

# Load networks
nets = {}
for name, ncfg in config["networks"].items():
	logging.info("Loading network %s" % name)
	nets[name] = Network(ncfg)

class Handler(BaseHTTPRequestHandler):
	def do_GET(self):
		self.response(501, "text/plain", "GET requests are not supported")

	def do_POST(self):
		path = self.path.replace("/","")
		if path not in nets:
			self.response(404, "text/plain", "Net %s not found" % path)
			return
		
		# get the file name (as raw post data)
		data = self.rfile.read(int(self.headers['Content-Length']))

		try:
			# load the image into numpy array
			f = tempfile.NamedTemporaryFile(delete=False)
			f.write(data)
			f.close()
			im = caffe.io.load_image(f.name)
			os.unlink(f.name)

			# do forward pass
			out = nets[path].forward(im)
		
			# send response
			self.response(200, 'application/json', json.dumps(out.tolist()))
		except Exception as e:
			logging.error("Exception: %s", str(e))
			self.response(500, 'text/plain', str(e))

	def response(self, code, content_type, body):
		# Construct a server response.
		self.send_response(code)
		self.send_header('Content-type', content_type)
		self.end_headers()
		self.wfile.write(bytes(body, 'utf8'))
		

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

host, port = config["server"]["host"], config["server"]["port"]
logging.info('Starting server on %s:%d', host, port)
httpd = ThreadedHTTPServer((host, port), Handler)
httpd.serve_forever()
