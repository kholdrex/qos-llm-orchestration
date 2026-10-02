"""Forward a JSON command from stdin to the in-container agent and print its reply."""

import sys
import urllib.request

if __name__ == "__main__":
    request = urllib.request.Request("http://127.0.0.1:8080", data=sys.stdin.buffer.read())
    sys.stdout.buffer.write(urllib.request.urlopen(request, timeout=900).read())
