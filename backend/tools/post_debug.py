"""post_debug.py
Small helper to POST an image to /detect/debug and pretty-print the JSON response.
Usage:
  python post_debug.py --file path/to/image.jpg --url http://127.0.0.1:8000/detect/debug

Requires: requests (pip install requests)
"""
import argparse
import sys

try:
    import requests
except Exception:
    print("The 'requests' library is required. Install with: pip install requests")
    sys.exit(2)

import json


def main():
    parser = argparse.ArgumentParser(description="POST an image to /detect/debug and print JSON response")
    parser.add_argument('--file', '-f', required=True, help='Path to image file')
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8000/detect/debug', help='Debug endpoint URL')
    parser.add_argument('--timeout', '-t', type=float, default=30.0, help='Request timeout in seconds')
    args = parser.parse_args()

    files = {'file': open(args.file, 'rb')}
    try:
        print(f"POSTing {args.file} -> {args.url}")
        resp = requests.post(args.url, files=files, timeout=args.timeout)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(3)
    finally:
        try:
            files['file'].close()
        except Exception:
            pass

    print(f"Status: {resp.status_code}")
    ctype = resp.headers.get('Content-Type', '')
    if 'application/json' in ctype:
        try:
            j = resp.json()
            print(json.dumps(j, indent=2))
        except Exception as e:
            print("Failed to parse JSON response:", e)
            print(resp.text)
    else:
        # Non-JSON (e.g., HTML form on GET) - print raw
        print(resp.text)


if __name__ == '__main__':
    main()
