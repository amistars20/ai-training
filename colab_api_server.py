"""
Threaded HTTP command server for remote Colab control via Cloudflare Tunnel.
"""
import subprocess, json, os, threading, time, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

JOBS = {}
JOBS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)
        if p.path == '/ping':
            return self.json({'status': 'ok', 'cwd': os.getcwd()})
        if p.path == '/read':
            path = q.get('path', [None])[0]
            if not path:
                return self.json({'error': 'no path'}, 400)
            n = os.path.normpath(path)
            if not os.path.exists(n):
                return self.json({'error': 'not found'}, 404)
            off = int(q.get('offset', [0])[0])
            limit = int(q.get('limit', [200000])[0])
            with open(n, 'r', errors='replace') as f:
                f.seek(off)
                c = f.read(limit)
            return self.json({'content': c, 'offset': off + len(c),
                              'size': os.path.getsize(n)})
        if p.path.startswith('/file/'):
            path = p.path[6:]
            if not os.path.exists(path):
                return self.json({'error': 'not found'}, 404)
            with open(path, 'rb') as f:
                d = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Disposition',
                             f'attachment; filename="{os.path.basename(path)}"')
            self.send_header('Content-Length', str(len(d)))
            self.end_headers()
            self.wfile.write(d)
            return
        if p.path.startswith('/job/'):
            jid = p.path[5:]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if not job:
                return self.json({'error': 'not found'}, 404)
            j = dict(job)
            if job.get('log') and os.path.exists(job['log']):
                with open(job['log'], 'r', errors='replace') as f:
                    j['output'] = f.read()[-200000:]
            return self.json(j)
        return self.json({'error': 'not found'}, 404)

    def do_POST(self):
        p = urlparse(self.path)
        l = int(self.headers.get('Content-Length', 0))
        b = json.loads(self.rfile.read(l)) if l else {}
        if p.path == '/exec':
            try:
                r = subprocess.run(b.get('command', ''), shell=True,
                                   capture_output=True, text=True,
                                   timeout=b.get('timeout', 120))
                return self.json({
                    'stdout': r.stdout[-200000:],
                    'stderr': r.stderr[-200000:],
                    'returncode': r.returncode,
                })
            except subprocess.TimeoutExpired:
                return self.json({'error': 'timeout'}, 408)
            except Exception as e:
                return self.json({'error': str(e)}, 500)
        if p.path == '/exec-bg':
            cmd = b.get('command', '')
            lp = b.get('log', '/tmp/colab_job.log')
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            jid = f'j{int(time.time())}'
            with JOBS_LOCK:
                JOBS[jid] = {'status': 'running', 'command': cmd, 'log': lp}

            def run(j, c, lpath):
                try:
                    with open(lpath, 'w') as f:
                        subprocess.Popen(c, shell=True, stdout=f,
                                         stderr=subprocess.STDOUT,
                                         text=True).wait()
                    with JOBS_LOCK:
                        JOBS[j]['status'] = 'done'
                        JOBS[j]['returncode'] = 0
                except Exception as e:
                    with JOBS_LOCK:
                        JOBS[j]['status'] = 'error'
                    with open(lpath, 'a') as f:
                        f.write(f'\nJOB ERROR: {e}\n')

            threading.Thread(target=run, args=(jid, cmd, lp),
                             daemon=True).start()
            return self.json({'job_id': jid, 'log': lp})
        return self.json({'error': 'not found'}, 404)

    def json(self, d, s=200):
        self.send_response(s)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(d).encode())

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'API server on 0.0.0.0:{port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
