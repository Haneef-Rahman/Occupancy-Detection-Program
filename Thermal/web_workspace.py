#!/usr/bin/env python3
"""Standalone browser workspace for Mac/Windows Lepton capture and replay."""
from pathlib import Path
import argparse,sys,threading,webbrowser


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8780)
    parser.add_argument('--weights',help='Thermal omega model; defaults to newest models/vN/best.pt')
    parser.add_argument('--thermal-root',type=Path,default=Path(__file__).resolve().parent,
                        help='Thermal source folder (normally detected automatically)')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    root=args.thermal_root.resolve()
    if not (root/'tracking_router.py').is_file():parser.error('Thermal folder not found; supply --thermal-root.')
    sys.path.insert(0,str(root))
    from thermal_web.server import Application,create_server
    app=Application(root,args.weights)
    try:server=create_server(app,args.port)
    except Exception:
        app.close();raise
    url=f'http://127.0.0.1:{server.server_address[1]}/'
    print('Thermal web workspace:',url,flush=True)
    print('Local USB capture and inference. No screen recording. Ctrl+C stops the server.',flush=True)
    if not args.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:server.server_close();app.close()


if __name__=='__main__':main()
