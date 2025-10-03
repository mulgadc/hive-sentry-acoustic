#!/usr/bin/env python3
"""
Minimal webhook receiver for the decoupled plotting pipeline.
- Accepts POST /init and POST /frame
- Stores most recent payloads in memory and prints brief logs

This is a stub to unblock the emitter while you build a full plotting process.
Run:
  uvicorn webhook_receiver:app --host 127.0.0.1 --port 8000

Dependencies:
  pip install fastapi uvicorn
"""
from __future__ import annotations

from typing import Any, Dict
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
import json
import asyncio
import gzip

app = FastAPI(title="SRP-PHAT Webhook Receiver (stub)")

async def _read_json_maybe_gzip(req: Request) -> dict:
    """Read JSON body from FastAPI Request, supporting optional gzip Content-Encoding."""
    try:
        enc = (req.headers.get('content-encoding') or '').lower()
        if 'gzip' in enc:
            raw = await req.body()
            data = gzip.decompress(raw)
            return json.loads(data)
        else:
            return await req.json()
    except Exception:
        # fallback best-effort
        try:
            raw = await req.body()
            return json.loads(raw)
        except Exception:
            return {}

state: Dict[str, Any] = {
    "init": None,
    "frames_received": 0,
    "last_frame": None,
}

# Track connected browser WebSocket clients
ws_clients: set[WebSocket] = set()

async def _ws_broadcast(event_type: str, payload: dict):
    if not ws_clients:
        return
    dead: list[WebSocket] = []
    message = {"type": event_type, "payload": payload}
    for ws in list(ws_clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            ws_clients.discard(ws)
        except Exception:
            pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    # On connect, if we have init/last_frame, send them so client can render immediately
    try:
        if state.get("init") is not None:
            await websocket.send_json({"type": "init", "payload": state["init"]})
        if state.get("last_frame") is not None:
            await websocket.send_json({"type": "frame", "payload": state["last_frame"]})
    except Exception:
        pass
    try:
        while True:
            # We don't expect messages from the client; keep the connection alive.
            # If client sends pings or small messages, just receive and ignore.
            try:
                _ = await websocket.receive_text()
            except Exception:
                # Brief pause to prevent tight loop on certain browsers
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            ws_clients.discard(websocket)
        except Exception:
            pass

@app.websocket("/ws_ingest")
async def websocket_ingest(websocket: WebSocket):
    """WebSocket ingest endpoint for the compute-side emitter.
    Accepts JSON messages of the form {"type": "init"|"frame", "payload": {...}}.
    Updates in-memory state and broadcasts to browser WS clients.
    """
    await websocket.accept()
    try:
        while True:
            msg_text = await websocket.receive_text()
            try:
                msg = json.loads(msg_text)
            except Exception:
                continue
            et = msg.get("type")
            payload = msg.get("payload", {})
            if et == "init":
                state["init"] = payload
                state["frames_received"] = 0
                try:
                    await _ws_broadcast("init", payload)
                except Exception:
                    pass
            elif et == "frame":
                state["last_frame"] = payload
                state["frames_received"] = state.get("frames_received", 0) + 1
                try:
                    await _ws_broadcast("frame", payload)
                except Exception:
                    pass
            # else ignore
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

@app.post("/init")
async def post_init(req: Request):
    payload = await _read_json_maybe_gzip(req)
    state["init"] = payload
    state["frames_received"] = 0
    # Lightweight log once on init to avoid excessive console I/O
    try:
        srp = payload.get("srp", {})
        az = len(srp.get("az_grid_deg", []))
        el = len(srp.get("el_grid_deg", []))
        print(f"[receiver] /init | sr={payload.get('sample_rate_hz')} frame={payload.get('frame_size')} grid=az{az}xel{el}")
    except Exception:
        pass
    # Push to WS clients
    try:
        asyncio.create_task(_ws_broadcast("init", payload))
    except Exception:
        pass
    # Avoid dumping full payload each time to keep server responsive
    return JSONResponse({"ok": True})

@app.post("/frame")
async def post_frame(req: Request):
    payload = await _read_json_maybe_gzip(req)
    state["frames_received"] += 1
    state["last_frame"] = payload
    # Reduce per-frame console logging to lower CPU overhead
    try:
        fi = payload.get("frame_index")
        if (state["frames_received"] % 30) == 0:
            srp = payload.get("srp", {})
            best = srp.get("best", {})
            print(
                f"[receiver] /frame #{fi} | frames_received={state['frames_received']} | "
                f"best=({best.get('azimuth_deg')},{best.get('elevation_deg')}) p={best.get('power')}"
            )
    except Exception:
        pass
    # Push to WS clients
    try:
        asyncio.create_task(_ws_broadcast("frame", payload))
    except Exception:
        pass
    # Avoid dumping full payload for performance; use /debug if needed
    return JSONResponse({"ok": True})

@app.get("/health")
async def health():
    return {"status": "ok", "frames_received": state.get("frames_received", 0)}


@app.get("/init_json")
async def get_init_json():
    if state.get("init") is None:
        return JSONResponse({"error": "no init received yet"}, status_code=404)
    return JSONResponse(state["init"]) 


@app.get("/frame_json")
async def get_frame_json():
    if state.get("last_frame") is None:
        return JSONResponse({"error": "no frame received yet"}, status_code=404)
    return JSONResponse(state["last_frame"]) 


@app.get("/debug")
async def debug_page():
    init_payload = state.get("init")
    frame_payload = state.get("last_frame")
    init_str = json.dumps(init_payload, indent=2) if init_payload is not None else "<no init yet>"
    frame_str = json.dumps(frame_payload, indent=2) if frame_payload is not None else "<no frame yet>"
    html = f"""
    <html>
      <head>
        <title>SRP-PHAT Receiver Debug</title>
        <style>
          body {{ font-family: monospace; background: #0e1117; color: #e6edf3; }}
          h1, h2 {{ color: #58a6ff; }}
          pre {{ background: #161b22; padding: 1rem; border-radius: 8px; overflow-x: auto; }}
          a {{ color: #a5d6ff; }}
        </style>
      </head>
      <body>
        <h1>Debug JSON</h1>
        <p><a href="/">dashboard</a> | <a href="/health">health</a> | <a href="/init_json">init_json</a> | <a href="/frame_json">frame_json</a></p>
        <h2>Init Payload</h2>
        <pre>{init_str}</pre>
        <h2>Last Frame Payload</h2>
        <pre>{frame_str}</pre>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/")
async def index():
    # Serve static HTML with client-side polling to avoid server-side string interpolation issues
    html = """
    <html>
      <head>
        <title>SRP-PHAT Webhook Receiver</title>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
        <style>
          body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; background: #0e1117; color: #e6edf3; margin: 0; }
          header { padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; height: 60px; box-sizing: border-box; }
          h1 { margin: 0; font-size: 20px; color: #58a6ff; }
          .meta { font-size: 14px; color: #8b949e; margin-top: 4px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
          .meta span { display: inline-block; min-width: 50px; font-family: 'Courier New', monospace; text-align: left; }
          #utc_now { font-size: 18px; font-weight: 600; color: #e6edf3; min-width: 120px; }
          main { display: grid; grid-template-columns: 1fr; grid-template-rows: auto auto; gap: 12px; padding: 12px; }
          section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; overflow: hidden; }
          h2 { margin: 4px 8px 8px; font-size: 16px; color: #58a6ff; }
          pre { background: #0e1117; padding: 8px; border-radius: 6px; border: 1px solid #30363d; max-height: 300px; overflow: auto; }
          a { color: #a5d6ff; }
          /* Classification Banner */
          #classification_banner {
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-size: clamp(16px, 2.2vw, 24px);
            font-weight: 700;
            background: #161b22;
            border: 2px solid #30363d;
            border-radius: 8px;
            transition: background-color 0.3s ease;
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            padding: 0 10px;
            margin: 0;
          }
          #cls_label { margin-left: 12px; }
          #cls_prob { margin-left: 8px; font-size: 20px; opacity: 0.8; }
          /* Plots grid (2x2):
             Row1: SRP (left) | DOA Dial (right)
             Row2: Spectrogram (left) | Levels (right)
             Columns keep widths (2fr | 1fr). Each row items share height. */
          #plots { display: grid; grid-template-columns: 2fr 1fr; grid-template-rows: 60vh 40vh; gap: 12px; align-items: stretch; align-content: stretch; }
          /* Use content-box so borders don't cause the Plotly canvas to overflow */
          #plots > div { min-width: 0; min-height: 0; width: 100%; height: 100%; box-sizing: content-box; overflow: hidden; }
          #srp_plot { grid-column: 1 / 2; grid-row: 1 / 2; }
          #dial_plot { grid-column: 2 / 3; grid-row: 1 / 2; background: #161b22; border-radius: 8px; border: 1px solid #30363d; }
          #spec_plot { grid-column: 1 / 2; grid-row: 2 / 3; }
          #levels_plot { grid-column: 2 / 3; grid-row: 2 / 3; }
        </style>
      </head>
      <body>
        <header>
          <h1>SRP-PHAT Webhook Receiver</h1>
          <div class="meta">
            UTC: <span id="utc_now">--</span> |
            Frames: <span id="frames_count">0</span> |
            RX FPS: <span id="rx_fps">0.0</span> |
            Timing: <span id="timing_status" style="font-weight: 600;">-</span> |
            <a href="/health">health</a> | <a href="/debug" target="_blank">debug</a>
          </div>
        </header>
        <main>
          <section id="classification_banner">
            CLASSIFICATION: <span id="cls_label">-</span> <span id="cls_prob">-</span>
          </section>
          <section>
            <div id="plots">
              <div id="srp_plot"></div>
              <div id="dial_plot"></div>
              <div id="spec_plot"></div>
              <div id="levels_plot"></div>
            </div>
          </section>
        </main>

        <script>
          // Grid and metadata
          let azGrid = [];
          let elGrid = [];
          let sampleRateHz = null;
          let specNfft = null;
          let specFreqs = [];
          let specRows = 128;
          const specColsMax = 240;
          let specData = [];
          let lastFrameIndex = -1;
          let gotFrameUtc = false;
          let fpsWindow = [];
          let wsActive = false;

          // Dimension locking
          let dimensionsLocked = false;
          let dialDimensionsLocked = false;

          // Color scale locks
          let srpZMin = null;
          let srpZMax = null;
          let specZMin = null;
          let specZMax = null;

          // Frame pacer (15 FPS expected)
          const EXPECTED_FPS = 15.0;
          const FRAME_PERIOD_MS = 1000.0 / EXPECTED_FPS;
          let sessionStartTime = null;
          let firstFrameIndex = null;

          async function fetchJSON(url) {
            const r = await fetch(url);
            if (!r.ok) throw new Error('fetch failed');
            return r.json();
          }

          let srpLayout = null;
          let specLayout = null;
          let levelsLayout = null;
          let dialLayout = null;

          function lockDimensionsOnce() {
            if (dimensionsLocked) return;

            const srpDiv = document.getElementById('srp_plot');
            const specDiv = document.getElementById('spec_plot');
            const levelsDiv = document.getElementById('levels_plot');
            const dialDiv = document.getElementById('dial_plot');

            const srpRect = srpDiv.getBoundingClientRect();
            const specRect = specDiv.getBoundingClientRect();
            const levelsRect = levelsDiv.getBoundingClientRect();
            const dialRect = dialDiv.getBoundingClientRect();

            // Only lock if all containers have been laid out
            if (srpRect.width > 0 && srpRect.height > 0 &&
                specRect.width > 0 && specRect.height > 0 &&
                levelsRect.width > 0 && levelsRect.height > 0 &&
                dialRect.width > 0 && dialRect.height > 0) {

              Plotly.relayout('srp_plot', {
                width: Math.floor(srpRect.width),
                height: Math.floor(srpRect.height),
                autosize: false
              });

              Plotly.relayout('spec_plot', {
                width: Math.floor(specRect.width),
                height: Math.floor(specRect.height),
                autosize: false
              });

              Plotly.relayout('levels_plot', {
                width: Math.floor(levelsRect.width),
                height: Math.floor(levelsRect.height),
                autosize: false
              });

              // Size DOA dial to its inner content box (subtract borders/padding)
              const dialStyle = window.getComputedStyle(dialDiv);
              const bl = parseInt(dialStyle.borderLeftWidth) || 0;
              const br = parseInt(dialStyle.borderRightWidth) || 0;
              const pl = parseInt(dialStyle.paddingLeft) || 0;
              const pr = parseInt(dialStyle.paddingRight) || 0;
              const bt = parseInt(dialStyle.borderTopWidth) || 0;
              const bb = parseInt(dialStyle.borderBottomWidth) || 0;
              const pt = parseInt(dialStyle.paddingTop) || 0;
              const pb = parseInt(dialStyle.paddingBottom) || 0;
              const inset = 12; // shrink slightly to avoid any visual overlap
              const dialInnerW = Math.max(0, Math.floor(dialRect.width - bl - br - pl - pr - inset));
              const dialInnerH = Math.max(0, Math.floor(dialRect.height - bt - bb - pt - pb - inset));
              Plotly.relayout('dial_plot', {
                width: dialInnerW,
                height: dialInnerH,
                autosize: false
              });

              dimensionsLocked = true;
              console.log('Dimensions locked:', {
                srp: {w: srpRect.width, h: srpRect.height},
                spec: {w: specRect.width, h: specRect.height},
                levels: {w: levelsRect.width, h: levelsRect.height},
                dial: {w: dialRect.width, h: dialRect.height}
              });
            }
          }

          function checkFrameTiming(frameIndex) {
            if (sessionStartTime === null) {
              sessionStartTime = performance.now();
              firstFrameIndex = frameIndex;
            }

            const now = performance.now();
            const framesElapsed = frameIndex - firstFrameIndex;
            const expectedTime = sessionStartTime + (framesElapsed * FRAME_PERIOD_MS);
            const drift = now - expectedTime;

            let status, color;
            if (Math.abs(drift) < 50) {
              status = "ON TIME";
              color = "#3fb950";
            } else if (drift > 0) {
              status = `BEHIND +${Math.round(drift)}ms`;
              color = "#f85149";
            } else {
              status = `AHEAD ${Math.round(drift)}ms`;
              color = "#58a6ff";
            }

            const statusEl = document.getElementById('timing_status');
            statusEl.textContent = status;
            statusEl.style.color = color;

            return { drift, status, color };
          }

          function initPlots() {
            // SRP heatmap with placeholder + overlay trace for best DOA marker
            const srpHeat = {z:[[0]], type:'heatmap', colorscale:'Viridis', showscale:true, zauto:false, zmin:0, zmax:1};
            const doaMarker = {x: [null], y: [null], mode: 'markers', type: 'scatter',
                               marker: {color:'#ffd166', size: 14, symbol:'x', line:{width:2, color:'#000'}}, name:'Best DOA'};
            srpLayout = {
              title: {text: 'SRP-PHAT Power Map', font: {size: 14}},
              paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
              margin: {l:50, r:60, t:35, b:40},
              xaxis: {
                title: 'Azimuth (deg)',
                gridcolor:'#30363d',
                zerolinecolor:'#30363d',
                fixedrange: true,
                autorange: true,
                type: 'linear',
                tickmode: 'linear',
                tick0: 0,
                dtick: 45,
                range: [0, 360]
              },
              yaxis: {
                title: 'Elevation (deg)',
                gridcolor:'#30363d',
                zerolinecolor:'#30363d',
                fixedrange: true,
                autorange: true,
                type: 'linear',
                tickmode: 'linear',
                tick0: 0,
                dtick: 15,
                range: [0, 90]
              },
              legend: {orientation:'h'},
              autosize: true
            };
            Plotly.newPlot('srp_plot', [srpHeat, doaMarker], srpLayout, {displayModeBar:false, responsive:true});

            // Spectrogram: use Viridis and locked y-axis
            specLayout = {
              title: {text: 'Spectrogram (mic0)', font: {size: 14}},
              paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
              margin: {l:50, r:10, t:35, b:40},
              xaxis: {
                title: 'Frame',
                gridcolor:'#30363d',
                fixedrange: true
              },
              yaxis: {
                title: 'Frequency (Hz)',
                gridcolor:'#30363d',
                fixedrange: true,
                autorange: false,
                range: [0, 24000],
                tickmode: 'array',
                tickvals: [0, 5000, 10000, 15000, 20000],
                ticktext: ['0', '5k', '10k', '15k', '20k']
              },
              autosize: true
            };
            Plotly.newPlot('spec_plot', [{z:[[0]], type:'heatmap', colorscale:'Viridis', zauto:false, zmin:-60, zmax:0}], specLayout, {displayModeBar:false, responsive:true});

            // DOA Dial (polar radar-style)
            const dialNeedle = {
              type: 'scatterpolar',
              r: [0, 1],
              theta: [0, 0],
              mode: 'lines+markers',
              line: {color: '#ffd166', width: 3},
              marker: {color: '#ffd166', size: 6},
              name: 'DOA'
            };
            dialLayout = {
              title: {text: 'DOA Dial (Azimuth)', font: {size: 14, color: '#ffffff'}},
              paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
              font: {color: '#ffffff'},
              margin: {l:16, r:16, t:28, b:16},
              polar: {
                bgcolor: '#161b22',
                radialaxis: {
                  range: [0, 1],
                  showticklabels: false,
                  ticks: '',
                  gridcolor:'#30363d',
                  linecolor: '#30363d'
                },
                angularaxis: {
                  rotation: 0,
                  direction: 'clockwise',
                  tickmode: 'array',
                  tickvals: [0,45,90,135,180,225,270,315],
                  ticktext: ['0','45','90','135','180','225','270','315'],
                  gridcolor:'#30363d',
                  tickfont: {color: '#ffffff'},
                  linecolor: '#30363d'
                },
                domain: {x: [0, 1], y: [0, 1]}
              },
              showlegend: false,
              autosize: true
            };
            Plotly.newPlot('dial_plot', [dialNeedle], dialLayout, {displayModeBar:false, responsive:true});

            // Compact Levels bar meter (first 5 channels)
            levelsLayout = {
              title: {text: 'Levels (RMS)', font: {size: 14}},
              paper_bgcolor:'#161b22', plot_bgcolor:'#161b22',
              margin: {l:40, r:10, t:30, b:30},
              yaxis: {
                gridcolor:'#30363d',
                rangemode:'tozero',
                fixedrange: true,
                range: [0, 0.5]
              },
              xaxis: {
                gridcolor:'#30363d',
                fixedrange: true
              },
              autosize: true
            };
            Plotly.newPlot('levels_plot', [{
              x: ['ch0','ch1','ch2','ch3','ch4'],
              y: [0,0,0,0,0],
              type: 'bar',
              marker: {color: ['#58a6ff','#8b949e','#d29922','#3fb950','#f85149']}
            }], levelsLayout, {displayModeBar:false, responsive:true});
          }

          function updateSrp(frame) {
            const srp = frame.srp || {};
            if (!srp.power_map) {
              console.warn('Frame missing srp.power_map:', frame.frame_index, 'srp keys:', Object.keys(srp));
              return;
            }

            const z = srp.power_map;
            const x = (Array.isArray(azGrid) && azGrid.length > 1) ? azGrid : (srp.az_grid_deg || []);
            const y = (Array.isArray(elGrid) && elGrid.length > 1) ? elGrid : (srp.el_grid_deg || []);

            console.log('updateSrp called:', {
              frameIndex: frame.frame_index,
              zDims: z.length + 'x' + (z[0]?.length || 0),
              xPoints: x.length,
              yPoints: y.length,
              xRange: [x[0], x[x.length-1]],
              yRange: [y[0], y[y.length-1]]
            });
            const best = srp.best || {};

            // Lock color scale on first frame
            if (srpZMin === null || srpZMax === null) {
              let zmin = Infinity, zmax = -Infinity;
              for (let i = 0; i < z.length; i++) {
                const row = z[i] || [];
                for (let j = 0; j < row.length; j++) {
                  const v = row[j];
                  if (v < zmin) zmin = v;
                  if (v > zmax) zmax = v;
                }
              }
              srpZMin = Number.isFinite(zmin) ? zmin : 0;
              srpZMax = Number.isFinite(zmax) ? zmax : 1;
              if (srpZMax === srpZMin) srpZMax = srpZMin + 1e-6;
              console.log('SRP colorscale locked:', {zmin: srpZMin, zmax: srpZMax, grid: z.length + 'x' + (z[0]?.length || 0)});
            }

            // Lock axis ranges on first frame
            if (srpLayout && srpLayout.xaxis.autorange !== false && x.length > 1) {
              const xmin = Math.min(x[0], x[x.length - 1]);
              const xmax = Math.max(x[0], x[x.length - 1]);
              srpLayout.xaxis.range = [xmin, xmax];
              srpLayout.xaxis.autorange = false;
              console.log('SRP x-axis locked:', {range: [xmin, xmax], points: x.length});
            }
            if (srpLayout && srpLayout.yaxis.autorange !== false && y.length > 1) {
              const ymin = Math.min(y[0], y[y.length - 1]);
              const ymax = Math.max(y[0], y[y.length - 1]);
              srpLayout.yaxis.range = [ymin, ymax];
              srpLayout.yaxis.autorange = false;
              console.log('SRP y-axis locked:', {range: [ymin, ymax], points: y.length});
            }

            // Check if grids are uniform (1° resolution typically)
            const isUniform = (arr) => {
              if (!Array.isArray(arr) || arr.length < 2) return false;
              const dx = arr[1] - arr[0];
              for (let i = 2; i < arr.length; i++) {
                if (Math.abs((arr[i] - arr[i-1]) - dx) > 1e-6) return false;
              }
              return true;
            };

            // Build heatmap trace - use x0/dx for uniform grids
            let heat;
            if (isUniform(x) && isUniform(y)) {
              const dx = x[1] - x[0];
              const dy = y[1] - y[0];
              heat = {
                z: z,
                x0: x[0],
                dx: dx,
                y0: y[0],
                dy: dy,
                type: 'heatmap',
                colorscale: 'Viridis',
                zmin: srpZMin,
                zmax: srpZMax,
                zauto: false,
                colorbar: {thickness: 15, len: 0.85, tickformat: '.2f', x: 1.0}
              };
            } else {
              heat = {
                z: z,
                x: x,
                y: y,
                type: 'heatmap',
                colorscale: 'Viridis',
                zmin: srpZMin,
                zmax: srpZMax,
                zauto: false,
                colorbar: {thickness: 15, len: 0.85, tickformat: '.2f', x: 1.0}
              };
            }

            const marker = {
              x: [best.azimuth_deg],
              y: [best.elevation_deg],
              mode: 'markers',
              type: 'scatter',
              marker: {color: '#ffd166', size: 14, symbol: 'x', line: {width: 2, color: '#000'}},
              name: 'Best DOA'
            };

            Plotly.react('srp_plot', [heat, marker], srpLayout, {displayModeBar: false, responsive: !dimensionsLocked});

            // Lock dimensions after first render
            if (!dimensionsLocked) {
              lockDimensionsOnce();
            }
          }

          function updateDial(frame) {
            try {
              const best = (frame && frame.srp && frame.srp.best) ? frame.srp.best : null;
              if (!best || typeof best.azimuth_deg !== 'number') return;
              const az = best.azimuth_deg % 360;
              const data = [{
                type: 'scatterpolar',
                r: [0, 1],
                theta: [az, az],
                mode: 'lines+markers',
                line: {color: '#ffd166', width: 3},
                marker: {color: '#ffd166', size: 6},
                name: 'DOA'
              }];
              Plotly.react('dial_plot', data, dialLayout, {displayModeBar:false, responsive: !dimensionsLocked});
              if (!dimensionsLocked) lockDimensionsOnce();
            } catch(e) { /* ignore */ }
          }

          function updateSpec(frame) {
            let col = frame.spec_power_mic0 || [];
            if ((!col || col.length === 0) && frame.spectrogram && Array.isArray(frame.spectrogram.ch0_power_slice)) {
              col = frame.spectrogram.ch0_power_slice;
            }
            if (col.length === 0) return;

            if (specData.length === 0) {
              specRows = col.length;
              specData = Array.from({length: specRows}, () => []);
            }

            // Append column to right
            for (let r = 0; r < specRows; r++) {
              const vlin = col[r] || 0;
              const v = 10 * Math.log10(vlin + 1e-12);
              const row = specData[r];
              row.push(v);
              if (row.length > specColsMax) {
                const drop = row.length - specColsMax;
                row.splice(0, drop);
              }
            }

            const z = specData.map(row => row.slice());

            // Dynamic 60 dB range
            let zmax = -1e9;
            for (let r = 0; r < z.length; r++) {
              for (let c = 0; c < z[r].length; c++) {
                zmax = Math.max(zmax, z[r][c]);
              }
            }
            const zmin = zmax - 60;

            // Build trace
            const trace = {
              z: z,
              type: 'heatmap',
              colorscale: 'Viridis',
              zmin: zmin,
              zmax: zmax,
              zauto: false,
              colorbar: {thickness: 10, len: 0.85}
            };

            // Y-axis in Hz
            if (sampleRateHz && specRows > 1) {
              if (specFreqs.length !== specRows) {
                specFreqs = Array.from({length: specRows}, (_, k) => (k * (sampleRateHz / 2) / (specRows - 1)));
              }
              trace.y = specFreqs;

              // Update layout with locked y-axis range
              if (specLayout) {
                const nyq = sampleRateHz / 2;
                specLayout.yaxis.range = [0, nyq];
                specLayout.yaxis.autorange = false;
              }
            }

            Plotly.react('spec_plot', [trace], specLayout, {displayModeBar: false, responsive: !dimensionsLocked});
          }

          function updateLevels(frame) {
            // Accept legacy or nested
            let lv = frame.levels_first5 || [];
            if ((!lv || lv.length === 0) && frame.levels && Array.isArray(frame.levels.rms_first5)) {
              lv = frame.levels.rms_first5;
            }
            if (!lv || lv.length === 0) return;
            const dataUpdate = { y: [lv.slice(0,5)] };
            Plotly.update('levels_plot', dataUpdate, {});
          }

          function updateClassification(frame) {
            try {
              const cls = frame.classification || null;
              if (!cls) {
                return;
              }

              const label = String(cls.label ?? '-');
              const prob = (typeof cls.prob === 'number') ? cls.prob : parseFloat(cls.prob || 'NaN');

              const lblEl = document.getElementById('cls_label');
              const probEl = document.getElementById('cls_prob');
              const bannerEl = document.getElementById('classification_banner');

              lblEl.textContent = label;
              probEl.textContent = isNaN(prob) ? '-' : `(${(prob * 100).toFixed(1)}%)`;

              // Color coding
              if (label.toLowerCase().startsWith('drone')) {
                lblEl.style.color = '#3fb950';
                bannerEl.style.borderColor = '#3fb950';
                bannerEl.style.backgroundColor = '#0d3a1f';
              } else if (label.toLowerCase().includes('non-drone') || label.toLowerCase() === 'background') {
                lblEl.style.color = '#8b949e';
                bannerEl.style.borderColor = '#30363d';
                bannerEl.style.backgroundColor = '#161b22';
              } else {
                lblEl.style.color = '#e6edf3';
                bannerEl.style.borderColor = '#30363d';
                bannerEl.style.backgroundColor = '#161b22';
              }
            } catch (e) { /* ignore */ }
          }

          function formatUtcMillis(s) {
            if (!s) return '';
            try {
              const d = new Date(s);
              if (isNaN(d.getTime())) return String(s);
              const pad = (n, w) => String(n).padStart(w, '0');
              return `${pad(d.getUTCHours(),2)}:${pad(d.getUTCMinutes(),2)}:${pad(d.getUTCSeconds(),2)}.${pad(d.getUTCMilliseconds(),3)}Z`;
            } catch(e) {
              return String(s);
            }
          }

          async function refresh() {
            try {
              const init = await fetchJSON('/init_json');
              azGrid = init?.srp?.az_grid_deg || azGrid;
              elGrid = init?.srp?.el_grid_deg || elGrid;
              if (init?.sample_rate_hz) sampleRateHz = init.sample_rate_hz;
              if (init?.spectrogram?.spec_nfft) specNfft = init.spectrogram.spec_nfft;
              else if (init?.spectrogram?.nfft) specNfft = init.spectrogram.nfft;
              const initUtc = init?.progress?.utc_iso;
              if (initUtc && !gotFrameUtc) {
                document.getElementById('utc_now').textContent = formatUtcMillis(initUtc);
              }
            } catch(e) { /* init not ready */ }

            try {
              const frame = await fetchJSON('/frame_json');
              const fi = frame.frame_index ?? -1;

              if (frame?.utc_iso) {
                document.getElementById('utc_now').textContent = formatUtcMillis(frame.utc_iso);
                gotFrameUtc = true;
              }

              if (!sampleRateHz && frame?.sample_rate_hz) sampleRateHz = frame.sample_rate_hz;

              if (fi !== lastFrameIndex) {
                lastFrameIndex = fi;
                document.getElementById('frames_count').textContent = String(fi + 1);

                // Check frame timing
                checkFrameTiming(fi);

                // Update plots
                updateSrp(frame);
                updateSpec(frame);
                updateLevels(frame);
                updateDial(frame);
                updateClassification(frame);

                // RX FPS over sliding 1s window
                const now = performance.now();
                fpsWindow.push(now);
                while (fpsWindow.length && (now - fpsWindow[0]) > 1000) fpsWindow.shift();
                const fps = fpsWindow.length;
                document.getElementById('rx_fps').textContent = fps.toFixed(1);
              }
            } catch(e) { /* no frame yet */ }
          }

          initPlots();
          // Try WebSocket first; fallback to polling if it fails
          function setupWebSocket() {
            try {
              const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
              const wsUrl = `${proto}://${location.host}/ws`;
              const ws = new WebSocket(wsUrl);
              ws.onopen = () => { wsActive = true; };
              ws.onclose = () => { wsActive = false; };
              ws.onerror = () => { wsActive = false; };
              ws.onmessage = (ev) => {
                try {
                  const msg = JSON.parse(ev.data);
                  const t = msg?.type;
                  const payload = msg?.payload || {};
                  if (t === 'init') {
                    // emulate init path
                    if (payload?.srp?.az_grid_deg) azGrid = payload.srp.az_grid_deg;
                    if (payload?.srp?.el_grid_deg) elGrid = payload.srp.el_grid_deg;
                    if (payload?.sample_rate_hz) sampleRateHz = payload.sample_rate_hz;
                    const spec = payload?.spectrogram || {};
                    if (spec?.spec_nfft) specNfft = spec.spec_nfft; else if (spec?.nfft) specNfft = spec.nfft;
                    const initUtc = payload?.progress?.utc_iso;
                    if (initUtc && !gotFrameUtc) {
                      document.getElementById('utc_now').textContent = formatUtcMillis(initUtc);
                    }
                  } else if (t === 'frame') {
                    const frame = payload;
                    const fi = frame.frame_index ?? -1;
                    if (frame?.utc_iso) {
                      document.getElementById('utc_now').textContent = formatUtcMillis(frame.utc_iso);
                      gotFrameUtc = true;
                    }
                    if (!sampleRateHz && frame?.sample_rate_hz) sampleRateHz = frame.sample_rate_hz;
                    if (fi !== lastFrameIndex) {
                      lastFrameIndex = fi;
                      document.getElementById('frames_count').textContent = String(fi + 1);

                      // Check frame timing
                      checkFrameTiming(fi);

                      updateSrp(frame);
                      updateSpec(frame);
                      updateLevels(frame);
                      updateDial(frame);
                      updateClassification(frame);
                      const now = performance.now();
                      fpsWindow.push(now);
                      while (fpsWindow.length && (now - fpsWindow[0]) > 1000) fpsWindow.shift();
                      document.getElementById('rx_fps').textContent = fpsWindow.length.toFixed(1);
                    }
                  }
                } catch(e) { /* ignore bad message */ }
              };
            } catch(e) {
              wsActive = false;
            }
          }

          setupWebSocket();
          setInterval(refresh, 50);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
