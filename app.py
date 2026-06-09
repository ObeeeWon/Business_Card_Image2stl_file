"""
Fung-AI Business Card STL Generator — v9 algorithm + adjustable parameters

Parameters:
  total_t      : total card thickness (mm)
  recess_front : front face engraving depth (mm) — independent of thickness
  recess_back  : back face engraving depth (mm)  — independent of thickness
  corner_r     : corner radius (mm)
  chamfer      : edge chamfer width (mm)

Geometry guarantee:
  core_thickness = total_t - recess_front - recess_back >= MIN_WALL (0.3mm)
  Validated on server before generation; client shows live warning.
"""

import os, io, struct, time, uuid, threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_DIR = Path("uploads");  UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs");  OUTPUT_DIR.mkdir(exist_ok=True)

jobs = {};  jobs_lock = threading.Lock()

PX_MM    = 6      # pixels per mm
SUPER    = 8      # supersampling for smooth boundary
MIN_WALL = 0.3    # mm — minimum solid core between engravings


# ── Parameter validation ──────────────────────────────────────────────────────

def validate_params(p):
    """Returns (ok, errors, warnings, core_thickness)"""
    t  = p['total_t']
    rf = p['recess_front']
    rb = p['recess_back']
    ch = p['chamfer']
    cr = p['corner_r']
    errors = []; warnings = []

    core = t - rf - rb
    if core < MIN_WALL:
        errors.append(
            f"Engravings overlap! Core = {core:.2f} mm (need ≥ {MIN_WALL} mm). "
            f"Increase thickness or reduce engraving depths."
        )
    if rf <= 0: errors.append("Front engraving depth must be > 0")
    if rb <= 0: errors.append("Back engraving depth must be > 0")
    if t < 0.8: errors.append("Thickness must be ≥ 0.8 mm")
    if t > 10:  errors.append("Thickness must be ≤ 10 mm")
    if rf > 3:  errors.append("Front engraving depth must be ≤ 3 mm")
    if rb > 3:  errors.append("Back engraving depth must be ≤ 3 mm")
    if ch < 0.2: errors.append("Chamfer must be ≥ 0.2 mm")
    if ch >= cr: errors.append(f"Chamfer ({ch}mm) must be less than corner radius ({cr}mm)")
    if cr < 0.5: errors.append("Corner radius must be ≥ 0.5 mm")
    if cr > 8:   errors.append("Corner radius must be ≤ 8 mm")

    if not errors:
        if core < 0.5:
            warnings.append(f"Core thickness {core:.2f} mm is very thin — may be fragile")
        if t < 1.5:
            warnings.append("Card under 1.5 mm may be brittle")

    return len(errors) == 0, errors, warnings, max(core, 0)


# ── Card mask & distance field ────────────────────────────────────────────────

def build_card_mask(corner_r, chamfer):
    GW = int(85.6 * PX_MM);  GH = int(54.0 * PX_MM)
    CR = int(corner_r * PX_MM)
    GW_S = GW*SUPER;  GH_S = GH*SUPER;  CR_S = CR*SUPER
    rr_s = np.zeros((GH_S, GW_S), bool)
    rr_s[CR_S:GH_S-CR_S, :] = True;  rr_s[:, CR_S:GW_S-CR_S] = True
    cy, cx = np.ogrid[:GH_S, :GW_S]
    for ry, rx in [(CR_S,CR_S),(CR_S,GW_S-CR_S-1),(GH_S-CR_S-1,CR_S),(GH_S-CR_S-1,GW_S-CR_S-1)]:
        rr_s |= ((cy-ry)**2 + (cx-rx)**2) <= CR_S**2
    rr = rr_s.reshape(GH,SUPER,GW,SUPER).mean((1,3)) >= 0.5

    CH = int(chamfer * PX_MM);  PAD = CH + 5
    padded = np.zeros((GH+2*PAD, GW+2*PAD), bool)
    padded[PAD:PAD+GH, PAD:PAD+GW] = rr
    dist = distance_transform_edt(padded)[PAD:PAD+GH, PAD:PAD+GW]
    chamfer_f = np.clip((dist-1)/(max(CH-1,1)), 0.0, 1.0).astype(np.float32)
    interior = rr & ~(dist < CH)
    return rr, chamfer_f, interior, GW, GH


# ── Depth loader ──────────────────────────────────────────────────────────────

def load_depth(img_pil, GW, GH, mirror_x=False):
    img = img_pil.convert("L")
    arr = np.array(img);  m = arr < 240
    rows = np.where(m.any(axis=1))[0];  cols = np.where(m.any(axis=0))[0]
    if len(rows) and len(cols):
        img = img.crop((cols[0], rows[0], cols[-1]+1, rows[-1]+1))
    img = img.resize((GW, GH), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Sharpness(img).enhance(2.5)
    arr = np.array(img, dtype=np.float32)
    depth = np.clip((210.0-arr)/180.0, 0.0, 1.0)
    # Gap enforcement for 0.4mm nozzle
    strokes = depth > 0.15
    d_in = distance_transform_edt(strokes).astype(np.float32)
    NOZZLE_PX = 0.4 * PX_MM
    depth = depth * np.clip(d_in / (NOZZLE_PX * 0.45), 0.0, 1.0)
    depth = gaussian_filter(depth, sigma=0.4)
    depth[depth < 0.06] = 0.0
    if mirror_x: depth = np.fliplr(depth)
    return depth.astype(np.float32)


# ── Height maps ───────────────────────────────────────────────────────────────

def build_height_maps(rr, chamfer_f, interior, front_depth, back_depth,
                      total_t, recess_front, recess_back):
    """
    Thickness is the DISTANCE between the two engraved faces.
    Adjusting total_t changes the solid core without affecting engraving depths.

      z_top (normal surface) = total_t
      z_top (engraved pixel) = total_t - recess_front * pixel_depth
      z_bot (normal surface) = 0
      z_bot (engraved pixel) = recess_back * pixel_depth  (bump UP)
    """
    GH, GW = rr.shape
    z_top = np.where(rr, chamfer_f * total_t, 0.0).astype(np.float32)
    z_top[interior] -= front_depth[interior] * recess_front
    z_top = np.clip(z_top, 0.0, total_t)

    z_bot = np.zeros((GH, GW), np.float32)
    z_bot[interior] = back_depth[interior] * recess_back

    # Enforce minimum wall
    thin = rr & ((z_top - z_bot) < MIN_WALL)
    z_top[thin] = np.minimum(total_t, z_bot[thin] + MIN_WALL)
    z_top = np.clip(z_top, 0.0, total_t)
    return z_top, z_bot


# ── STL surface packer ────────────────────────────────────────────────────────

def pack_surface(cx, cxp, ry0, ry1, z00, z01, z10, z11, flip):
    N = len(cx)
    out = bytearray(N * 2 * 50)
    off = 0;  pi = struct.pack_into
    for i in range(N):
        x0=float(cx[i]); x1=float(cxp[i]); y0=float(ry0[i]); y1=float(ry1[i])
        v00=(x0,y0,float(z00[i])); v01=(x1,y0,float(z01[i]))
        v10=(x0,y1,float(z10[i])); v11=(x1,y1,float(z11[i]))
        if not flip:
            ax,ay,az=x1-x0,0.,float(z01[i]-z00[i]); bx,by,bz=0.,y1-y0,float(z10[i]-z00[i])
            nx=ay*bz-az*by; ny=az*bx-ax*bz; nz=ax*by-ay*bx
            ln=(nx*nx+ny*ny+nz*nz)**.5
            if ln>1e-12: nx/=ln;ny/=ln;nz/=ln
            else: nx,ny,nz=0,0,1
            pi("<3f3f3f3fH",out,off,nx,ny,nz,*v00,*v01,*v10,0); off+=50
            ax,ay,az=0.,y1-y0,float(z11[i]-z01[i]); bx,by,bz=x0-x1,0.,float(z10[i]-z01[i])
            nx=ay*bz-az*by; ny=az*bx-ax*bz; nz=ax*by-ay*bx
            ln=(nx*nx+ny*ny+nz*nz)**.5
            if ln>1e-12: nx/=ln;ny/=ln;nz/=ln
            else: nx,ny,nz=0,0,1
            pi("<3f3f3f3fH",out,off,nx,ny,nz,*v01,*v11,*v10,0); off+=50
        else:
            ax,ay,az=0.,y1-y0,float(z10[i]-z00[i]); bx,by,bz=x1-x0,0.,float(z01[i]-z00[i])
            nx=ay*bz-az*by; ny=az*bx-ax*bz; nz=ax*by-ay*bx
            ln=(nx*nx+ny*ny+nz*nz)**.5
            if ln>1e-12: nx/=ln;ny/=ln;nz/=ln
            else: nx,ny,nz=0,0,-1
            pi("<3f3f3f3fH",out,off,nx,ny,nz,*v00,*v10,*v01,0); off+=50
            ax,ay,az=x0-x1,0.,float(z10[i]-z01[i]); bx,by,bz=0.,y1-y0,float(z11[i]-z01[i])
            nx=ay*bz-az*by; ny=az*bx-ax*bz; nz=ax*by-ay*bx
            ln=(nx*nx+ny*ny+nz*nz)**.5
            if ln>1e-12: nx/=ln;ny/=ln;nz/=ln
            else: nx,ny,nz=0,0,-1
            pi("<3f3f3f3fH",out,off,nx,ny,nz,*v01,*v10,*v11,0); off+=50
    return bytes(out)


def build_walls(rr, z_top, z_bot, GW, GH, mmpp):
    wall = bytearray()
    def wt(nx,ny,nz,a,b,c):
        wall.extend(struct.pack("<3f3f3f3fH",
            float(nx),float(ny),float(nz),
            float(a[0]),float(a[1]),float(a[2]),
            float(b[0]),float(b[1]),float(b[2]),
            float(c[0]),float(c[1]),float(c[2]),0))
    for row in range(GH):
        for col in range(GW):
            if not rr[row,col]: continue
            zt=float(z_top[row,col]); zb=float(z_bot[row,col])
            x0,x1=col*mmpp,(col+1)*mmpp; y0_=(GH-row)*mmpp; y1_=(GH-row-1)*mmpp
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=row+dr,col+dc
                if 0<=nr<GH and 0<=nc<GW and rr[nr,nc]: continue
                if dc==1:
                    wt(1,0,0,(x1,y1_,zb),(x1,y0_,zb),(x1,y1_,zt))
                    wt(1,0,0,(x1,y0_,zb),(x1,y0_,zt),(x1,y1_,zt))
                elif dc==-1:
                    wt(-1,0,0,(x0,y0_,zb),(x0,y1_,zb),(x0,y0_,zt))
                    wt(-1,0,0,(x0,y1_,zb),(x0,y1_,zt),(x0,y0_,zt))
                elif dr==-1:
                    wt(0,1,0,(x0,y0_,zb),(x1,y0_,zb),(x0,y0_,zt))
                    wt(0,1,0,(x1,y0_,zb),(x1,y0_,zt),(x0,y0_,zt))
                else:
                    wt(0,-1,0,(x1,y1_,zb),(x0,y1_,zb),(x1,y1_,zt))
                    wt(0,-1,0,(x0,y1_,zb),(x0,y1_,zt),(x1,y1_,zt))
    return wall


# ── Full pipeline ─────────────────────────────────────────────────────────────

def generate_stl(front_pil, back_pil, params, progress_cb=None):
    def prog(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    total_t = params['total_t'];  rf = params['recess_front'];  rb = params['recess_back']
    chamfer = params['chamfer'];  corner_r = params['corner_r']
    mmpp = float(1.0 / PX_MM)

    prog("Building card geometry…", 5)
    rr, chamfer_f, interior, GW, GH = build_card_mask(corner_r, chamfer)

    prog("Processing front image…", 20)
    front_depth = load_depth(front_pil, GW, GH, mirror_x=False)
    prog("Processing back image…", 35)
    back_depth  = load_depth(back_pil,  GW, GH, mirror_x=True)

    prog("Computing height maps…", 50)
    z_top, z_bot = build_height_maps(rr, chamfer_f, interior,
                                     front_depth, back_depth, total_t, rf, rb)

    prog("Building top surface…", 60)
    valid = rr[:-1,:-1] & rr[1:,:-1] & rr[:-1,1:] & rr[1:,1:]
    rv, cv = np.where(valid);  N = len(rv)
    cx=cv.astype(np.float32)*mmpp; cxp=cx+mmpp
    ry0=(GH-rv).astype(np.float32)*mmpp; ry1=ry0-mmpp
    zt00=z_top[rv,cv]; zt01=z_top[rv,cv+1]; zt10=z_top[rv+1,cv]; zt11=z_top[rv+1,cv+1]
    zb00=z_bot[rv,cv]; zb01=z_bot[rv,cv+1]; zb10=z_bot[rv+1,cv]; zb11=z_bot[rv+1,cv+1]

    top_b = pack_surface(cx,cxp,ry0,ry1,zt00,zt01,zt10,zt11,False)
    prog("Building bottom surface…", 75)
    bot_b = pack_surface(cx,cxp,ry0,ry1,zb00,zb01,zb10,zb11,True)
    prog("Building perimeter walls…", 88)
    wall_b = build_walls(rr, z_top, z_bot, GW, GH, mmpp)

    prog("Writing STL…", 96)
    total_tris = N*4 + len(wall_b)//50
    hdr = b"Fung-AI Business Card STL" + b"\x00"*80
    stl = hdr[:80] + struct.pack("<I", total_tris) + top_b + bot_b + wall_b
    prog("Done!", 100)
    return stl


# ── Background job ────────────────────────────────────────────────────────────

def run_job(job_id, front_path, back_path, params):
    def progress(msg, pct):
        with jobs_lock:
            jobs[job_id].update({"progress": pct, "status_msg": msg})
    try:
        stl = generate_stl(Image.open(front_path), Image.open(back_path),
                           params, progress_cb=progress)
        out = OUTPUT_DIR / f"{job_id}.stl"
        out.write_bytes(stl)
        with jobs_lock:
            jobs[job_id].update({"status":"done","progress":100,
                                 "file":str(out),"size_mb":len(stl)/1024/1024,
                                 "params": params})
    except Exception as e:
        with jobs_lock:
            jobs[job_id].update({"status":"error","error":str(e)})
    finally:
        for p in [front_path, back_path]:
            try: os.unlink(p)
            except: pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return render_template_string(HTML_PAGE)

@app.route("/api/validate", methods=["POST"])
def api_validate():
    p = request.json
    ok, errors, warnings, core = validate_params(p)
    return jsonify({"ok": ok, "errors": errors, "warnings": warnings, "core": round(core,3)})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    # Parse params
    try:
        params = {
            "total_t":      float(request.form.get("total_t",      2.0)),
            "recess_front": float(request.form.get("recess_front", 0.5)),
            "recess_back":  float(request.form.get("recess_back",  0.5)),
            "chamfer":      float(request.form.get("chamfer",      1.5)),
            "corner_r":     float(request.form.get("corner_r",     3.0)),
        }
    except Exception as e:
        return jsonify({"error": f"Invalid parameter: {e}"}), 400

    ok, errors, warnings, core = validate_params(params)
    if not ok:
        return jsonify({"error": errors[0]}), 400

    if "front" not in request.files or "back" not in request.files:
        return jsonify({"error": "Upload both front and back images"}), 400

    job_id = str(uuid.uuid4())[:8]
    front_path = str(UPLOAD_DIR / f"{job_id}_front.png")
    back_path  = str(UPLOAD_DIR / f"{job_id}_back.png")

    try:
        Image.open(request.files["front"]).convert("RGB").save(front_path)
        Image.open(request.files["back"]).convert("RGB").save(back_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    with jobs_lock:
        jobs[job_id] = {"status":"running","progress":0,"status_msg":"Starting…"}

    t = threading.Thread(target=run_job, args=(job_id, front_path, back_path, params))
    t.daemon = True; t.start()
    return jsonify({"job_id": job_id})

@app.route("/api/status/<job_id>")
def api_status(job_id):
    with jobs_lock: job = jobs.get(job_id)
    if not job: return jsonify({"error":"Not found"}), 404
    return jsonify(job)

@app.route("/api/download/<job_id>")
def api_download(job_id):
    with jobs_lock: job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error":"Not ready"}), 404
    return send_file(job["file"], as_attachment=True,
                     download_name="business_card.stl", mimetype="model/stl")


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Card→STL · Fung-AI Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{
  --bg:#0d0d0f;--surface:#141417;--surface2:#1a1a20;
  --border:#2a2a30;--border-hover:#3a3a45;
  --accent:#e8ff5a;--accent2:#5affcd;--accent3:#ff9f5a;
  --text:#e8e8ec;--muted:#6b6b78;--danger:#ff5a5a;--warn:#ffaa5a;
  --r:12px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:var(--bg);color:var(--text);
  font-family:'DM Sans',sans-serif;font-weight:300;
  min-height:100vh;display:flex;flex-direction:column;align-items:center;
  padding:0 1rem 5rem;
}

/* Header */
header{width:100%;max-width:860px;padding:2.5rem 0 1.5rem;display:flex;flex-direction:column;gap:.4rem}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(1.8rem,4vw,2.8rem);
  line-height:1;letter-spacing:-.03em;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.logo-sub{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted);letter-spacing:.15em;text-transform:uppercase}
.badge{display:inline-flex;align-items:center;gap:.4rem;background:#1a1a20;
  border:1px solid var(--border);border-radius:999px;padding:.2rem .7rem;
  font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted);margin-top:.3rem;width:fit-content}
.badge em{color:var(--accent);font-style:normal}

/* Layout */
.main{width:100%;max-width:860px;display:flex;flex-direction:column;gap:1rem}

/* Panel */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:1.5rem;display:flex;flex-direction:column;gap:1.2rem}
.panel-title{font-family:'DM Mono',monospace;font-size:.62rem;color:var(--muted);
  letter-spacing:.12em;text-transform:uppercase;border-bottom:1px solid var(--border);
  padding-bottom:.5rem;display:flex;justify-content:space-between;align-items:center}

/* Upload grid */
.upload-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
@media(max-width:520px){.upload-grid{grid-template-columns:1fr}}
.drop-zone{
  border:2px dashed var(--border);border-radius:var(--r);
  min-height:150px;padding:1.2rem 1rem;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.6rem;
  cursor:pointer;transition:border-color .2s,background .2s;
  position:relative;text-align:center;
}
.drop-zone:hover{border-color:var(--accent);background:rgba(232,255,90,.03)}
.drop-zone.filled{border-color:var(--accent2);border-style:solid}
.drop-zone.drag-over{border-color:var(--accent);background:rgba(232,255,90,.06)}
.drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dz-icon{font-size:1.8rem;line-height:1}
.dz-title{font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700}
.dz-hint{font-size:.72rem;color:var(--muted)}
.dz-preview{width:100%;max-height:110px;object-fit:contain;border-radius:6px;display:none}
.drop-zone.filled .dz-preview{display:block}
.drop-zone.filled .dz-icon,.drop-zone.filled .dz-hint{display:none}
.dz-filename{font-family:'DM Mono',monospace;font-size:.65rem;color:var(--accent2);
  margin-top:.3rem;word-break:break-all;display:none}
.drop-zone.filled .dz-filename{display:block}

/* Parameters */
.param-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.75rem}
.param-row{display:flex;flex-direction:column;gap:.35rem}
.param-label{font-family:'DM Mono',monospace;font-size:.62rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.08em;display:flex;justify-content:space-between;align-items:center}
.param-unit{color:var(--border-hover);font-size:.58rem}
.param-input{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;padding:.5rem .75rem;
  font-family:'DM Mono',monospace;font-size:.95rem;color:var(--accent);
  width:100%;transition:border-color .2s;
  -moz-appearance:textfield;
}
.param-input::-webkit-outer-spin-button,
.param-input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
.param-input:focus{outline:none;border-color:var(--accent)}
.param-input.error{border-color:var(--danger);color:var(--danger)}
.param-input.warn{border-color:var(--warn)}

/* Geometry diagram */
.diagram-wrap{background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r);padding:1rem;overflow:hidden}
.diagram-svg{width:100%;height:auto}

/* Validation bar */
.val-bar{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r);padding:.75rem 1rem;
  font-family:'DM Mono',monospace;font-size:.75rem;
  display:flex;flex-direction:column;gap:.4rem;
  min-height:3rem;
}
.val-row{display:flex;justify-content:space-between;align-items:center;gap:.5rem}
.val-core{display:flex;gap:.5rem;align-items:center}
.val-core-label{color:var(--muted);font-size:.65rem}
.val-core-val{font-size:.85rem}
.val-core-val.ok{color:var(--accent2)}
.val-core-val.warn{color:var(--warn)}
.val-core-val.err{color:var(--danger)}
.val-msgs{display:flex;flex-direction:column;gap:.2rem}
.val-msg-item{display:flex;gap:.4rem;align-items:flex-start;font-size:.68rem}
.val-msg-item.error{color:var(--danger)}
.val-msg-item.warning{color:var(--warn)}
.val-dot{margin-top:.05rem;flex-shrink:0}

/* Generate button */
.btn-gen{
  width:100%;padding:.95rem;
  background:var(--accent);color:#0d0d0f;
  border:none;border-radius:var(--r);
  font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;
  cursor:pointer;transition:transform .15s,box-shadow .15s,opacity .2s;
}
.btn-gen:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 24px rgba(232,255,90,.25)}
.btn-gen:disabled{opacity:.35;cursor:not-allowed}

/* Progress */
.prog-wrap{display:none;flex-direction:column;gap:.6rem}
.prog-wrap.on{display:flex}
.prog-track{width:100%;height:5px;background:var(--border);border-radius:999px;overflow:hidden}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));
  border-radius:999px;width:0%;transition:width .4s ease}
.prog-info{display:flex;justify-content:space-between;
  font-family:'DM Mono',monospace;font-size:.68rem;color:var(--muted)}
.prog-pct{color:var(--accent)}

/* Result */
.result-wrap{display:none;flex-direction:column;gap:.9rem}
.result-wrap.on{display:flex}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:.5rem}
.stat{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:.6rem .9rem}
.stat-l{font-family:'DM Mono',monospace;font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.stat-v{font-family:'DM Mono',monospace;font-size:.88rem;color:var(--accent2);margin-top:.15rem}
.btn-dl{
  padding:.85rem 1.4rem;background:transparent;color:var(--accent2);
  border:2px solid var(--accent2);border-radius:var(--r);
  font-family:'Syne',sans-serif;font-weight:700;font-size:.9rem;
  cursor:pointer;transition:background .2s,color .2s,transform .15s;
  display:flex;align-items:center;gap:.5rem;justify-content:center;
}
.btn-dl:hover{background:var(--accent2);color:#0d0d0f;transform:translateY(-1px)}

/* Error banner */
.err-banner{display:none;background:rgba(255,90,90,.07);border:1px solid rgba(255,90,90,.3);
  border-radius:var(--r);padding:.8rem 1rem;
  font-family:'DM Mono',monospace;font-size:.73rem;color:var(--danger);line-height:1.5}
.err-banner.on{display:block}

/* Alert modal */
.modal-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);
  z-index:1000;align-items:center;justify-content:center;
}
.modal-overlay.on{display:flex}
.modal{
  background:var(--surface);border:1px solid var(--border);
  border-radius:20px;padding:2rem;max-width:440px;width:90%;
  display:flex;flex-direction:column;gap:1.2rem;
}
.modal-icon{font-size:2.5rem;text-align:center}
.modal-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;text-align:center;color:var(--danger)}
.modal-body{font-size:.85rem;color:var(--muted);line-height:1.6;text-align:center}
.modal-body strong{color:var(--text)}
.modal-diagram{background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r);padding:.75rem;text-align:center}
.modal-actions{display:flex;gap:.75rem}
.modal-btn{flex:1;padding:.75rem;border-radius:var(--r);font-family:'Syne',sans-serif;
  font-weight:700;font-size:.85rem;cursor:pointer;transition:all .15s}
.modal-btn.primary{background:var(--accent);color:#0d0d0f;border:none}
.modal-btn.primary:hover{transform:translateY(-1px)}
.modal-btn.secondary{background:transparent;color:var(--muted);border:1px solid var(--border)}
.modal-btn.secondary:hover{border-color:var(--accent2);color:var(--text)}
</style>
</head>
<body>

<header>
  <div class="logo">Card → STL</div>
  <div class="logo-sub">Fung-AI Studio · Business Card Generator</div>
  <div class="badge">Algorithm <em>v9</em> · Pure per-pixel · Zero seams · Adjustable parameters</div>
</header>

<div class="main">

  <!-- Step 1: Images -->
  <div class="panel">
    <div class="panel-title">01 — Upload Images</div>
    <div class="upload-grid">
      <div class="drop-zone" id="zone-front">
        <input type="file" id="inp-front" accept="image/*">
        <div class="dz-icon">🖼</div>
        <div class="dz-title">Front Face</div>
        <div class="dz-hint">Text · Name · Contact</div>
        <img class="dz-preview" id="prev-front" alt="">
        <div class="dz-filename" id="name-front"></div>
      </div>
      <div class="drop-zone" id="zone-back">
        <input type="file" id="inp-back" accept="image/*">
        <div class="dz-icon">🎨</div>
        <div class="dz-title">Back Face</div>
        <div class="dz-hint">Icons · Logo · Illustration</div>
        <img class="dz-preview" id="prev-back" alt="">
        <div class="dz-filename" id="name-back"></div>
      </div>
    </div>
  </div>

  <!-- Step 2: Parameters -->
  <div class="panel">
    <div class="panel-title">
      <span>02 — Print Parameters</span>
      <span style="color:var(--accent2);font-size:.6rem;cursor:pointer" onclick="resetParams()">↺ Reset defaults</span>
    </div>

    <!-- Geometry diagram -->
    <div class="diagram-wrap">
      <svg class="diagram-svg" id="geo-svg" viewBox="0 0 600 130" xmlns="http://www.w3.org/2000/svg">
        <!-- drawn by JS -->
      </svg>
    </div>

    <div class="param-grid">
      <div class="param-row">
        <div class="param-label">Total Thickness <span class="param-unit">mm</span></div>
        <input class="param-input" id="p-total_t" type="number" min="0.8" max="10" step="0.1" value="2.0">
      </div>
      <div class="param-row">
        <div class="param-label">Front Engraving <span class="param-unit">mm</span></div>
        <input class="param-input" id="p-recess_front" type="number" min="0.1" max="3" step="0.05" value="0.5">
      </div>
      <div class="param-row">
        <div class="param-label">Back Engraving <span class="param-unit">mm</span></div>
        <input class="param-input" id="p-recess_back" type="number" min="0.1" max="3" step="0.05" value="0.5">
      </div>
      <div class="param-row">
        <div class="param-label">Corner Radius <span class="param-unit">mm</span></div>
        <input class="param-input" id="p-corner_r" type="number" min="0.5" max="8" step="0.5" value="3.0">
      </div>
      <div class="param-row">
        <div class="param-label">Edge Chamfer <span class="param-unit">mm</span></div>
        <input class="param-input" id="p-chamfer" type="number" min="0.2" max="3" step="0.1" value="1.5">
      </div>
    </div>

    <!-- Live validation bar -->
    <div class="val-bar" id="val-bar">
      <div class="val-row">
        <div class="val-core">
          <span class="val-core-label">SOLID CORE:</span>
          <span class="val-core-val" id="core-val">—</span>
        </div>
        <span style="font-size:.6rem;color:var(--muted)">= Thickness − Front − Back</span>
      </div>
      <div class="val-msgs" id="val-msgs"></div>
    </div>
  </div>

  <!-- Step 3: Generate -->
  <div class="panel">
    <div class="panel-title">03 — Generate STL</div>
    <button class="btn-gen" id="btn-gen" disabled>Upload images to continue</button>
    <div class="prog-wrap" id="prog-wrap">
      <div class="prog-track"><div class="prog-fill" id="prog-fill"></div></div>
      <div class="prog-info"><span id="prog-msg">Starting…</span><span class="prog-pct" id="prog-pct">0%</span></div>
    </div>
    <div class="err-banner" id="err-banner"></div>
    <div class="result-wrap" id="result-wrap">
      <div class="panel-title">04 — Download</div>
      <div class="stats-grid" id="stats-grid"></div>
      <button class="btn-dl" id="btn-dl">↓ Download STL</button>
    </div>
  </div>

</div>

<!-- Overlap warning modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-icon">⚠️</div>
    <div class="modal-title">Engraving Overlap Detected</div>
    <div class="modal-body" id="modal-body">
      The front and back engravings intersect inside the card,
      creating hollow or impossible geometry.
    </div>
    <div class="modal-diagram" id="modal-diagram"></div>
    <div class="modal-actions">
      <button class="modal-btn secondary" onclick="closeModal()">Dismiss</button>
      <button class="modal-btn primary" onclick="autoFix()">Auto-fix thickness ↑</button>
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let frontFile = null, backFile = null, jobId = null;
let lastValidation = {ok: true, core: 1.0, errors: [], warnings: []};
let validateTimer = null;

const DEFAULTS = {total_t:2.0, recess_front:0.5, recess_back:0.5, corner_r:3.0, chamfer:1.5};

// ── Param helpers ──────────────────────────────────────────────────────────
function getParams() {
  return {
    total_t:      parseFloat(document.getElementById('p-total_t').value)      || 2.0,
    recess_front: parseFloat(document.getElementById('p-recess_front').value) || 0.5,
    recess_back:  parseFloat(document.getElementById('p-recess_back').value)  || 0.5,
    corner_r:     parseFloat(document.getElementById('p-corner_r').value)     || 3.0,
    chamfer:      parseFloat(document.getElementById('p-chamfer').value)      || 1.5,
  };
}

function resetParams() {
  Object.entries(DEFAULTS).forEach(([k,v]) => {
    const el = document.getElementById('p-' + k);
    if (el) el.value = v;
  });
  scheduleValidate();
}

// ── Upload zones ───────────────────────────────────────────────────────────
function setupZone(inputId, zoneId, prevId, nameId, which) {
  const input = document.getElementById(inputId);
  const zone  = document.getElementById(zoneId);
  const prev  = document.getElementById(prevId);
  const name  = document.getElementById(nameId);

  function handleFile(file) {
    if (!file) return;
    if (which === 'front') frontFile = file;
    else backFile = file;
    const reader = new FileReader();
    reader.onload = e => { prev.src = e.target.result; };
    reader.readAsDataURL(file);
    zone.classList.add('filled');
    name.textContent = file.name;
    updateGenBtn();
  }

  input.addEventListener('change', e => handleFile(e.target.files[0]));
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
  });
}

setupZone('inp-front','zone-front','prev-front','name-front','front');
setupZone('inp-back', 'zone-back', 'prev-back', 'name-back', 'back');

// ── Validation ─────────────────────────────────────────────────────────────
function scheduleValidate() {
  clearTimeout(validateTimer);
  validateTimer = setTimeout(doValidate, 300);
  // Instant local pre-check
  localValidate();
}

function localValidate() {
  const p = getParams();
  const core = p.total_t - p.recess_front - p.recess_back;
  updateDiagram(p, core);
  updateCoreDisplay(core, []);
}

async function doValidate() {
  const p = getParams();
  try {
    const res  = await fetch('/api/validate', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(p)
    });
    const data = await res.json();
    lastValidation = data;
    updateValidationUI(data);
    updateDiagram(p, data.core);

    // Show modal if error and user was editing
    if (!data.ok && data.errors.length > 0 && data.errors[0].includes('overlap')) {
      showOverlapModal(p, data.core);
    }
  } catch(e) { /* network error - use local */ }
  updateGenBtn();
}

function updateCoreDisplay(core, errors) {
  const el = document.getElementById('core-val');
  el.textContent = core.toFixed(3) + ' mm';
  el.className = 'val-core-val';
  if (core < 0.3)  el.classList.add('err');
  else if (core < 0.6) el.classList.add('warn');
  else el.classList.add('ok');
}

function updateValidationUI(data) {
  updateCoreDisplay(data.core, data.errors);
  const msgs = document.getElementById('val-msgs');
  msgs.innerHTML = '';
  [...data.errors.map(m=>({t:'error',m})), ...data.warnings.map(m=>({t:'warning',m}))].forEach(({t,m}) => {
    const div = document.createElement('div');
    div.className = 'val-msg-item ' + t;
    div.innerHTML = `<span class="val-dot">${t==='error'?'✕':'⚠'}</span><span>${m}</span>`;
    msgs.appendChild(div);
  });
  // Highlight relevant inputs
  document.getElementById('p-total_t').classList.remove('error','warn');
  document.getElementById('p-recess_front').classList.remove('error','warn');
  document.getElementById('p-recess_back').classList.remove('error','warn');
  if (!data.ok) {
    ['p-total_t','p-recess_front','p-recess_back'].forEach(id =>
      document.getElementById(id).classList.add('error'));
  }
}

// ── Live geometry diagram ──────────────────────────────────────────────────
function updateDiagram(p, core) {
  const svg = document.getElementById('geo-svg');
  const W=600, H=130, pad=40;
  const cardW = W - pad*2;
  const cardX = pad;
  const scaleY = (H - 40) / Math.max(p.total_t, 1);
  const totalPx = Math.min(p.total_t * scaleY, H-40);
  const frontPx = Math.min(p.recess_front * scaleY, totalPx * 0.45);
  const backPx  = Math.min(p.recess_back  * scaleY, totalPx * 0.45);
  const corePx  = totalPx - frontPx - backPx;
  const cardTop = (H - totalPx) / 2;
  const cardBot = cardTop + totalPx;

  const overlap = core < 0.3;
  const coreColor = core < 0 ? '#ff5a5a' : core < 0.3 ? '#ffaa5a' : '#5affcd';
  const frontColor = '#e8ff5a';
  const backColor  = '#ff9f5a';

  svg.innerHTML = `
    <!-- Card outline -->
    <rect x="${cardX}" y="${cardTop}" width="${cardW}" height="${totalPx}"
      fill="#1a1a20" stroke="#2a2a30" stroke-width="1" rx="4"/>

    <!-- Front engraving (top) -->
    <rect x="${cardX}" y="${cardTop}" width="${cardW}" height="${frontPx}"
      fill="${overlap ? 'rgba(255,90,90,.25)' : 'rgba(232,255,90,.15)'}"
      stroke="${frontColor}" stroke-width="1" stroke-dasharray="${overlap?'4,3':''}"/>
    <text x="${cardX + cardW/2}" y="${cardTop + frontPx/2 + 4}"
      fill="${frontColor}" font-family="DM Mono,monospace" font-size="10" text-anchor="middle">
      Front ${p.recess_front.toFixed(2)}mm</text>

    <!-- Back engraving (bottom) -->
    <rect x="${cardX}" y="${cardBot - backPx}" width="${cardW}" height="${backPx}"
      fill="${overlap ? 'rgba(255,90,90,.25)' : 'rgba(255,159,90,.15)'}"
      stroke="${backColor}" stroke-width="1" stroke-dasharray="${overlap?'4,3':''}"/>
    <text x="${cardX + cardW/2}" y="${cardBot - backPx/2 + 4}"
      fill="${backColor}" font-family="DM Mono,monospace" font-size="10" text-anchor="middle">
      Back ${p.recess_back.toFixed(2)}mm</text>

    <!-- Core (solid PLA) -->
    ${corePx > 2 ? `
    <rect x="${cardX}" y="${cardTop + frontPx}" width="${cardW}" height="${Math.max(corePx,0)}"
      fill="${overlap ? 'rgba(255,90,90,.1)' : 'rgba(90,255,205,.07)'}"
      stroke="${coreColor}" stroke-width="1"/>
    <text x="${cardX + cardW/2}" y="${cardTop + frontPx + Math.max(corePx,0)/2 + 4}"
      fill="${coreColor}" font-family="DM Mono,monospace" font-size="11" text-anchor="middle" font-weight="bold">
      Core ${Math.max(core,0).toFixed(3)}mm ${overlap?'⚠ OVERLAP':''}</text>` : `
    <line x1="${cardX}" y1="${cardTop+frontPx}" x2="${cardX+cardW}" y2="${cardTop+frontPx}"
      stroke="${coreColor}" stroke-width="2" stroke-dasharray="4,3"/>
    <text x="${cardX+cardW/2}" y="${cardTop+frontPx-4}"
      fill="${coreColor}" font-family="DM Mono,monospace" font-size="10" text-anchor="middle">
      Core: ${Math.max(core,0).toFixed(3)}mm ⚠</text>`}

    <!-- Total thickness label -->
    <line x1="${cardX-8}" y1="${cardTop}" x2="${cardX-8}" y2="${cardBot}"
      stroke="#3a3a45" stroke-width="1"/>
    <line x1="${cardX-12}" y1="${cardTop}" x2="${cardX-4}" y2="${cardTop}" stroke="#3a3a45" stroke-width="1"/>
    <line x1="${cardX-12}" y1="${cardBot}" x2="${cardX-4}" y2="${cardBot}" stroke="#3a3a45" stroke-width="1"/>
    <text x="${cardX-14}" y="${(cardTop+cardBot)/2+4}"
      fill="#6b6b78" font-family="DM Mono,monospace" font-size="10" text-anchor="end">
      ${p.total_t.toFixed(1)}mm</text>

    <!-- Chamfer indicator -->
    <text x="${cardX+cardW+6}" y="${cardTop+16}"
      fill="#3a3a45" font-family="DM Mono,monospace" font-size="9">
      ⌐${p.chamfer.toFixed(1)}mm</text>
  `;
}

// ── Overlap modal ──────────────────────────────────────────────────────────
let modalShownForCore = null;

function showOverlapModal(p, core) {
  const key = `${p.total_t}_${p.recess_front}_${p.recess_back}`;
  if (modalShownForCore === key) return;
  modalShownForCore = key;

  const needed = (p.recess_front + p.recess_back + 0.3).toFixed(2);
  document.getElementById('modal-body').innerHTML =
    `The front engraving (<strong>${p.recess_front}mm</strong>) and back engraving
    (<strong>${p.recess_back}mm</strong>) together exceed the card thickness
    (<strong>${p.total_t}mm</strong>).<br><br>
    Solid core = <strong style="color:var(--danger)">${core.toFixed(3)}mm</strong>
    — must be ≥ 0.3mm.<br>
    Minimum required thickness: <strong>${needed}mm</strong>.`;

  document.getElementById('modal').classList.add('on');
}

function closeModal() {
  document.getElementById('modal').classList.remove('on');
}

function autoFix() {
  const p = getParams();
  const needed = p.recess_front + p.recess_back + 0.3;
  const rounded = Math.ceil(needed * 10) / 10;
  document.getElementById('p-total_t').value = rounded.toFixed(1);
  closeModal();
  modalShownForCore = null;
  scheduleValidate();
}

document.getElementById('modal').addEventListener('click', e => {
  if (e.target === document.getElementById('modal')) closeModal();
});

// ── Param listeners ────────────────────────────────────────────────────────
['p-total_t','p-recess_front','p-recess_back','p-corner_r','p-chamfer'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    scheduleValidate();
  });
});

// ── Generate button state ──────────────────────────────────────────────────
function updateGenBtn() {
  const btn = document.getElementById('btn-gen');
  if (!frontFile || !backFile) {
    btn.disabled = true;
    btn.textContent = 'Upload both images to continue';
  } else if (!lastValidation.ok) {
    btn.disabled = true;
    btn.textContent = 'Fix parameter errors first';
  } else {
    btn.disabled = false;
    btn.textContent = 'Generate STL →';
  }
}

// ── Generate ───────────────────────────────────────────────────────────────
document.getElementById('btn-gen').addEventListener('click', async () => {
  if (!frontFile || !backFile || !lastValidation.ok) return;

  clearError();
  document.getElementById('result-wrap').classList.remove('on');
  document.getElementById('prog-wrap').classList.add('on');
  document.getElementById('btn-gen').disabled = true;
  document.getElementById('btn-gen').textContent = 'Processing…';
  setProgress(0, 'Uploading…');

  const fd = new FormData();
  fd.append('front', frontFile);
  fd.append('back',  backFile);
  const p = getParams();
  Object.entries(p).forEach(([k,v]) => fd.append(k, v));

  try {
    const res = await fetch('/api/generate', {method:'POST', body:fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    jobId = data.job_id;
    pollStatus();
  } catch(e) {
    showError(e.message);
    resetGenBtn();
  }
});

function pollStatus() {
  const iv = setInterval(async () => {
    try {
      const res  = await fetch('/api/status/' + jobId);
      const data = await res.json();
      setProgress(data.progress||0, data.status_msg||'…');
      if (data.status === 'done')  { clearInterval(iv); showResult(data); }
      if (data.status === 'error') { clearInterval(iv); showError(data.error||'Unknown'); resetGenBtn(); }
    } catch(e) { clearInterval(iv); showError(e.message); }
  }, 600);
}

function setProgress(pct, msg) {
  document.getElementById('prog-fill').style.width = pct + '%';
  document.getElementById('prog-msg').textContent = msg;
  document.getElementById('prog-pct').textContent = pct + '%';
}

function showResult(data) {
  document.getElementById('prog-wrap').classList.remove('on');
  document.getElementById('result-wrap').classList.add('on');
  resetGenBtn();
  const p = data.params || {};
  const core = ((p.total_t||2) - (p.recess_front||.5) - (p.recess_back||.5)).toFixed(2);
  document.getElementById('stats-grid').innerHTML = `
    <div class="stat"><div class="stat-l">File Size</div><div class="stat-v">${(data.size_mb||0).toFixed(1)} MB</div></div>
    <div class="stat"><div class="stat-l">Thickness</div><div class="stat-v">${(p.total_t||2).toFixed(1)} mm</div></div>
    <div class="stat"><div class="stat-l">Front engrave</div><div class="stat-v">${(p.recess_front||.5).toFixed(2)} mm</div></div>
    <div class="stat"><div class="stat-l">Back engrave</div><div class="stat-v">${(p.recess_back||.5).toFixed(2)} mm</div></div>
    <div class="stat"><div class="stat-l">Solid core</div><div class="stat-v">${core} mm</div></div>
    <div class="stat"><div class="stat-l">Algorithm</div><div class="stat-v">v9 ✓</div></div>
  `;
  document.getElementById('btn-dl').onclick = () => { window.location.href = '/api/download/' + jobId; };
}

function showError(msg) {
  const el = document.getElementById('err-banner');
  el.textContent = '✕ ' + msg; el.classList.add('on');
}
function clearError() { document.getElementById('err-banner').classList.remove('on'); }
function resetGenBtn() {
  document.getElementById('btn-gen').disabled = false;
  document.getElementById('btn-gen').textContent = 'Generate Again →';
  updateGenBtn();
}

// Init
localValidate();
doValidate();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("=" * 52)
    print("  Fung-AI Card→STL  (v9 · adjustable params)")
    print("  http://localhost:5000")
    print("=" * 52)
    app.run(debug=False, host="0.0.0.0", port=5000)
