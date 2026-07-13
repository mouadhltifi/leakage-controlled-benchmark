# CAMEF repro — step 0: download the authors' dataset and map its structure,
# so we can wire CAMEF's expected data/event + data/series layout before any GPU run.
import os, sys, subprocess, zipfile, glob

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
import gdown

FID = "1ejmUgVOOiHST3RjHPqsn2P3i_woGPYX_"   # dataset.zip (from the CAMEF README)
OUT = "/kaggle/working/dataset.zip"

print("=== downloading dataset.zip from Google Drive ===", flush=True)
try:
    gdown.download(id=FID, output=OUT, quiet=False)
except Exception as e:
    print("id-form download failed:", e, "\n-> retrying URL form with fuzzy", flush=True)
    gdown.download(f"https://drive.google.com/uc?id={FID}", OUT, quiet=False, fuzzy=True)

print(f"\n=== downloaded {os.path.getsize(OUT)/1e6:.1f} MB ===", flush=True)

# Inspect zip entries (top of the listing) without extracting everything first
with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
    print(f"zip entries: {len(names)}")
    for n in names[:100]:
        print("  E", n)

# Extract and walk the top 3 levels so we can see series CSVs + event folders
EX = "/kaggle/working/x"
os.makedirs(EX, exist_ok=True)
with zipfile.ZipFile(OUT) as z:
    z.extractall(EX)

print("\n=== directory tree (top 3 levels) ===", flush=True)
base_depth = EX.count("/")
for root, dirs, files in os.walk(EX):
    depth = root.count("/") - base_depth
    if depth <= 3:
        print(f"{'  '*depth}{os.path.basename(root) or root}/  dirs={dirs[:12]}  files={files[:6]}")

print("\n=== any *.csv (candidate series files) ===", flush=True)
for p in glob.glob(EX + "/**/*.csv", recursive=True)[:30]:
    sz = os.path.getsize(p)
    print(f"  {sz/1e6:7.2f} MB  {p}")
    # peek the header of the first one
    try:
        with open(p) as f:
            print("     header:", f.readline().strip()[:160])
    except Exception:
        pass

print("\n=== sample event text files (first few *.txt) ===", flush=True)
for p in glob.glob(EX + "/**/*full_summary*.txt", recursive=True)[:5]:
    print("  ", p)

print("\nDONE", flush=True)
