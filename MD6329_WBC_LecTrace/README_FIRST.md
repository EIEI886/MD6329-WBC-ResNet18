# MD6329 WBC LecTrace — Instructions

A standalone English Python presentation designed for approximately ten minutes. No retraining is required.

## Package contents

- `01_wbc_lectrace.py`: main presentation, with four selected execution states.
- `_wbc_support.py`: setup, optional checkpoint inference, and figure generation. Do not run it separately.
- `assets/`: actual exported tensors, metrics, and images. Keep this folder intact.
- `_site/`: prebuilt presentation, viewed through an HTTP server.
- `requirements.txt`: dependencies; Python 3.11 or later is required.

## Colab: append four code cells

Download `MD6329_WBC_LecTrace.zip`. Append the following four **code cells** to the bottom of your existing Colab notebook and run them in order. Do not select **Run all** or rerun the training cells. Upload the ZIP directly; no manual extraction is needed.

### Cell 1 — Install LecTrace

```python
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'lectrace==1.1.30', 'numpy', 'matplotlib', 'pillow'])
```

### Cell 2 — Upload and extract the presentation ZIP

Select **MD6329_WBC_LecTrace.zip**, not the original results ZIP.

```python
from google.colab import files
from pathlib import Path
import zipfile

uploaded = files.upload()
archive_name = next(name for name in uploaded if name.endswith('.zip'))
with zipfile.ZipFile(archive_name) as z:
    z.extractall('/content')
LECTRACE_DIR = Path('/content/MD6329_WBC_LecTrace')
assert (LECTRACE_DIR/'01_wbc_lectrace.py').is_file()
print('LecTrace folder:', LECTRACE_DIR)
```

### Cell 3 — Connect Drive and build

The script automatically finds the original C/seed42 checkpoint in Drive and performs one CPU forward pass. No model training is performed.

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run(['lectrace', 'build', '01_wbc_lectrace.py',
                '--output', '_site', '--no-incremental'],
               cwd=LECTRACE_DIR, check=True)
```

Default checkpoint location:

`/content/drive/MyDrive/MD6329_WBC_ResNet18_v2_dedup/C_layer4_noaug_seed42/best.pt`

If you changed the checkpoint location, run this optional snippet **before Cell 3**:

```python
import os
os.environ['WBC_CHECKPOINT'] = '/your/actual/path/C_layer4_noaug_seed42/best.pt'
```

The fixed example and normalized input are already included. No dataset download is needed. If no checkpoint is found, the script explicitly switches to recorded-state replay.

### Cell 4 — Open the preview

```python
from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from threading import Thread
from urllib.request import urlopen
from google.colab import output

site_dir = Path('/content/MD6329_WBC_LecTrace/_site')
assert (site_dir / 'index.html').is_file(), (
    'Presentation files are missing. Rerun the build cell first.'
)

# Stop a previous server created by this cell.
if 'wbc_httpd' in globals():
    wbc_httpd.shutdown()
    wbc_httpd.server_close()

handler = partial(SimpleHTTPRequestHandler, directory=str(site_dir))
wbc_httpd = ThreadingHTTPServer(('0.0.0.0', 0), handler)
wbc_port = wbc_httpd.server_address[1]
Thread(target=wbc_httpd.serve_forever, daemon=True).start()

with urlopen(f'http://127.0.0.1:{wbc_port}/index.html', timeout=5) as response:
    print('Presentation server:', response.status)

output.serve_kernel_port_as_iframe(wbc_port, height=850)
```

The presentation appears below the cell inside Colab. Click inside it and use the left/right arrow keys to step through the code and variables. A successful local server check prints `Presentation server: 200`.

## Updating an existing Colab preview

If LecTrace is already installed, rerun only Cells 2, 3, and 4 with the updated ZIP. Cell 2 overwrites the package files, Cell 3 rebuilds the presentation, and Cell 4 restarts the preview. No training cells need to run.

## Run locally instead

Extract the complete package, open a terminal in its folder, and run:

```bash
python3 -m pip install -r requirements.txt
lectrace serve 01_wbc_lectrace.py --port 7000
```

Open `http://localhost:7000`. GitHub deployment is unnecessary. Without a checkpoint, the presentation uses recorded-state replay; it does not download pretrained weights or train a model.

## Execution modes and verification

The presentation identifies its mode explicitly:

1. **Live checkpoint inference:** Original C/seed42 weights recompute convolutional and pooling states on CPU. The presentation then executes pooling, the classifier matrix multiplication, softmax, and argmax. This mode requires `torch` and `torchvision`, already used in the original Colab.
2. **Recorded-state replay:** Without weights, the script loads the actual recorded convolutional features and logits, recomputes pooling, softmax, and argmax, and checks the saved outputs. It does not rerun the CNN. The bundled `_site` uses this portable mode.

Recorded-state execution and static-site generation were tested during package preparation. The original `best.pt` is not included in this package; live inference requires the original checkpoint in your Colab environment. To submit a static presentation built in live mode, rebuild with the original checkpoint available and then repackage the folder.

Variable previews show tensor shapes, ranges, and selected values to keep the viewer responsive. Calculations use the complete arrays.

## Suggested ten-minute presentation

- **0:00–2:30:** healthcare problem, data, and A/B/C controlled comparisons.
- **2:30–3:30:** input `[1,3,224,224]`; resizing and normalization.
- **3:30–4:45:** convolutional features `[1,512,7,7]`; the mean feature map is not Grad-CAM.
- **4:45–5:45:** spatial averaging produces 512 features.
- **5:45–7:30:** logits → softmax → argmax; output probability is not clinical certainty.
- **7:30–8:45:** actual results and learning curves; substantial fine-tuning improvement and a small augmentation difference.
- **8:45–9:30:** two errors and predicted-class Grad-CAM.
- **9:30–10:00:** conclusions, course connections, and limitations.

The script's `note()` calls provide audience-facing explanations. The code steps cover **four selected execution states**, plus introductory and concluding material.

## Submission

Retain the main script, support script, assets, requirements, and README in the final package. Include `_site` if convenient. Screenshots alone do not replace the Python presentation. After rebuilding in Colab, use this optional additional cell to download the updated folder:

```python
import shutil
from google.colab import files
updated_zip = shutil.make_archive('/content/MD6329_WBC_LecTrace_ready',
                                  'zip', root_dir='/content',
                                  base_dir='MD6329_WBC_LecTrace')
files.download(updated_zip)
```

This package does not include the report or executed training notebook. A video is optional under the assignment instructions. The presentation's source list does not replace the report's requirement for at least ten relevant citations.
