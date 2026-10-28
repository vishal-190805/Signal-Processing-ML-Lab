import nbformat
import base64
import os
from PIL import Image
import io

nb_path = "SVD_Image_Compression_Assignment (1).ipynb"
out_dir = "extracted_figures"
os.makedirs(out_dir, exist_ok=True)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

count = 0
saved = []
for i, cell in enumerate(nb.cells):
    if cell.get("cell_type") != "code":
        continue
    outputs = cell.get("outputs", [])
    for j, output in enumerate(outputs):
        if output.get("output_type") == "display_data":
            data = output.get("data", {})
            if "image/png" in data:
                count += 1
                png_b64 = data["image/png"]
                img_bytes = base64.b64decode(png_b64)
                img = Image.open(io.BytesIO(img_bytes))
                fname = f"fig_{count:02d}_cell{i}_out{j}.png"
                fpath = os.path.join(out_dir, fname)
                img.save(fpath)
                saved.append((count, i, j, fpath, img.size))
                print(f"Saved {fpath} size={img.size}")
        elif output.get("output_type") == "execute_result":
            data = output.get("data", {})
            if "image/png" in data:
                count += 1
                png_b64 = data["image/png"]
                img_bytes = base64.b64decode(png_b64)
                img = Image.open(io.BytesIO(img_bytes))
                fname = f"fig_{count:02d}_cell{i}_out{j}.png"
                fpath = os.path.join(out_dir, fname)
                img.save(fpath)
                saved.append((count, i, j, fpath, img.size))
                print(f"Saved {fpath} size={img.size}")

print(f"\nTotal images extracted: {count}")
print("\nSummary (fig_num, cell_idx, output_idx, filename, size):")
for s in saved:
    print(s)
