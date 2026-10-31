"""Assemble existing charts for visual review without modifying originals."""
from pathlib import Path
import json
from PIL import Image, ImageDraw

pv = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
parent = pv / "运行结果" / "20260929_000009_045346"
child = pv / "运行结果" / "20261009_daylight_recalibration_01"


def sheet(paths, filename, columns, cell_size):
    width, height = cell_size
    canvas = Image.new("RGB", (columns * width, ((len(paths) + columns - 1) // columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    refs = []
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image.load()
            image.thumbnail((width - 16, height - 36), Image.Resampling.LANCZOS)
            x, y = (i % columns) * width, (i // columns) * height
            canvas.paste(image, (x + 8, y + 28))
        label = str(path.relative_to(parent)) if path.is_relative_to(parent) else path.name
        draw.text((x + 8, y + 8), label, fill="black")
        refs.append(str(path.relative_to(pv)))
    canvas.save(out / filename)
    return {"sheet": filename, "source_pngs": refs}


result = {
    "rolling_loss": sheet(sorted(parent.glob("experiments/*/*/visualizations/training_loss.png")),
                          "rolling_loss_review.png", 3, (580, 340)),
    "recalibrated_inference": sheet(sorted((child / "visualizations").glob("*.png")),
                                   "recalibrated_inference_review.png", 2, (800, 420)),
}
(out / "png_review_sources.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value["sheet"] for key, value in result.items()}))
