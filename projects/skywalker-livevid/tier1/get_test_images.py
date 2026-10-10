"""Downloads the 3 test images and saves them center-cropped to 512x512 under tier1/out/inputs."""
import io
import urllib.request

from PIL import Image

from common import INPUT_DIR, SIZE, TEST_IMAGES


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in TEST_IMAGES.items():
        dst = INPUT_DIR / f"{name}.png"
        if dst.exists():
            print(f"skip {dst}")
            continue
        with urllib.request.urlopen(url, timeout=60) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)
        img.save(dst)
        print(f"saved {dst} from {url}")


if __name__ == "__main__":
    main()
