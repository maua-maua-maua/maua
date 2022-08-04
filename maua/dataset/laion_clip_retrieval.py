import argparse
import base64
import json
import multiprocessing as mp
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

import filetype
import requests
from numpy import unique
from PIL import ImageFile
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

USER_AGENT = {"User-Agent": "Maua", "From": "https://github.com/maua-maua-maua/maua"}


def remote_image_size(url):
    resume_header = {"Range": "bytes=0-2000000", **USER_AGENT}
    data = requests.get(url, stream=True, headers=resume_header).content
    p = ImageFile.Parser()
    p.feed(data)
    if p.image:
        return p.image.size
    else:
        return (-1, -1)


def encode_image_prompt(file):
    if file is None:
        return None
    with open(file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=True)
    parser.add_argument("--out_dir", type=str, default='output/', help="Directory to save images in")
    parser.add_argument("--texts", type=str, default=[], nargs="*", help="Text prompt to retrieve images for")
    parser.add_argument("--images", type=str, default=[], nargs="*", help="Image file to retrieve images for")
    parser.add_argument("--urls", type=str, default=[], nargs="*", help="Image URL to retrieve images for")
    parser.add_argument("--modality", default="image", choices=["image", "text"], help="Whether to compare prompt with image or caption embedding")
    parser.add_argument("--number", type=int, default=40, help="Number of images to retrieve")
    parser.add_argument("--index", type=str, default="laion5B", choices=["laion5B", "laion_400m"], help="Which dataset to search in")
    parser.add_argument("--multilingual", action="store_true", help="Use multilingual CLIP embeddings")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable deduplication (might return same image multiple times)")
    parser.add_argument("--safety", action="store_true", help="Enable safe search (do not return NSFW results)")
    parser.add_argument("--no-violence", action="store_true", help="Enable violence filtering")
    parser.add_argument("--aesthetic-score", type=int, default=9, help="Return results with at least the specified aesthetic score. 0 to disable")
    parser.add_argument("--aesthetic-weight", type=float, default=0.5, help="Weight of aesthetic score versus similarity to prompt")
    parser.add_argument("--size", type=int, default=None, help="Minimum side length of images retrieved (will return less than specified --number!)")
    args = parser.parse_args()
    # fmt: on

    assert (
        len(args.texts) > 0 or len(args.images) > 0 or len(args.urls) > 0
    ), "At least one text, image, or url prompt must be supplied!"

    texts = args.texts + [None for _ in args.images] + [None for _ in args.urls]
    images = [None for _ in args.texts] + args.images + [None for _ in args.urls]
    urls = [None for _ in args.texts] + [None for _ in args.images] + args.urls

    candidates = []
    for text, image, url in zip(tqdm(texts, desc="Retrieving similar images from knn5.laion.ai"), images, urls):
        data = (
            json.dumps(
                {
                    "text": "|T|E|X|T|",
                    "image": encode_image_prompt(image),
                    "image_url": url,
                    "embedding_input": None,
                    "modality": args.modality,
                    "num_images": args.number,
                    "indice_name": args.index,
                    "num_result_ids": args.number,
                    "use_mclip": args.multilingual,
                    "deduplicate": not args.no_deduplicate,
                    "use_safety_model": args.safety,
                    "use_violence_detector": not args.no_violence,
                    "aesthetic_score": str(args.aesthetic_score) if args.aesthetic_score else '""',
                    "aesthetic_weight": str(args.aesthetic_weight),
                }
            )
            .replace(" ", "")
            .replace("|T|E|X|T|", text if text is not None else "null")
            .replace('"null"', "null")
        )
        response = requests.post("https://knn5.laion.ai/knn-service", data=data)
        candidates.extend([r["url"] for r in response.json()])
    candidates = unique(candidates)
    print(f"Found {len(candidates)} candidates.")

    def maybe_download(url):
        try:
            if args.size is not None and min(remote_image_size(url)) < args.size:
                return

            session = requests.Session()
            adapter = HTTPAdapter(max_retries=Retry(connect=3, backoff_factor=0.25))
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            with session.get(url, allow_redirects=True, headers=USER_AGENT) as response:

                # figure out filename (preferably based on what tne server thinks the filename is)
                fname = Path(urlparse(url).path).name
                if "Content-Disposition" in response.headers:
                    try:
                        fname = response.headers.get("Content-Disposition").split("filename=")[1]
                    except:
                        pass
                fname = unquote(fname).strip('"').strip().replace(" ", "_")

                # try to guess extension
                extension = filetype.guess(response.content).extension
                if extension is not None:
                    fname = "_".join(fname.split(".")[:-1]) + "." + extension

                # save image content
                with open(f"{args.out_dir}/{fname}", "wb") as file:
                    file.write(response.content)

            return True

        except:
            return

    with mp.Pool(mp.cpu_count()) as pool:
        num = 0
        for succeeded in tqdm(
            pool.imap_unordered(maybe_download, candidates), total=len(candidates), desc="Downloading images..."
        ):
            if succeeded:
                num += 1
    print(f"Downloaded {num} images.")
