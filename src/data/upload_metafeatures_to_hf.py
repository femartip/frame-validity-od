from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import create_repo, upload_file
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class DatasetSpec:
    local_path: Path
    repo_id: str
    title: str
    description: str
    has_target: bool


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index(names="frame_id")
    return df


def build_readme(spec: DatasetSpec, df: pd.DataFrame) -> str:
    target_note = (
        "- `iou`: mean IoU for the scene\n"
        "- `lrp`: LRP for the scene\n\n"
        "Use either target column downstream depending on the assessor task.\n"
        if spec.has_target
        else "This dataset does not include a prediction target. It is intended as a feature-only baseline.\n"
    )

    columns = "\n".join(f"- `{column}`" for column in df.columns)

    return f"""---
dataset_info:
  description: {spec.description}
---

# {spec.title}

{spec.description}

## Source

- Extracted from the validation split of the ZOD frames dataset
- One row per scene/frame
- `frame_id` keeps the original ZOD frame identifier

## Contents

{target_note}## Columns

{columns}

## Notes

- The tabular features are the meta-features computed in this repository.
- The detector-specific datasets include both detector quality targets so that you can choose the target metric later.
"""


def upload_file_with_retries(
    *,
    path_or_fileobj: str,
    path_in_repo: str,
    repo_id: str,
    token: str,
    max_retries: int,
    retry_wait_seconds: float,
) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            return
        except RequestException as e:
            if attempt == max_retries:
                raise
            logger.warning(
                "Upload failed for %s on attempt %s/%s: %s. Retrying in %.1fs.",
                path_in_repo,
                attempt,
                max_retries,
                e,
                retry_wait_seconds,
            )
            time.sleep(retry_wait_seconds)


def run_git(args: list[str], cwd: str) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def upload_dataset_via_git(spec: DatasetSpec, repo_id: str, token: str, readme_content: str) -> None:
    with tempfile.TemporaryDirectory(prefix="hf-upload-") as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        clone_url = f"https://oauth2:{quote(token, safe='')}@huggingface.co/datasets/{repo_id}"

        run_git(["git", "clone", clone_url, str(repo_dir)], cwd=tmpdir)
        run_git(["git", "checkout", "-B", "main"], cwd=str(repo_dir))
        run_git(["git", "config", "user.name", "Codex Upload Bot"], cwd=str(repo_dir))
        run_git(["git", "config", "user.email", "codex@example.com"], cwd=str(repo_dir))

        shutil.copy2(spec.local_path, repo_dir / spec.local_path.name)
        (repo_dir / "README.md").write_text(readme_content, encoding="utf-8")

        run_git(["git", "add", spec.local_path.name, "README.md"], cwd=str(repo_dir))

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            logger.info("No changes to push for %s", repo_id)
            return

        run_git(
            ["git", "commit", "-m", f"Upload {spec.local_path.name} and README"],
            cwd=str(repo_dir),
        )
        run_git(["git", "push", "origin", "HEAD:main"], cwd=str(repo_dir))


def upload_dataset(
    spec: DatasetSpec,
    token: str,
    namespace: str,
    private: bool,
    max_retries: int,
    retry_wait_seconds: float,
    transport: str,
) -> None:
    """Upload the source CSV and dataset card to a dataset repository."""
    df = read_table(spec.local_path)
    repo_id = f"{namespace}/{spec.repo_id}"
    readme_content = build_readme(spec, df)

    logger.info(f"Creating/accessing repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True, private=private)

    def upload_via_api() -> None:
        logger.info(f"Uploading CSV {spec.local_path.name} ({len(df)} rows) to Hugging Face...")
        upload_file_with_retries(
            path_or_fileobj=str(spec.local_path),
            path_in_repo=spec.local_path.name,
            repo_id=repo_id,
            token=token,
            max_retries=max_retries,
            retry_wait_seconds=retry_wait_seconds,
        )

        logger.info("Uploading dataset card...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            f.flush()
            temp_path = f.name

        try:
            upload_file_with_retries(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token,
                max_retries=max_retries,
                retry_wait_seconds=retry_wait_seconds,
            )
        finally:
            os.unlink(temp_path)

    if transport == "git":
        logger.info("Uploading via git transport...")
        upload_dataset_via_git(spec, repo_id, token, readme_content)
    elif transport == "api":
        upload_via_api()
    else:
        try:
            upload_via_api()
        except RequestException as e:
            logger.warning("API upload failed for %s: %s. Falling back to git transport.", repo_id, e)
            upload_dataset_via_git(spec, repo_id, token, readme_content)

    logger.info(f"Successfully uploaded {spec.local_path.name} -> {repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the ZOD meta-feature tables to Hugging Face as separate datasets."
    )
    parser.add_argument("--namespace", default="femartip", help="Hugging Face namespace / username or org.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument("--private", action="store_true", help="Create the dataset repos as private.")
    parser.add_argument("--enable-hf-transfer", action="store_true", help="Enable hf_transfer accelerated uploads.")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries for each file upload.")
    parser.add_argument("--retry-wait-seconds", type=float, default=5.0, help="Wait time between upload retries.")
    parser.add_argument(
        "--transport",
        choices=["auto", "api", "git"],
        default="auto",
        help="Upload transport. 'auto' tries the API first and falls back to git.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not args.token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN or pass --token.")

    # These CSVs are small enough that reliability matters more than transfer acceleration.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if args.enable_hf_transfer else "0"

    logger.info(f"Uploading to namespace: {args.namespace}")
    logger.info(f"Private repos: {args.private}")
    logger.info(f"hf_transfer enabled: {os.environ['HF_HUB_ENABLE_HF_TRANSFER']}")
    logger.info(f"transport: {args.transport}")

    specs = [
        DatasetSpec(
            local_path=DATA_DIR / "metafeatures.csv",
            repo_id="zod-metafeatures",
            title="ZOD validation meta-features",
            description="Feature-only table built from meta-features extracted from the ZOD validation split.",
            has_target=False,
        ),
        DatasetSpec(
            local_path=DATA_DIR / "faster-rcnn_metafeatures.csv",
            repo_id="zod-faster-rcnn-metafeatures",
            title="ZOD validation meta-features + Faster R-CNN targets",
            description="Meta-features extracted from ZOD validation scenes with Faster R-CNN quality targets.",
            has_target=True,
        ),
        DatasetSpec(
            local_path=DATA_DIR / "yolo_metafeatures.csv",
            repo_id="zod-yolo-metafeatures",
            title="ZOD validation meta-features + YOLO targets",
            description="Meta-features extracted from ZOD validation scenes with YOLO quality targets.",
            has_target=True,
        ),
    ]

    for spec in specs:
        if not spec.local_path.exists():
            raise FileNotFoundError(f"Missing input file: {spec.local_path}")

    failed = []
    for i, spec in enumerate(specs, 1):
        try:
            logger.info(f"[{i}/{len(specs)}] Processing {spec.repo_id}")
            upload_dataset(
                spec,
                args.token,
                args.namespace,
                args.private,
                args.max_retries,
                args.retry_wait_seconds,
                args.transport,
            )
        except Exception as e:
            logger.error(f"Failed to upload {spec.repo_id}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(spec.repo_id)

    if failed:
        logger.error(f"\nFailed datasets ({len(failed)}/{len(specs)}): {', '.join(failed)}")
        raise RuntimeError(f"Upload failed for: {', '.join(failed)}")
    else:
        logger.info(f"\n✓ All {len(specs)} datasets uploaded successfully!")


if __name__ == "__main__":
    main()
