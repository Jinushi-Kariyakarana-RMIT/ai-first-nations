from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image


def _validate_tiling(tile_size: int, overlap: int) -> None:
    if tile_size <= 0:
        raise ValueError("tile_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")


def _axis_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def _open_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _has_content(
    tile: Image.Image,
    valid_size: tuple[int, int],
    min_content_fraction: float,
    blank_threshold: int,
) -> bool:
    valid_width, valid_height = valid_size
    valid_tile = tile.crop((0, 0, valid_width, valid_height))
    arr = np.asarray(valid_tile)

    if arr.size == 0:
        return False

    non_blank = np.any(arr > blank_threshold, axis=2)
    return float(non_blank.mean()) >= min_content_fraction


def tile_image(
    image: Image.Image | str | Path,
    tile_size: int = 512,
    overlap: int = 64,
    pad: bool = True,
    skip_empty: bool = False,
    min_content_fraction: float = 0.01,
    blank_threshold: int = 5,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]], tuple[int, int]]:
    """
    Split an image into fixed-size overlapping tiles.

    Returns PIL tiles, their original-image positions, and the original image
    size. Edge tiles are included; when pad=True, smaller edge tiles are padded
    to tile_size so downstream models receive consistent input dimensions.
    """
    _validate_tiling(tile_size, overlap)

    img = _open_image(image)
    width, height = img.size
    step = tile_size - overlap

    tiles = []
    positions = []

    for top in _axis_starts(height, tile_size, step):
        for left in _axis_starts(width, tile_size, step):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            valid_width = right - left
            valid_height = bottom - top

            tile = img.crop((left, top, right, bottom))

            if pad and tile.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            if skip_empty and not _has_content(
                tile,
                (valid_width, valid_height),
                min_content_fraction,
                blank_threshold,
            ):
                continue

            tiles.append(tile)
            positions.append((left, top, right, bottom))

    return tiles, positions, img.size


def tile_image_for_model(
    image: Image.Image | str | Path,
    transform: Callable[[Image.Image], torch.Tensor],
    tile_size: int = 512,
    overlap: int = 64,
    pad: bool = True,
    skip_empty: bool = False,
) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    """Return transformed image tiles stacked as a tensor batch."""
    tiles, positions, _ = tile_image(
        image=image,
        tile_size=tile_size,
        overlap=overlap,
        pad=pad,
        skip_empty=skip_empty,
    )

    if not tiles:
        return torch.empty(0), positions

    return torch.stack([transform(tile) for tile in tiles]), positions
