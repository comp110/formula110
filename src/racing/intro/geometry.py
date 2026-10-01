"""Beveled solid geometry traced from the project's F110 mark."""

from __future__ import annotations

import json
from importlib import import_module
from math import hypot
from pathlib import Path
from typing import Any

from racing.graphics.mesh_utils import mesh_from_quads

Point = tuple[float, float]


def area(points: list[Point]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=True)) / 2


def inset(points: list[Point], distance: float) -> list[Point]:
    """Offset toward the left of an oriented contour, with limited miters."""
    result: list[Point] = []
    for i, point in enumerate(points):
        prev, following = points[i - 1], points[(i + 1) % len(points)]
        ax, ay = point[0] - prev[0], point[1] - prev[1]
        bx, by = following[0] - point[0], following[1] - point[1]
        al, bl = hypot(ax, ay), hypot(bx, by)
        na, nb = (-ay / al, ax / al), (-by / bl, bx / bl)
        factor = distance / max(0.35, 1 + na[0] * nb[0] + na[1] * nb[1])
        result.append((point[0] + (na[0] + nb[0]) * factor, point[1] + (na[1] + nb[1]) * factor))
    return result


def logo_mesh(ursina: Any, width: float = 10.0, depth: float = 0.32, bevel: float = 0.023) -> Any:
    """Extrude five logo islands, preserving the counter in the zero."""
    core = import_module("panda3d.core")
    data = json.loads(Path(__file__).with_name("logo.json").read_text())
    scale = width / 1797
    contours = [[((x - 1023.5) * scale, (222 - y) * scale) for x, y in p] for p in data["contours"]]
    islands = [[p] for p in contours[:5]]
    islands[-1].append(contours[5])
    faces: list[tuple[tuple[float, float, float], ...]] = []
    for rings in islands:
        oriented = [p if (area(p) > 0) == (i == 0) else list(reversed(p)) for i, p in enumerate(rings)]
        inner = [inset(p, bevel) for p in oriented]
        triangulator = core.Triangulator()
        for i, ring in enumerate(inner):
            if i:
                triangulator.beginHole()
            for x, y in ring:
                index = triangulator.addVertex(x, y)
                if i:
                    triangulator.addHoleVertex(index)
                else:
                    triangulator.addPolygonVertex(index)
        triangulator.triangulate()
        for i in range(triangulator.getNumTriangles()):
            points = [
                triangulator.getVertex(getter(i))
                for getter in (
                    triangulator.getTriangleV0,
                    triangulator.getTriangleV1,
                    triangulator.getTriangleV2,
                )
            ]
            for side in (-1, 1):
                face = tuple((float(p[0]), float(p[1]), side * depth / 2) for p in points)
                faces.append(face if side == 1 else tuple(reversed(face)))
        for outer, inside in zip(oriented, inner, strict=True):
            for i, a in enumerate(outer):
                j = (i + 1) % len(outer)
                b, c, d = outer[j], inside[j], inside[i]
                z = depth / 2 - bevel
                faces.append(((a[0], a[1], -z), (b[0], b[1], -z), (b[0], b[1], z), (a[0], a[1], z)))
                for side in (-1, 1):
                    face = (
                        (a[0], a[1], side * z),
                        (b[0], b[1], side * z),
                        (c[0], c[1], side * depth / 2),
                        (d[0], d[1], side * depth / 2),
                    )
                    faces.append(face if side == -1 else tuple(reversed(face)))
    return mesh_from_quads(ursina, tuple(faces))
