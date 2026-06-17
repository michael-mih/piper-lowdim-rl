from __future__ import annotations

import argparse
import struct
from pathlib import Path


def _resolve_index(raw: str, count: int) -> int | None:
    if not raw:
        return None
    index = int(raw)
    if index > 0:
        return index - 1
    return count + index


def _parse_face_token(token: str, nverts: int, nuvs: int, nnormals: int) -> tuple[int, int | None, int | None]:
    parts = token.split("/")
    vertex = _resolve_index(parts[0], nverts)
    uv = _resolve_index(parts[1], nuvs) if len(parts) > 1 else None
    normal = _resolve_index(parts[2], nnormals) if len(parts) > 2 else None
    if vertex is None:
        raise ValueError(f"Face token {token!r} is missing a vertex index")
    return vertex, uv, normal


def convert_obj_to_msh(obj_path: Path, msh_path: Path) -> None:
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []

    vertices: list[tuple[float, float, float]] = []
    vertex_normals: list[tuple[float, float, float]] = []
    vertex_texcoords: list[tuple[float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_map: dict[tuple[int, int | None, int | None], int] = {}

    def get_vertex(key: tuple[int, int | None, int | None]) -> int:
        if key in vertex_map:
            return vertex_map[key]

        pos_index, tex_index, normal_index = key
        vertices.append(positions[pos_index])
        vertex_texcoords.append(texcoords[tex_index] if tex_index is not None else (0.0, 0.0))
        vertex_normals.append(normals[normal_index] if normal_index is not None else (0.0, 0.0, 0.0))
        vertex_map[key] = len(vertices) - 1
        return vertex_map[key]

    with obj_path.open("r", encoding="utf-8") as obj_file:
        for line in obj_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]
            if tag == "v":
                positions.append(tuple(float(value) for value in parts[1:4]))
            elif tag == "vt":
                texcoords.append(tuple(float(value) for value in parts[1:3]))
            elif tag == "vn":
                normals.append(tuple(float(value) for value in parts[1:4]))
            elif tag == "f":
                face_vertices = [
                    get_vertex(_parse_face_token(token, len(positions), len(texcoords), len(normals)))
                    for token in parts[1:]
                ]
                for index in range(1, len(face_vertices) - 1):
                    faces.append((face_vertices[0], face_vertices[index], face_vertices[index + 1]))

    if len(vertices) < 4:
        raise ValueError("MuJoCo MSH files require at least 4 vertices")
    if not faces:
        raise ValueError("OBJ did not contain any faces")

    msh_path.parent.mkdir(parents=True, exist_ok=True)
    with msh_path.open("wb") as msh_file:
        msh_file.write(struct.pack("<4i", len(vertices), len(vertex_normals), len(vertex_texcoords), len(faces)))
        for vertex in vertices:
            msh_file.write(struct.pack("<3f", *vertex))
        for normal in vertex_normals:
            msh_file.write(struct.pack("<3f", *normal))
        for texcoord in vertex_texcoords:
            msh_file.write(struct.pack("<2f", *texcoord))
        for face in faces:
            msh_file.write(struct.pack("<3i", *face))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a textured OBJ mesh to MuJoCo's binary MSH format.")
    parser.add_argument("obj", type=Path)
    parser.add_argument("msh", type=Path)
    args = parser.parse_args()

    convert_obj_to_msh(args.obj, args.msh)


if __name__ == "__main__":
    main()
