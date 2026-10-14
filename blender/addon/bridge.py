"""Blocking binary calls to one resident CUDA process."""
from pathlib import Path
import json
import struct
import subprocess
import time

import numpy as np
import openvdb

from . import inputs, process
from .settings import CONFIG


HEADER = struct.Struct("<iidfffiifQfiidi3f4i")
SPARSE_BLOCK = np.dtype([("coordinate", "<i4", (3,)), ("level", "<i4"), ("quiet", "<i4"), ("remaining", "<i4"), ("density", "<f4", (512,))])
GRID_BLOCK = np.dtype([("coordinate", "<i4", (3,)), ("level", "<i4"), ("quiet", "<i4"), ("remaining", "<i4")])


def acknowledge(session):
    solver = session["solver"]
    solver.stdin.flush()
    if solver.stdout.read(1) != b"K":
        raise RuntimeError((session["directory"] / "solver.log").read_text(encoding="utf-8", errors="replace") or "Solver pipe closed")


def start(scene, objects, geometry, corners, dynamic, directory):
    with (directory / "solver.log").open("wb") as log:
        solver = subprocess.Popen([str(Path(__file__).parent / "bin" / "pyro.exe"), "--stream"], cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        job = process.attach(solver.pid)
    except Exception:
        solver.terminate()
        solver.wait()
        raise
    session = {"solver": solver, "job": job, "algorithm": scene.pyro.algorithm, "directory": directory, "objects": objects, "corners": corners, "dynamic": dynamic, "current": scene.frame_start, "samples": {}, "fps": scene.render.fps / scene.render.fps_base, "start": scene.frame_start}
    try:
        config = CONFIG[session["algorithm"]]
        values = []
        for name, kind in config:
            value = getattr(scene.pyro, name)
            values.extend(value if kind in {"3f", "3i"} else [value])
        solver.stdin.write(b"PYROSTR2" + {"SPARSE": b"S", "DENSE": b"D"}[session["algorithm"]])
        solver.stdin.write(struct.pack("<" + "".join(kind for _, kind in config), *values))
        solver.stdin.write(struct.pack("<i", len(objects)))
        for triangles in geometry:
            solver.stdin.write(struct.pack("<i", len(triangles)) + triangles.tobytes())
        acknowledge(session)
    except Exception:
        close(session)
        raise
    return session


def advance(scene, session):
    current, start, fps = session["current"], session["start"], session["fps"]
    samples = session["samples"]
    segments = [(float(current), float(current + 1))]
    while segments:
        a, b = segments.pop()
        middle = (a + b) * 0.5
        for frame in (a, middle, b):
            if frame not in samples:
                samples[frame] = inputs.sample(scene, session["objects"], frame) if session["dynamic"] or not samples else next(iter(samples.values()))
        first, mid, last = samples[a], samples[middle], samples[b]
        error = max((np.linalg.norm((mid[i][0] - (first[i][0] + last[i][0]) * 0.5) @ session["corners"][i], axis=0).max() for i in range(len(first))), default=0)
        if error > scene.pyro.cell_size * scene.pyro.animation_tolerance:
            segments.extend([(middle, b), (a, middle)])
            continue
        sources, colliders = [], []
        for i, obj in enumerate(session["objects"]):
            _, active, values = mid[i]
            if not active:
                continue
            derivative = (last[i][0] - first[i][0]) * (fps / (b - a))
            points = np.concatenate((first[i][0] @ session["corners"][i], last[i][0] @ session["corners"][i]), axis=1)
            speed = np.linalg.norm(derivative @ session["corners"][i], axis=0).max()
            lower, upper = points.min(axis=1), points.max(axis=1)
            if obj.pyro.role == "COLLIDER":
                surface = values[0]
                lower, upper = lower - abs(surface), upper + abs(surface)
            motion = struct.pack("<i32f", i, *first[i][0].flat, *derivative.flat, (a-start)/fps, speed, *lower, *upper)
            if obj.pyro.role == "EMITTER":
                sources.append(motion + struct.pack("<7f", *values))
            else:
                colliders.append(motion + struct.pack("<f", surface))
        session["solver"].stdin.write(b"A" + struct.pack("<dii", (b-start)/fps, len(sources), len(colliders)) + b"".join(sources + colliders))
        acknowledge(session)
    session["current"] = current + 1
    session["samples"] = {float(current + 1): samples[float(current + 1)]}


def export(session, write_volume):
    solver = session["solver"]
    solver.stdin.write(b"F" + struct.pack("<i?", session["current"], write_volume))
    if not write_volume:
        acknowledge(session)
        return
    solver.stdin.flush()
    packed = solver.stdout.read(HEADER.size)
    if len(packed) != HEADER.size:
        raise RuntimeError((session["directory"] / "solver.log").read_text(encoding="utf-8", errors="replace") or "Incomplete frame header")
    frame, count, seconds, spacing, step_ms, export_ms, blocks, cells, divergence, allocated, speed, substeps, iterations, smoke, capacity, ox, oy, oz, nx, ny, nz, levels = HEADER.unpack(packed)
    algorithm = session["algorithm"]
    topology = np.zeros(blocks, dtype=GRID_BLOCK)
    if algorithm == "DENSE":
        density = np.frombuffer(solver.stdout.read(count * 4), dtype="<f4", count=count)
    else:
        records = np.frombuffer(solver.stdout.read(count * SPARSE_BLOCK.itemsize), dtype=SPARSE_BLOCK, count=count)
        for name in topology.dtype.names:
            topology[name] = records[name]
    begin = time.perf_counter()
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    grid.transform = openvdb.createLinearTransform([[spacing, 0, 0, 0], [0, spacing, 0, 0], [0, 0, spacing, 0], [ox+spacing/2, oy+spacing/2, oz+spacing/2, 1]])
    if algorithm == "DENSE":
        grid.copyFromArray(np.ascontiguousarray(density.reshape(nz, ny, nx).transpose(2, 1, 0)))
    else:
        for record in records:
            values = record["density"]
            if not np.any(values):
                continue
            level = int(record["level"])
            scale = 1 << level
            origin = record["coordinate"].astype(np.int64) * (8 * scale)
            if level == 0:
                grid.copyFromArray(np.ascontiguousarray(values.reshape(8, 8, 8).transpose(2, 1, 0)), ijk=tuple(origin))
            elif np.all(values == values[0]):
                grid.fill(tuple(origin), tuple(origin + 8*scale - 1), float(values[0]))
            else:
                # Constant cell averages occupy exactly their physical volume. OpenVDB
                # fills/prunes tiles directly, without a dense finest-resolution domain.
                for local in np.flatnonzero(values):
                    lower = origin + np.array((local % 8, local // 8 % 8, local // 64)) * scale
                    grid.fill(tuple(lower), tuple(lower + scale - 1), float(values[local]))
    grid.prune()
    destination = session["directory"] / f"density_{frame-session['start']+1:06d}.vdb"
    temporary = destination.with_suffix(".tmp.vdb")
    openvdb.write(str(temporary), grids=[grid], metadata={"simulation_time": seconds, "solver_algorithm": algorithm})
    temporary.replace(destination)
    stats = {"algorithm": algorithm, "frame": frame, "time": seconds, "step_ms": step_ms, "export_ms": export_ms, "convert_ms": (time.perf_counter()-begin)*1000, "blocks": blocks, "cells": cells, "divergence": divergence, "bytes": allocated, "max_speed": speed, "substeps": substeps, "pressure_iterations": iterations, "smoke_integral": smoke, "spacing": spacing, "origin": [ox, oy, oz], "resolution": [nx, ny, nz], "capacity": capacity, "levels": levels, "level_blocks": np.bincount(topology["level"], minlength=levels).tolist()}
    destination = session["directory"] / f"grid_{frame-session['start']+1:06d}.npz"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(output, stats=json.dumps(stats, separators=(",", ":")), blocks=topology)
    temporary.replace(destination)


def checkpoint(session, restore=False):
    name = b"state.bin" if restore else b"state.tmp"
    session["solver"].stdin.write((b"L" if restore else b"S") + struct.pack("<I", len(name)) + name)
    acknowledge(session)
    if restore:
        seconds, = struct.unpack("<d", session["solver"].stdout.read(8))
        session["current"] = session["start"] + round(seconds * session["fps"])
        session["samples"].clear()
    else:
        (session["directory"] / "state.tmp").replace(session["directory"] / "state.bin")


def close(session):
    process.kernel.CloseHandle(session["job"])
    solver = session["solver"]
    solver.wait()
    solver.stdin.close()
    solver.stdout.close()
