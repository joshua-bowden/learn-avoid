"""Copy only successful episodes from augmented HDF5 + results.csv into a new HDF5."""
from __future__ import annotations

import csv
import dataclasses
import pathlib

import h5py
import tyro


@dataclasses.dataclass
class Args:
    hdf5: str = "data/research/mjpl/obstacle_demo.hdf5"
    csv: str = "data/research/mjpl/results.csv"
    output: str = "data/research/mjpl/obstacle_demo_success.hdf5"


def main(cfg: Args) -> None:
    src = pathlib.Path(cfg.hdf5)
    out = pathlib.Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    success_eps = []
    with open(cfg.csv) as fh:
        for row in csv.DictReader(fh):
            if row["success"] == "1":
                success_eps.append(int(row["episode"]))
    if not success_eps:
        raise SystemExit("no successful episodes in csv")

    total = 0
    with h5py.File(src, "r") as fin, h5py.File(out, "w") as fout:
        grp = fout.create_group("data")
        for k, v in fin["data"].attrs.items():
            grp.attrs[k] = v
        for i, ep in enumerate(success_eps):
            name = f"demo_{i}"
            fin.copy(f"data/demo_{ep}", grp, name=name)
            total += int(fin[f"data/demo_{ep}"].attrs.get("num_samples", 0))
        grp.attrs["num_demos"] = len(success_eps)
        grp.attrs["total"] = total

    print(f"wrote {out}  demos={len(success_eps)}  episodes={success_eps}  samples={total}")


if __name__ == "__main__":
    main(tyro.cli(Args))
