from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from phe import paillier

PaillierPublicKey = paillier.PaillierPublicKey
PaillierSecretKey = paillier.PaillierPrivateKey


@dataclass
class PaillierKeyPair:
    public_key: PaillierPublicKey
    private_key: PaillierSecretKey

    def __iter__(self):
        yield self.public_key
        yield self.private_key

    @property
    def secret_key(self):
        return self.private_key


def generate_keypair(n_length: int = 2048):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_length)
    return PaillierKeyPair(public_key, private_key)


def save_public_key(public_key, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"n": public_key.n}, f)


def load_public_key(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return paillier.PaillierPublicKey(n=int(data["n"]))


def save_private_key(private_key, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "public_key": {"n": private_key.public_key.n},
        "p": private_key.p,
        "q": private_key.q,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def load_private_key(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    public_key = paillier.PaillierPublicKey(n=int(data["public_key"]["n"]))
    return paillier.PaillierPrivateKey(
        public_key,
        p=int(data["p"]),
        q=int(data["q"]),
    )


def save_keypair(public_key, private_key, directory: str | Path) -> None:
    directory = Path(directory)
    save_public_key(public_key, directory / "public_key.json")
    save_private_key(private_key, directory / "private_key.json")
