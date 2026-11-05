import logging
import shutil

from musetric_toolkit.listening.round_files import (
    CLIPS,
    MANIFEST,
    read_json,
    write_json,
)

LOG = logging.getLogger(__name__)


def copy_round_command(args) -> None:
    if args.out.exists():
        raise ValueError(f"{args.out} exists: a check round starts from nothing")
    if args.out.name == args.round.name:
        raise ValueError("a check round must not carry the name of the round itself")
    manifest = read_json(args.round / MANIFEST)
    manifest["clips"] = manifest["clips"][: args.clips]
    manifest["round"] = args.out.name
    (args.out / CLIPS).mkdir(parents=True)
    for clip in manifest["clips"]:
        name = clip["id"]
        shutil.copytree(args.round / CLIPS / name, args.out / CLIPS / name)
    write_json(args.out / MANIFEST, manifest)
    LOG.info("made %s", args.out)
