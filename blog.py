import argparse
import os
import shutil
import datetime
from typing import cast
from pathlib import Path
from dataclasses import dataclass

EDITORS = ["nvim", "vim", "emacs", "vi", "nano"]


def template(title: str):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    meta = f"""---
title: '{title}'
author: ''
layout: ../../layouts/post.astro
date: {date}
---"""
    return meta


@dataclass
class Args(argparse.Namespace):
    action: str
    target: str


def build_parser():
    parser = argparse.ArgumentParser(
        prog="Markdown helper",
        description="Make things easy to add/edit/delete markdowns",
    )

    _ = parser.add_argument("action")
    _ = parser.add_argument("target", nargs="?")
    args: Args = cast(Args, parser.parse_args())
    return args


def get_editor():
    if "EDITOR" in os.environ:
        editor = os.environ["EDITOR"]
        return editor
    else:
        for editor in EDITORS:
            print(f"Checking for editor: {editor}, {shutil.which(editor)}")
            if shutil.which(editor) is not None:
                print(f"Using editor: {editor}")
                return editor

        raise OSError("No editor found in $EDITOR or in the system path.")


def open_file(path: Path):
    try:
        editor = get_editor()
        os.execvp(editor, [editor, path.as_posix()])
    except OSError as e:
        print(e)
        os._exit(1)


def new(args: Args):
    if args.target and args.target != "":
        # remove any special symbols from the target
        target = "".join(
            c for c in args.target if c.isalnum() or c.isspace() or c == "_" or c == "-"
        )

        # lowercase the target and repalce spaces with underscores
        filename: str = target.lower().replace(" ", "_") + ".md"

        path = Path("src") / "pages" / "posts" / filename

        with open(path, "w") as f:
            _ = f.write(template(args.target))

        open_file(path)

    else:
        print("Invalid title!!")


def edit():
    files = ls()
    i = input("Index to edit: ")
    open_file(files[int(i)])


def move_to_trash_folder(path: Path):
    # check if the trash folder exists, if not create it
    if not (Path(".trash").exists()):
        Path(".trash").mkdir()

    _ = shutil.move(path, Path(".trash") / path.name)


def rm(trash: bool = True):
    files = ls()
    i = input("Index to remove: ")

    if trash:
        # move to trash
        move_to_trash_folder(files[int(i)])
        print(f"Moved {files[int(i)].name} to .trash.")
        return
    else:
        os.remove(files[int(i)])


def ls():
    path = Path("src") / "pages" / "posts"
    files = list(path.iterdir())
    print()
    print("=" * 10)
    for i, file in enumerate(files):
        print(f"{i}. {file.name}")
    print("=" * 10)
    print()
    return files


def main():
    args = build_parser()
    if args.action == "new":
        new(args)

    elif args.action == "edit":
        edit()

    elif args.action == "rm":
        rm()

    elif args.action == "ls":
        _ = ls()

    else:
        print(" Invalid action!!")


if __name__ == "__main__":
    main()
