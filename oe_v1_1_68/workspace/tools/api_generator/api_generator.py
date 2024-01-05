import argparse
import os
import pkgutil
from importlib import import_module

import yaml
from easydict import EasyDict as edict
from termcolor import cprint

from hat.registry import OBJECT_REGISTRY

__all__ = [
    "api_generator",
]

title_list = ["=", "-", "^", "*"]
max_depth = 4


def add_title_to_write_msg(write_msg, this_title, title_msg):
    write_msg.append(this_title + "\n")
    write_msg.append("".join(title_msg * len(this_title)) + "\n")
    write_msg.append("\n")
    return write_msg


def add_to_write_msg(
    write_msg,
    currentmodule,
    keep_in_all,
    need_write_title=False,
    ignore_keys=None,
):
    module_str1 = ".. currentmodule:: "
    module_str2 = ".. autosummary::"
    module_str3 = "    :nosignatures:"

    all_module = currentmodule.split(".")
    parent_module = "hat.{}".format(".".join(all_module[:-1]))
    child_module = all_module[-1]

    module_titles = []
    if need_write_title:
        if len(all_module) > 2:
            title_pro = len(all_module) - 1
            title_pro = min(len(title_list) - 1, title_pro)
            module_titles = add_title_to_write_msg(
                module_titles, all_module[-2], title_list[title_pro]
            )
        module_titles.append(module_str1 + parent_module + "\n")
        module_titles.append("\n")
        module_titles.append(module_str2 + "\n")
        module_titles.append(module_str3 + "\n")
        module_titles.append("\n")

    values_mm = []
    for module in keep_in_all:
        if f"hat.{currentmodule}" in ignore_keys:
            if module in ignore_keys[f"hat.{currentmodule}"]:
                continue

        values_mm.append(f"    {child_module}.{module}\n")
        values_mm.append("\n")
        need_write_title = False

    if not need_write_title:
        write_msg.extend(module_titles)
        write_msg.extend(values_mm)

    return write_msg, need_write_title


def add_to_write_api(write_api, currentmodule, values, ignore_keys):

    api_str1 = ".. automodule:: "
    api_str2 = "    :members:"
    api_title = api_str1 + f"hat.{currentmodule}" + "\n"

    api_members_num = 0

    for value in values:
        if f"hat.{currentmodule}" in ignore_keys:
            if value in ignore_keys[f"hat.{currentmodule}"]:
                continue
        api_str2 = f"{api_str2} {value},"
        api_members_num = api_members_num + 1

    if api_members_num:
        write_api.append(api_title)
        write_api.append(api_str2[:-1] + "\n")
        write_api.append("\n")

    return write_api


def get_all_sub_values(values):

    new_values = []
    sub_values = []
    for key, value in values.items():
        if isinstance(value, dict):
            sub_value_tmp = get_all_sub_values(value)
            for _sub_value_tmp in sub_value_tmp:
                sub_values.append(f"{key}.{_sub_value_tmp}")
        else:
            for value_tmp in value:
                new_values.append(f"{key}.{value_tmp}")
    if sub_values:
        new_values.extend(sub_values)
    return new_values


def get_write_mgs(module_info, parent_key, write_mgs, write_api, ignore_keys):
    sub_write_mgs = []
    sub_write_apis = []

    need_write_title = True

    for key, values in module_info.items():
        new_key = "{}.{}".format(parent_key, key)
        if isinstance(values, dict):
            if len(new_key.split(".")) >= max_depth:
                new_values = get_all_sub_values(values)

                write_mgs, need_write_title = add_to_write_msg(
                    write_mgs,
                    new_key,
                    new_values,
                    need_write_title,
                    ignore_keys,
                )

                sub_api = []
                tmp_sub = []
                _, sub_api = get_write_mgs(
                    values, new_key, tmp_sub, sub_api, ignore_keys
                )

                sub_write_apis.append(sub_api)
            else:
                sub = []
                sub_api = []
                sub, sub_api = get_write_mgs(
                    values, new_key, sub, sub_api, ignore_keys
                )
                sub_write_mgs.append(sub)
                sub_write_apis.append(sub_api)
        else:
            write_mgs, need_write_title = add_to_write_msg(
                write_mgs, new_key, values, need_write_title, ignore_keys
            )
            write_api = add_to_write_api(
                write_api, new_key, values, ignore_keys
            )

    for sub_write_ in sub_write_mgs:
        write_mgs = write_mgs + sub_write_

    for sub_write_api in sub_write_apis:
        write_api = write_api + sub_write_api
    return write_mgs, write_api


def api_generator(
    target_dir,
    module_name,
    all_module_info,
    docstring=None,
    ignore=None,
):
    write_msg = []

    # write title
    title = module_name
    write_msg.append(title + "\n")
    write_msg.append("".join(title_list[0] * len(title)) + "\n")
    write_msg.append("\n")

    if docstring is not None:
        write_msg.append(docstring)
        write_msg.append("\n")
        write_msg.append("\n")

    write_msg.append(title + "\n")
    write_msg.append("".join(title_list[1] * len(title)) + "\n")
    write_msg.append("\n")

    write_api = []
    api_title = "API Reference\n"
    write_api.append(api_title)
    write_api.append("".join(title_list[1] * len(api_title)) + "\n")
    write_api.append("\n")

    ignore_keys = []
    if ignore is not None:
        ignore_keys = ignore

    # write module
    write_msg, write_api = get_write_mgs(
        all_module_info[module_name],
        module_name,
        write_msg,
        write_api,
        ignore_keys,
    )

    write_msg = write_msg + write_api
    output_file_name = os.path.join(target_dir, module_name + ".rst")

    with open(output_file_name, "w") as f:
        for line in write_msg:
            f.write(line)
    cprint(f"[docs] create {output_file_name}", "green")


def main(args):

    file_list = edict(
        yaml.load(open(args.api_module_list, "r"), Loader=yaml.SafeLoader)
    )

    if os.path.exists(args.target_dir):
        if args.override:
            cprint(f"Override {args.target_dir} with clean", "red")
            os.system(f"rm -rf {args.target_dir}")
            os.mkdir(args.target_dir)
        else:
            cprint(
                f"{args.target_dir} already exists,"
                "if your want to override with clean, please use --override",
                "red",
            )
            return
    else:
        os.mkdir(args.target_dir)

    for _, module_name, _ in pkgutil.walk_packages(
        [os.path.join(args.root, "hat")], prefix="hat."
    ):
        try:
            import_module(module_name)
        except Exception:
            continue

    all_module_info = {}

    for name in OBJECT_REGISTRY.keys():
        module = OBJECT_REGISTRY.get(name)
        parent_module = module.__module__

        module_path = parent_module.split(".")

        tmp_dict = all_module_info

        for idx, submodule in enumerate(module_path):
            if idx == len(module_path) - 1:
                tmp_dict.setdefault(submodule, [])
                if isinstance(name, str):
                    tmp_dict[submodule].append(name)
            else:
                tmp_dict.setdefault(submodule, {})
            tmp_dict = tmp_dict[submodule]

    hat_all_module = all_module_info["hat"]

    for module_name, module_keys in file_list.items():
        if module_name in hat_all_module:
            docstring = None
            ignore = None
            if "docstring" in module_keys:
                docstring = module_keys["docstring"][0]
            if "ignore" in module_keys:
                ignore = module_keys["ignore"]
            api_generator(
                args.target_dir, module_name, hat_all_module, docstring, ignore
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-module-list",
        type=str,
        required=True,
        help="The api module and docstring file.",
    )
    parser.add_argument(
        "--root", type=str, default="../../", help="The root dir of HAT."
    )
    parser.add_argument(
        "--target-dir", type=str, default="../../docs/source/api_reference"
    )
    parser.add_argument("--override", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
