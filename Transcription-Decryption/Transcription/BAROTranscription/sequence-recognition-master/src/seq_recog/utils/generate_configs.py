"""Model configuration generator.

Generate configurations for grid/random search automatically from a
special configuration file.
"""

from __future__ import annotations

import json
from math import log10
from random import choice, uniform

from argparse import ArgumentParser, Namespace
from copy import deepcopy

# from inspect import isfunction
from lark import Lark, Transformer, v_args
from typing import Dict, List
from pathlib import Path


LARK_GRAMMAR = r"""
?start: value

?value: object
      | array
      | string
      | SIGNED_FLOAT       -> nfloat
      | SIGNED_INT         -> nint
      | "true"             -> true
      | "false"            -> false
      | "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" INSTRUCTION? value

string : ESCAPED_STRING
INSTRUCTION: "anyof"
           | "range"
           | "logrange"

%import common.ESCAPED_STRING
%import common.SIGNED_FLOAT
%import common.SIGNED_INT
%import common.WS

%ignore WS
"""


class GeneratorFactory:
    @classmethod
    def create_anyof(cls, values):
        def anyof(values=values):
            return choice(values)

        return anyof

    @classmethod
    def create_range(cls, values):
        def rrange(values=values):
            return uniform(*values)

        return rrange

    @classmethod
    def create_logrange(cls, values):
        def lrange(values=values):
            return 10 ** (uniform(*map(log10, values)))

        return lrange


class TJSONTransformer(Transformer):
    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')

    array = list
    object = dict
    nfloat = v_args(inline=True)(float)
    nint = v_args(inline=True)(int)

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False

    instruction = lambda self, x: x.value

    def pair(self, children):
        if len(children) == 2:
            name = children[0]
            instruction = None
            value = children[1]
        else:
            name = children[0]
            instruction = children[1]
            value = children[2]

        # return children
        if instruction is not None:
            assert type(value) is list
            if instruction == "anyof":
                return (name, GeneratorFactory.create_anyof(value))
            elif instruction == "range":
                return (name, GeneratorFactory.create_range(value))
            elif instruction == "logrange":
                return (name, GeneratorFactory.create_logrange(value))
            else:
                raise NotImplementedError
        else:
            return (name, value)


class TemplateParser:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._parser = Lark(LARK_GRAMMAR)
        self._tformer = TJSONTransformer()

        with open(self._path, "r") as f_template:
            self._model = f_template.read()
            self._model = self._parser.parse(self._model)
            self._model = self._tformer.transform(self._model)

    def generate_config(self, name: str) -> Dict:
        baseconfig = deepcopy(self._model)
        newconfig = self.run_tree(baseconfig)
        newconfig["exp_name"] = name

        return newconfig

    def run_tree(self, base_object: Dict) -> Dict:
        for k, v in base_object.items():
            if callable(v):
                base_object[k] = v()
            if type(v) == dict:
                base_object[k] = self.run_tree(v)
        return base_object


def setup() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "template_file",
        type=str,
        help="Config template file to employ",
    )
    parser.add_argument(
        "nfiles",
        type=int,
        help="Number of files to generate",
    )
    parser.add_argument("exp_name", type=str, help="How to name generated experiments")
    parser.add_argument(
        "output_path", type=str, help="What folder to store the generated configs into"
    )

    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    temp_parser = TemplateParser(args.template_file)
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    for ii in range(args.nfiles):
        name = args.exp_name + f"{ii:03d}"
        config = temp_parser.generate_config(name)

        with open(output_path / (name + ".json"), "w") as f_json:
            json.dump(config, f_json, indent=4)


if __name__ == "__main__":
    main(setup())
