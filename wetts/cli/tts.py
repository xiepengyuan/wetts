# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from wetts.cli.hub import Hub
from wetts.cli.frontend import Frontend


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('text', help='text to synthesis')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    front_dir = Hub.get_model("frontend")
    frontend = Frontend(front_dir)
    print(frontend.compute(args.text))


if __name__ == '__main__':
    main()
