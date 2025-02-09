# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Makes sure all files have a Meta header.

import os

# Define the header content
header = """# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


def check_and_add_header(file_path):
    """Check if the header is present in the file, and add it if it's not."""
    with open(file_path, "r+", encoding="utf-8") as file:
        content = file.read()
        if not content.startswith(header):
            file.seek(0, 0)
            file.write(header + "\n" + content)


def scan_directory(directory):
    """Scan the directory for Python files and apply the header check."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                check_and_add_header(file_path)


if __name__ == "__main__":
    directory = input("Enter the directory path to scan: ")
    scan_directory(directory)
    print("Header check and addition completed.")
