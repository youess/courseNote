#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import re
import sys
import zipfile


zip_archive_path = sys.argv[1]
archive_extract_path = sys.argv[2]
extract_file_pattern = re.compile('.*\\.mat|.*\\.pdf')


# check if the data directory existed
if not os.path.exists(archive_extract_path):
    os.mkdir(archive_extract_path)

with zipfile.ZipFile(zip_archive_path, 'r') as zfile:
    fp = filter(lambda x: extract_file_pattern.match(x), zfile.namelist())
    for f in fp:
        dest = os.path.join(
            archive_extract_path,
            os.path.basename(f)
        )
        unpacked = open(dest, 'wb')
        unpacked.write(zfile.read(f))
        unpacked.close()
