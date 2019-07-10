# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

iterator reverse*[T](a: seq[T]): T =
    var i = len(a) - 1
    while i > -1:
        yield a[i]
        dec(i)
