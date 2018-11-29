# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

func toHex*(bytes: openarray[byte]): string =
  const hexChars = "0123456789abcdef"

  result = newString(3 * bytes.len)
  for i in 0 ..< bytes.len:
    result[3*i  ] = hexChars[int bytes[i] shr 4 and 0xF]
    result[3*i+1] = hexChars[int bytes[i]       and 0xF]
    result[3*i+2] = ' '
