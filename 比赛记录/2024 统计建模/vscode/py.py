v4=[get_wide_byte(0x00401625+i*7) for i in range(24)]
for i in range(0,len(v4) - 1,4):
    v4[i + 2] ^= v4[i+3]
    v4[i + 1] ^= v4[i + 2]
    v4[i] ^= v4[i + 1]
print(bytes(v4).decode())