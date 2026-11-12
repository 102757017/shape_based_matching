#!/usr/bin/env python3
"""check_abi.py - inspect shape_based_matching_c.dll and verify ABI contract.

A reusable PE inspector plus deterministic guardrail checks. The freshly built
C API DLL is consumed by an external C# project through P/Invoke, so it must keep
a contract that this script verifies:

  1. Architecture must be x64 (the C# consumer runs 64-bit, AnyCPU + Prefer32Bit=false).
  2. The OpenCV world DLL it links (opencv_world4XXX.dll) must exist in the
     consumer's libs folder (SBM_CONSUMER_LIBS). A mismatch means the C# app
     fails at load with DllNotFoundException / "找不到指定的模块".
  3. Optional: every P/Invoke method name declared in MatcherNative.cs
     (SBM_CONSUMER_CS) must appear in the DLL's export table, otherwise the C#
     call throws EntryPointNotFoundException at runtime.

Note: this script only reads files; it never writes or executes the DLL.

Usage:
  python check_abi.py --dll <path> [--libs <dir>] [--cs <file>]
"""
import argparse
import os
import re
import struct
import sys


def get_optional_offset(data):
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    coff = e_lfanew + 4
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt_off = coff + 20
    magic = struct.unpack_from("<H", data, opt_off)[0]
    is64 = magic == 0x20B
    dd_off = opt_off + (112 if is64 else 96)
    return is64, opt_off, opt_size, dd_off


def parse_sections(data, opt_off, opt_size):
    coff = opt_off - 20
    nsec = struct.unpack_from("<H", data, coff + 2)[0]
    sec_off = opt_off + opt_size
    sections = []
    for i in range(nsec):
        b = sec_off + i * 40
        vsize = struct.unpack_from("<I", data, b + 8)[0]
        va = struct.unpack_from("<I", data, b + 12)[0]
        rsize = struct.unpack_from("<I", data, b + 16)[0]
        raw = struct.unpack_from("<I", data, b + 20)[0]
        sections.append((va, raw, max(vsize, rsize)))
    return sections


def rva_to_raw(rva, sections):
    for va, raw, size in sections:
        if va <= rva < va + size:
            return raw + (rva - va)
    return None


def get_architecture(data):
    is64, opt_off, _, _ = get_optional_offset(data)
    return "x64" if is64 else "x86"


def get_import_dlls(data):
    is64, opt_off, opt_size, dd_off = get_optional_offset(data)
    sections = parse_sections(data, opt_off, opt_size)
    import_rva = struct.unpack_from("<I", data, dd_off + 8)[0]
    import_size = struct.unpack_from("<I", data, dd_off + 12)[0]
    if import_rva == 0 or import_size == 0:
        return []
    raw = rva_to_raw(import_rva, sections)
    if raw is None:
        return []
    dlls = []
    off = raw
    end = raw + import_size
    while off + 20 <= end:
        ilt, _ts, _fc, name_rva, _ft = struct.unpack_from("<5I", data, off)
        if ilt == 0 and name_rva == 0 and _ft == 0:
            break
        if name_rva != 0:
            nr = rva_to_raw(name_rva, sections)
            if nr is not None:
                win = data.find(b"\x00", nr, min(nr + 260, len(data)))
                name = data[nr:win].decode("ascii", "replace") if win >= 0 else ""
                if name:
                    dlls.append(name)
        off += 20
    return dlls


def get_export_names(data):
    is64, opt_off, opt_size, dd_off = get_optional_offset(data)
    sections = parse_sections(data, opt_off, opt_size)
    export_rva = struct.unpack_from("<I", data, dd_off)[0]
    if export_rva == 0:
        return []
    raw = rva_to_raw(export_rva, sections)
    if raw is None:
        return []
    num_names = struct.unpack_from("<I", data, raw + 24)[0]
    addr_of_names = struct.unpack_from("<I", data, raw + 32)[0]
    names_raw = rva_to_raw(addr_of_names, sections)
    if names_raw is None:
        return []
    names = []
    for i in range(num_names):
        nr = struct.unpack_from("<I", data, names_raw + i * 4)[0]
        sr = rva_to_raw(nr, sections)
        if sr is None:
            continue
        win = data.find(b"\x00", sr, min(sr + 260, len(data)))
        name = data[sr:win].decode("ascii", "replace") if win >= 0 else ""
        if name:
            names.append(name)
    return names


def normalize(name):
    n = name
    if n.startswith("_"):
        n = n[1:]
    n = re.sub(r"@\d+$", "", n)
    return n


def extract_csharp_names(cs_path):
    with open(cs_path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    pat = re.compile(
        r"(?i)DllImport\s*\(([^)]*)\)[^\w]*"
        r"(?:public\s+|internal\s+|private\s+|protected\s+|static\s+|extern\s+|unsafe\s+)*"
        r"[A-Za-z_][\w.<>\[\]\s,]*?\s+([A-Za-z_]\w*)\s*\("
    )
    names = set()
    for m in pat.finditer(text):
        attr = m.group(1)
        method = m.group(2)
        ep = re.search(r'(?i)EntryPoint\s*=\s*"([^"]+)"', attr)
        name = ep.group(1) if ep else method
        names.add(normalize(name))
    return names


def main():
    ap = argparse.ArgumentParser(description="Verify the C API DLL ABI contract.")
    ap.add_argument("--dll", required=True, help="path to shape_based_matching_c.dll")
    ap.add_argument("--libs", default="", help="consumer libs dir (SBM_CONSUMER_LIBS)")
    ap.add_argument("--cs", default="", help="consumer MatcherNative.cs (SBM_CONSUMER_CS)")
    args = ap.parse_args()

    fails = []
    warns = []

    if not os.path.isfile(args.dll):
        print("[FAIL] DLL not found: %s" % args.dll)
        return 1
    with open(args.dll, "rb") as f:
        data = f.read()

    arch = get_architecture(data)
    print("[INFO] architecture: %s" % arch)
    if arch != "x64":
        fails.append("architecture is %s, expected x64 (C# consumer runs 64-bit)" % arch)

    imports = get_import_dlls(data)
    ocv = [d for d in imports if re.search(r"opencv_world\d+\.dll", d, re.IGNORECASE)]
    if not ocv:
        warns.append("no opencv_world4XXX.dll import found")
        linked_ver = None
    else:
        linked_ver = re.search(r"opencv_world(\d+)\.dll", ocv[0], re.IGNORECASE).group(1)
        print("[INFO] links OpenCV runtime: %s" % ocv[0])

    if args.libs:
        if not os.path.isdir(args.libs):
            warns.append("consumer libs dir not found: %s" % args.libs)
        elif linked_ver is not None:
            present = [f for f in os.listdir(args.libs)
                       if re.search(r"opencv_world\d+\.dll", f, re.IGNORECASE)]
            target = "opencv_world%s.dll" % linked_ver
            if target not in [p.lower() for p in present]:
                fails.append(
                    "DLL links %s but consumer libs only has: %s"
                    % (target, ", ".join(present) if present else "(none)"))
            else:
                print("[OK]   OpenCV runtime %s present in consumer libs" % target)
    else:
        warns.append("SBM_CONSUMER_LIBS not set - skipped OpenCV runtime presence check")

    if args.cs:
        if not os.path.isfile(args.cs):
            warns.append("consumer C# file not found: %s" % args.cs)
        else:
            cs_names = extract_csharp_names(args.cs)
            exports = set(normalize(n) for n in get_export_names(data))
            missing = sorted(n for n in cs_names if normalize(n) not in exports)
            print("[INFO] C# P/Invoke methods: %d, DLL exports: %d"
                  % (len(cs_names), len(exports)))
            if missing:
                fails.append(
                    "C# declares P/Invoke names with no matching export "
                    "(EntryPointNotFoundException risk): %s" % ", ".join(missing))
            else:
                print("[OK]   all C# P/Invoke names resolve to DLL exports")
    else:
        warns.append("SBM_CONSUMER_CS not set - skipped export/DllImport diff")

    print("")
    for w in warns:
        print("[WARN] %s" % w)
    for fl in fails:
        print("[FAIL] %s" % fl)

    if fails:
        print("\n[RESULT] FAIL - %d problem(s) found" % len(fails))
        return 1
    print("[RESULT] PASS - ABI contract checks cleared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
