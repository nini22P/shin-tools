import argparse
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM

# Fixed VM instructions (0x40-0x53)
VM_INSTRUCTIONS = {
    0x40: 'uo',
    0x41: 'bo',
    0x42: 'exp',
    0x43: 'mm',
    0x44: 'gt',
    0x45: 'st',
    0x46: 'jc',
    0x47: 'j',
    0x48: 'gosub',
    0x49: 'retsub',
    0x4A: 'jt',
    0x4B: 'gosubt',
    0x4C: 'rnd',
    0x4D: 'push',
    0x4E: 'pop',
    0x4F: 'call',
    0x50: 'r#return',
    0x51: 'fmt',
    0x52: 'fnmt',
    0x53: 'getbupid',
}


class ElfImage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self._validate_elf()
        self.sections = self._parse_sections()
        if '.text' not in self.sections or '.rela.dyn' not in self.sections:
            raise RuntimeError(f'Missing .text or .rela.dyn section in ELF: {self.path}')
        self.text = self.sections['.text']
        self.text_lo = self.text['addr']
        self.text_hi = self_hi = self.text_lo + self.text['size']
        self.text_bytes = self.data[self.text['offset']:self.text['offset'] + self.text['size']]
        self.rela = self._parse_rela()
        self.md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        self.md.detail = True

    def _validate_elf(self) -> None:
        if len(self.data) < 64 or self.data[:4] != b'\x7fELF':
            raise RuntimeError(f'Not an ELF file (convert NSO with nx2elf first): {self.path}')
        ei_class, ei_data = self.data[4], self.data[5]
        e_machine = struct.unpack_from('<H', self.data, 0x12)[0]
        if ei_class != 2 or ei_data != 1 or e_machine != 183:  # ELF64 LE AArch64
            raise RuntimeError(
                f'Unsupported ELF format (expected 64-bit little-endian AArch64): {self.path}'
            )

    def _parse_sections(self) -> dict[str, dict]:
        e_shoff = struct.unpack_from('<Q', self.data, 0x28)[0]
        e_shnum = struct.unpack_from('<H', self.data, 0x3C)[0]
        e_shstrndx = struct.unpack_from('<H', self.data, 0x3E)[0]
        secs = []
        for i in range(e_shnum):
            off = e_shoff + i * 64
            secs.append({
                'name_off': struct.unpack_from('<I', self.data, off)[0],
                'addr': struct.unpack_from('<Q', self.data, off + 16)[0],
                'offset': struct.unpack_from('<Q', self.data, off + 24)[0],
                'size': struct.unpack_from('<Q', self.data, off + 32)[0],
            })
        s = secs[e_shstrndx]
        strtab = self.data[s['offset']:s['offset'] + s['size']]
        named = {}
        for sec in secs:
            end = strtab.index(b'\0', sec['name_off'])
            sec['name'] = strtab[sec['name_off']:end].decode()
            if sec['name']:
                named[sec['name']] = sec
        return named

    def _parse_rela(self) -> dict[int, int]:
        rela = self.sections['.rela.dyn']
        rdata = self.data[rela['offset']:rela['offset'] + rela['size']]
        entries = {}
        for i in range(rela['size'] // 24):
            off = i * 24
            r_offset, r_info, r_addend = struct.unpack_from('<QQQ', rdata, off)
            if (r_info & 0xFFFFFFFF) == 1027:  # R_AARCH64_RELATIVE
                entries[r_offset] = r_addend
        return entries

    def disasm_func(self, addr: int, max_insns: int = 120):
        off = addr - self.text_lo
        if off < 0 or off >= len(self.text_bytes):
            return []
        insns = []
        max_branch_tgt = addr
        for i in range(max_insns):
            pos = off + i * 4
            if pos + 4 > len(self.text_bytes):
                break
            decoded = list(self.md.disasm(self.text_bytes[pos:pos + 4], addr + i * 4))
            if not decoded:
                break
            ins = decoded[0]
            insns.append(ins)
            if ins.mnemonic.startswith('b.') or ins.mnemonic in ('cbz', 'cbnz', 'tbz', 'tbnz'):
                if ins.operands:
                    tgt = ins.operands[-1].imm
                    if addr <= tgt < addr + max_insns * 4:
                        max_branch_tgt = max(max_branch_tgt, tgt)
            if ins.mnemonic == 'ret' and ins.address >= max_branch_tgt:
                break
            if ins.mnemonic == 'b' and ins.operands:
                tgt = ins.operands[0].imm
                if not (addr <= tgt < addr + max_insns * 4) and ins.address >= max_branch_tgt:
                    break
        return insns

    def find_jump_table(self) -> tuple[int, dict[int, int], int]:
        text_keys = sorted(k for k, v in self.rela.items() if self.text_lo <= v < self.text_hi)
        key_set = set(text_keys)
        for k in text_keys:
            if (k + 255 * 16) not in key_set:
                continue
            if not all((k + i * 16) in key_set for i in range(256)):
                continue
            table = {i: self.rela[k + i * 16] for i in range(256)}
            counts = Counter(table.values())
            stub, stub_cnt = counts.most_common(1)[0]
            # 0x00 (EXIT) and 0xFF (DEBUGOUT) are real handlers; 0x01..0x3F are stubs; 0x40..0x53 are instructions
            if (
                stub_cnt >= 100
                and table[0x00] != stub
                and table[0xFF] != stub
                and all(table[i] == stub for i in range(0x01, 0x40))
                and all(table[i] != stub for i in range(0x40, 0x54))
            ):
                return k, table, stub
        raise RuntimeError(f'Failed to locate opcode jump table: {self.path}')

    def build_adv_rtti_map(self) -> dict[int, str]:
        if '.rodata' not in self.sections:
            return {}
        ro = self.sections['.rodata']
        rodata = self.data[ro['offset']:ro['offset'] + ro['size']]

        str_addr_to_name: dict[int, str] = {}
        for m in re.finditer(rb'N4shin3ADV(\d+)([A-Z0-9_]+)E\x00', rodata):
            length = int(m.group(1))
            name = m.group(2).decode()
            if len(name) == length:
                str_addr_to_name[ro['addr'] + m.start()] = name

        ti_addr_to_name: dict[int, str] = {}
        for r_off, r_addend in self.rela.items():
            if r_addend in str_addr_to_name:
                ti_addr_to_name[r_off - 8] = str_addr_to_name[r_addend]

        vtable_to_name: dict[int, str] = {}
        for r_off, r_addend in self.rela.items():
            if r_addend in ti_addr_to_name:
                name = ti_addr_to_name[r_addend]
                vt_base = r_off - 8
                vtable_to_name[vt_base] = name
                vtable_to_name[vt_base + 16] = name  # address point

        slot_to_name = dict(vtable_to_name)
        for r_off, r_addend in self.rela.items():
            if r_addend in vtable_to_name:
                slot_to_name[r_off] = vtable_to_name[r_addend]

        return slot_to_name

    def find_candidate_vtables(self, min_entries: int) -> list[tuple[int, int]]:
        keys = sorted(k for k, v in self.rela.items() if self.text_lo <= v < self.text_hi)
        runs = []
        i = 0
        while i < len(keys):
            j = i
            while j + 1 < len(keys) and keys[j + 1] == keys[j] + 8:
                j += 1
            length = j - i + 1
            if length >= min_entries:
                runs.append((keys[i], length))
            i = j + 1
        return runs

    def scan_vtable_offset(self, handler_addr: int) -> int | None:
        insns = self.disasm_func(handler_addr, 140)
        for idx, ins in enumerate(insns):
            if ins.mnemonic == 'blr':
                for back in range(1, min(6, idx + 1)):
                    prev = insns[idx - back]
                    if prev.mnemonic == 'ldr':
                        m = re.search(r'#(0x[0-9a-f]+|\d+)', prev.op_str)
                        if m:
                            return int(m.group(1), 0)
        return None

    def resolve_adv_name_in_func(self, fn_addr: int, rtti_map: dict[int, str], depth: int = 2) -> str | None:
        hits = self._collect_adv_names(fn_addr, rtti_map, depth)
        if not hits:
            return None
        for name in hits:
            if name != 'NOP':
                return name
        return hits[0]

    def _collect_adv_names(self, fn_addr: int, rtti_map: dict[int, str], depth: int) -> list[str]:
        insns = self.disasm_func(fn_addr, 80)
        pages: dict[str, int] = {}
        callees: list[int] = []
        hits: list[str] = []

        for ins in insns:
            if ins.mnemonic == 'adrp':
                reg = ins.op_str.split(',')[0].strip()
                pages[reg] = ins.operands[1].imm
            elif ins.mnemonic in ('ldr', 'add') and len(ins.operands) >= 2:
                # Match `ldr x9, [x9, #0x10]`, `add x9, x9, #0x10`, and zero-offset `ldr x9, [x9]`
                m = re.match(r'(\w+),\s*\[?(\w+)(?:,\s*#(0x[0-9a-f]+|\d+))?\]?$', ins.op_str)
                if m:
                    dst, base, imm_s = m.group(1), m.group(2), m.group(3)
                    if base in pages:
                        target = pages[base] + (int(imm_s, 0) if imm_s else 0)
                        if target in rtti_map:
                            hits.append(rtti_map[target])
                        elif target in self.rela and self.rela[target] in rtti_map:
                            hits.append(rtti_map[self.rela[target]])
                    if dst in pages and dst != base:
                        del pages[dst]
            elif ins.mnemonic in ('bl', 'b') and ins.operands:
                tgt = ins.operands[0].imm
                if self.text_lo <= tgt < self.text_hi:
                    callees.append(tgt)

        if depth > 0 and not any(h != 'NOP' for h in hits):
            for callee in callees:
                hits.extend(self._collect_adv_names(callee, rtti_map, depth - 1))
        return hits

    def identify_read_primitives(self, table: dict[int, int]) -> dict[int, str]:
        # Use fixed VM instructions so detection is independent of command opcode differences
        prims: dict[int, str] = {}
        for op, expected in [(0x47, ['u32']), (0x46, ['u8', 'varint']), (0x44, ['u16'])]:
            h = table.get(op, 0)
            if not (self.text_lo <= h < self.text_hi):
                continue
            bls = [ins.operands[0].imm for ins in self.disasm_func(h, 40) if ins.mnemonic == 'bl']
            for idx, name in enumerate(expected):
                if idx < len(bls):
                    prims[bls[idx]] = name
        return prims

    def get_handler_signature(self, handler_addr: int, prims: dict[int, str]) -> tuple[list[str], bool]:
        insns = self.disasm_func(handler_addr, 140)
        calls = []
        has_loop = False
        for ins in insns:
            if ins.mnemonic == 'bl':
                tgt = ins.operands[0].imm
                calls.append(prims.get(tgt, f'fn_0x{tgt:x}'))
            elif ins.mnemonic in ('b.ne', 'b.lt', 'b.lo', 'cbnz', 'tbnz', 'tbz') and ins.operands:
                if ins.operands[-1].imm < ins.address:
                    has_loop = True
        return calls, has_loop


def format_sig(calls: list[str], has_loop: bool) -> str:
    sig = ', '.join(calls) if calls else '(none)'
    return f'{sig} [LOOP]' if has_loop else sig


def scan_opcodes(elf_path: str | Path) -> dict:
    img = ElfImage(elf_path)
    tbl_addr, table, stub = img.find_jump_table()
    rtti_map = img.build_adv_rtti_map()
    prims = img.identify_read_primitives(table)

    real_ops = sorted(op for op, h in table.items() if h != 0 and h != stub and img.text_lo <= h < img.text_hi)
    cmd_ops = [op for op in real_ops if op == 0 or op >= 0x80]
    insn_ops = [op for op in real_ops if 0 < op < 0x80]

    op_to_vtoff: dict[int, int] = {}
    for op in cmd_ops:
        vt_off = img.scan_vtable_offset(table[op])
        if vt_off is not None:
            op_to_vtoff[op] = vt_off

    max_vtoff = max(op_to_vtoff.values()) if op_to_vtoff else 0
    needed_len = max_vtoff // 8 + 1

    candidates = img.find_candidate_vtables(needed_len)
    best_vt = None
    best_names: dict[int, str] = {}
    best_score = (-1, False, -1)
    for vt_start, vt_len in candidates:
        resolved: dict[int, str] = {}
        for op, vt_off in op_to_vtoff.items():
            if vt_off < vt_len * 8:
                fn = img.rela.get(vt_start + vt_off, 0)
                if fn:
                    name = img.resolve_adv_name_in_func(fn, rtti_map, depth=2)
                    if name:
                        resolved[op] = name
        score = (len(set(resolved.values())), vt_len == needed_len, len(resolved))
        if score > best_score:
            best_score = score
            best_names = resolved
            best_vt = (vt_start, vt_len)

    instructions = []
    for op in insn_ops:
        calls, has_loop = img.get_handler_signature(table[op], prims)
        instructions.append({
            'opcode': op,
            'name': VM_INSTRUCTIONS.get(op, f'UNK_{op:02X}'),
            'addr': table[op],
            'args': calls,
            'loop': has_loop,
        })

    commands = []
    for op in cmd_ops:
        calls, has_loop = img.get_handler_signature(table[op], prims)
        commands.append({
            'opcode': op,
            'name': best_names.get(op, f'UNK_{op:02X}'),
            'addr': table[op],
            'vt_off': op_to_vtoff.get(op),
            'args': calls,
            'loop': has_loop,
        })

    return {
        'elf': str(elf_path),
        'jump_table': tbl_addr,
        'stub': stub,
        'dispatcher_vtable': best_vt,
        'instructions': instructions,
        'commands': commands,
    }


def load_opcode_rs(opcode_rs_path: Path) -> dict[str, dict[int, tuple[str, int | None]]]:
    src = opcode_rs_path.read_text(encoding='utf-8')
    result = {}
    for m in re.finditer(r'ShinVersion::(\w+) => Some\(if opcode != 0 && opcode < 0x80', src):
        ver = m.group(1)
        rest = src[m.end():]
        nxt = re.search(r'ShinVersion::\w+ =>', rest)
        body = rest[:nxt.start()] if nxt else rest
        pairs = {}
        for pm in re.finditer(
            r'0x([0-9a-fA-F]{2})\s*=>\s*([a-zA-Z_][\w#]*)\s*,?(?:\s*//\s*0x([0-9a-fA-F]+))?',
            body,
        ):
            op = int(pm.group(1), 16)
            name = pm.group(2)
            addr = int(pm.group(3), 16) if pm.group(3) else None
            pairs[op] = (name, addr)
        result[ver] = pairs
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Scan Shin Engine opcodes, handler addresses, and argument signatures from a Switch ELF.'
    )
    parser.add_argument('elfs', nargs='+', help='Path(s) to Switch ELF file(s) converted from NSO')
    parser.add_argument('--rust', action='store_true', help='Print match arms formatted for opcode.rs')
    parser.add_argument('--verify', metavar='OPCODE_RS', help='Verify scanned table against opcode.rs')
    parser.add_argument('--version', metavar='NAME', help='ShinVersion variant name in opcode.rs (e.g. SugarStyle)')
    args = parser.parse_args()

    rs_tables = load_opcode_rs(Path(args.verify)) if args.verify else {}
    results = []
    exit_code = 0

    for elf_path in args.elfs:
        label = args.version if (args.version and len(args.elfs) == 1) else Path(elf_path).stem
        res = scan_opcodes(elf_path)
        results.append(res)

        vt_info = (
            f"0x{res['dispatcher_vtable'][0]:x} (len={res['dispatcher_vtable'][1]})"
            if res['dispatcher_vtable']
            else 'None'
        )
        print('=' * 76)
        print(f"{label} ({res['elf']})")
        print(f"  jump_table=0x{res['jump_table']:x}  stub=0x{res['stub']:x}  dispatcher_vt={vt_info}")
        print(
            f"  instructions={len(res['instructions'])}  commands={len(res['commands'])}"
            f"  total={len(res['instructions']) + len(res['commands'])}"
        )

        if args.rust:
            print('  // Instructions')
            for item in res['instructions']:
                print(f"  0x{item['opcode']:02x} => {item['name']}, // 0x7100{item['addr']:06x}")
            print('  // Commands')
            for item in res['commands']:
                print(f"  0x{item['opcode']:02x} => {item['name']}, // 0x7100{item['addr']:06x}")
        else:
            print('  -- Instructions (0x40-0x53) --')
            for item in res['instructions']:
                sig = format_sig(item['args'], item['loop'])
                print(f"  0x{item['opcode']:02X}  {item['name']:14s}  0x7100{item['addr']:06x}  args=[{sig}]")
            print('  -- Commands --')
            unresolved = [c for c in res['commands'] if c['name'].startswith('UNK_')]
            print(f"  RTTI resolved: {len(res['commands']) - len(unresolved)}/{len(res['commands'])}")
            for item in res['commands']:
                vt_s = f"vt+0x{item['vt_off']:03x}" if item['vt_off'] is not None else 'vt=?'
                sig = format_sig(item['args'], item['loop'])
                print(f"  0x{item['opcode']:02X}  {item['name']:14s}  0x7100{item['addr']:06x}  {vt_s}  args=[{sig}]")

        if args.verify:
            if label not in rs_tables:
                print(f'  [VERIFY SKIP]: ShinVersion::{label} not found in {args.verify} (pass --version)')
                exit_code = 1
            else:
                rs = rs_tables[label]
                scanned = {item['opcode']: (item['name'], item['addr']) for item in res['instructions']}
                scanned.update({item['opcode']: (item['name'], item['addr']) for item in res['commands']})
                diffs = []
                for op in sorted(set(rs) | set(scanned)):
                    r = rs.get(op)
                    e = scanned.get(op)
                    if r is None:
                        diffs.append(f'  MISSING in opcode.rs: 0x{op:02X} {e[0]} @0x{e[1]:x}')
                    elif e is None:
                        diffs.append(f'  EXTRA in opcode.rs:   0x{op:02X} {r[0]}')
                    else:
                        r_addr = (r[1] & 0xFFFFF) if (r[1] is not None and r[1] >= 0x710000000) else r[1]
                        if r[0] != e[0] or (r_addr is not None and r_addr != e[1]):
                            diffs.append(
                                f'  DIFF 0x{op:02X}: opcode.rs=({r[0]}, {hex(r_addr) if r_addr else None})'
                                f' vs ELF=({e[0]}, 0x{e[1]:x})'
                            )
                if diffs:
                    exit_code = 1
                    print('  [VERIFY MISMATCHES]:')
                    for d in diffs:
                        print(d)
                else:
                    print('  [VERIFY OK]: 100% match with opcode.rs (names + addresses)!')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
