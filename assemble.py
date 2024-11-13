#!/usr/bin/env python3

# Author: Jacob Knowlton
# Date: Nov 2024
# Assembler for CS3810 A8, designed to work with the CPU instruction specification from Fall 2024

from dataclasses import dataclass, field
from typing import cast, Callable, Literal, List, Optional, Union
import re
import sys

NUM_REGS = 4

OPCODES = {
    "and": 0x10,
    "or": 0x12,
    "add": 0x14,
    "sub": 0x1C,
    "andi": 0x11,
    "ori": 0x13,
    "addi": 0x15,
    "subi": 0x1D,
    "slt": 0x1E,
    "lw": 0x35,
    "sw": 0x25,
    "blt": 0x0D,
    "halt": 0x0F,
}


@dataclass
class Register:
    num: int

    def __repr__(self):
        return f"r{self.num}"


@dataclass
class Immediate:
    value: int

    def __repr__(self):
        return f"{self.value}"


@dataclass
class Comment:
    text: str

    def __repr__(self):
        return f"# {self.text}"


@dataclass
class Instruction:
    instr_type: Literal["r", "i", "j", "z"]
    opcode: str
    operands: List[Union["Register", "Immediate"]] = field(default_factory=list)
    comment: Optional[Comment] = None

    def __repr__(self):
        operands_str = ", ".join(map(str, self.operands))
        comment_str = f"  {self.comment}" if self.comment else ""
        return f"{self.opcode} {operands_str}{comment_str}"

    def assemble(self) -> int:
        machine = OPCODES[self.opcode]
        if self.instr_type == "r":
            rd = cast(Register, self.operands[0])
            rs = cast(Register, self.operands[1])
            rt = cast(Register, self.operands[2])
            machine |= (rd.num << 8) | (rs.num << 12) | (rt.num << 10)
        elif self.instr_type in {"i", "j"}:
            rt = cast(Register, self.operands[0])
            rs = cast(Register, self.operands[1])
            imm = cast(Immediate, self.operands[2])
            machine |= (rt.num << 10) | (rs.num << 12) | (imm.value << 6)
        elif self.instr_type == "z":
            pass  # No additional encoding needed for zero-operand instructions
        return machine


def parse_register(reg_str: str) -> Register:
    if not re.fullmatch(r"r\d+", reg_str):
        raise ValueError(f"Invalid register format: '{reg_str}'")
    num = int(reg_str[1:])
    if not (0 <= num < NUM_REGS):
        raise ValueError(f"Register number out of range: r{num}")
    return Register(num)


def parse_immediate(imm_str: str) -> Immediate:
    try:
        value = int(imm_str, 0)  # Supports decimal and hex (e.g., 0xF)
    except ValueError:
        raise ValueError(f"Invalid immediate value: '{imm_str}'")
    if not (0 <= value <= 0xF):
        raise ValueError(f"Immediate value out of range (0-15): {value}")
    return Immediate(value)


def parse_operand(op: str):
    op_str = op.strip()
    if op_str.startswith("r"):
        return parse_register(op_str)
    else:
        return parse_immediate(op_str)


def parse_comment(line: str) -> tuple[str, str | None]:
    match = re.search(r"#(.*)", line)
    if match:
        code_part = line[: match.start()].strip()
        comment_part = match.group(1).strip()
        return code_part, comment_part
    return line.strip(), None


def parse_r_type(line: str) -> Instruction:
    code, comment = parse_comment(line)
    parts = code.casefold().split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Missing operands for R-type instruction: {line}")
    opcode, operands_str = parts
    operands = [parse_operand(op.strip()) for op in operands_str.split(",")]
    if len(operands) != 3 or not all(isinstance(op, Register) for op in operands):
        raise ValueError(f"Invalid operands for R-type instruction: '{line}'")
    return Instruction("r", opcode, operands, Comment(comment) if comment else None)


def parse_imm_type(instr_type: Literal["i", "j"]) -> Callable[[str], Instruction]:
    def inner_parse(line: str) -> Instruction:
        code, comment = parse_comment(line)
        parts = code.casefold().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Missing operands for {instr_type.upper()}-type instruction: '{line}'"
            )
        opcode, operands_str = parts
        operands = [parse_operand(op.strip()) for op in operands_str.split(",")]
        if (
            len(operands) != 3
            or not isinstance(operands[0], Register)
            or not isinstance(operands[1], Register)
            or not isinstance(operands[2], Immediate)
        ):
            raise ValueError(
                f"Invalid operands for {instr_type.upper()}-type instruction: '{line}'"
            )
        return Instruction(
            instr_type, opcode, operands, Comment(comment) if comment else None
        )

    return inner_parse


def parse_mem_type(line: str) -> Instruction:
    code, comment = parse_comment(line)
    parts = code.casefold().split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Missing operands for memory instruction: '{line}'")
    opcode, operands_str = parts
    operands = operands_str.split(",")
    if len(operands) != 2:
        raise ValueError(
            f"Incorrect number of operands for memory instruction: '{line}'"
        )

    dest_reg_str = operands[0].strip()
    mem_operand_str = operands[1].strip()

    dest_reg = parse_register(dest_reg_str)

    mem_match = re.fullmatch(r"(\d+)?\s*\(\s*(r\d+)\s*\)", mem_operand_str)
    if not mem_match:
        raise ValueError(f"Invalid memory operand format: '{mem_operand_str}'")

    offset_str, base_reg_str = mem_match.groups()
    offset = parse_immediate(offset_str) if offset_str else Immediate(0)
    base_reg = parse_register(base_reg_str)

    return Instruction(
        "i", opcode, [dest_reg, base_reg, offset], Comment(comment) if comment else None
    )


def parse_zero_operand_type(line: str) -> Instruction:
    code, comment = parse_comment(line)
    parts = code.casefold().split()
    if len(parts) != 1:
        raise ValueError(f"Zero-operand instruction should not have operands: '{line}'")
    opcode = parts[0]
    return Instruction("z", opcode, [], Comment(comment) if comment else None)


# Dictionary for opcode-specific parsing and handlers
OPCODE_HANDLERS: dict[str, Callable[[str], Instruction]] = {
    "and": parse_r_type,
    "or": parse_r_type,
    "add": parse_r_type,
    "sub": parse_r_type,
    "slt": parse_r_type,
    "andi": parse_imm_type("i"),
    "ori": parse_imm_type("i"),
    "addi": parse_imm_type("i"),
    "subi": parse_imm_type("i"),
    "blt": parse_imm_type("j"),
    "lw": parse_mem_type,
    "sw": parse_mem_type,
    "halt": parse_zero_operand_type,
}


def parse_assembly(lines: List[str]) -> List[Instruction]:
    instructions: List[Instruction] = []
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue  # Skip empty lines and full-line comments
        try:
            opcode = stripped_line.split()[0].casefold()
            handler = OPCODE_HANDLERS.get(opcode)
            if not handler:
                print(f"Line {line_num}: Unknown opcode '{opcode}'")
                continue
            instruction = handler(stripped_line)
            instructions.append(instruction)
        except ValueError as ve:
            print(f"Line {line_num}: {ve}")
    return instructions


def read_assembly_from_file(filename: str) -> List[str]:
    try:
        with open(filename, "r") as file:
            return file.readlines()
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)


def read_assembly_from_input() -> List[str]:
    print("Enter assembly instructions (press Enter twice to finish):")
    lines = []
    empty_count = 0
    while True:
        line = input()
        if line.strip() == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
            lines.append(line)
    return lines


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        assembly_lines = read_assembly_from_file(filename)
    else:
        assembly_lines = read_assembly_from_input()

    instructions = parse_assembly(assembly_lines)

    if not instructions:
        print("No valid instructions to assemble.")
        return

    print("\nAssembled Instructions:")
    for instr in instructions:
        machine_code = instr.assemble()
        print(f"0x{machine_code:04x},", end="")
    print()


if __name__ == "__main__":
    main()
