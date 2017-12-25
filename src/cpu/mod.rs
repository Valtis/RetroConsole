
use memory_controller::MemoryController;
use std::rc::Rc;
use std::cell::RefCell;

/* CPU flags */
const ZERO_FLAG: u8 = 0x01;
const NEGATIVE_FLAG: u8 = 0x02;
const CARRY_FLAG: u8 = 0x04;
const OVERFLOW_FLAG: u8 = 0x08;
const INTERRUPT_ENABLE_FLAG: u8 = 0x10;
const BREAK_FLAG: u8  = 0x20;
const AUTO_INCREMENT_FLAG: u8 = 0x40;
const FAULT_FLAG: u8 = 0x80;

/* Register encodings */
const ENCODING_R1:u8 = 0x00;
const ENCODING_R2:u8 = 0x01;
const ENCODING_R3:u8 = 0x02;
const ENCODING_R4:u8 = 0x03;

/* LDR/STR addressing modes */
const IMMEDIATE_ADDRESSING: u8 = 0x00;
const ABSOLUTE_ADDRESSING: u8 = 0x01;
const INDEXED_ABSOLUTE_ADDRESSING: u8 = 0x02;
const INDIRECT_ADDRESSING: u8 = 0x03;

/* Arithmetic operation addressing modes */
const REGISTER_REGISTER_ADDRESSING: u8 = 0x00;
const REGISTER_IMMEDIATE_ADDRESSING: u8 = 0x01;

/*
    Opcodes. Note that only 6 bits are reserved for opcodes in the instruction
    encoding, so largest possible value fo opcode is 0x3F. This also means
    there are only 64 possible instructions, with 4 possible addressing modes
    for each
*/

/* Data movement instructions  */
const LOAD_REGISTER: u8 = 0x10;
const STORE_REGISTER: u8 = 0x11;

/* Arithmetic instructions */

const ADD_WITH_CARRY:u8 = 0x30;
const ADD_WITHOUT_CARRY: u8 = 0x31;

const SIGNED_MULTIPLY: u8 = 0x34;
const UNSIGNED_MULTIPLY:u8 = 0x35;



/* Flag instructions */
const SET_AUTO_INCREMENT_FLAG: u8 = 0x07;
const CLEAR_AUTO_INCREMENT_FLAG: u8 = 0x06;

/* Initial value of the stack pointer */
const SP_INITIAL_VALUE: u16 = 0x200;


/*
    Micro ops that actually define what the CPU is doing currently.
    Using micro ops means certain instructions naturally take certain
    number of cycles, rather than arbitrarily deciding that some instructions
    take certain number of cycles.
*/
#[derive(Clone, Copy)]
enum MicroOp {
    Nop,
    SetAutoIncrementFlag,
    ClearAutoIncrementFlag,
    FetchValue,
    FetchDestSrc,
    FetchLowAddress,
    FetchHighAddress,
    FetchIndirectLowAddress,
    FetchIndirectHighAddress,
    IncrementAndStoreIndirectLowByte,
    StoreIndirectHighByte,
    ImmedateToRegister,
    AbsoluteToRegister,
    IndexedAddress,
    AddWithoutCarryRegister,
    AddWithoutCarryImmediate,
    BeginMultiply,
    SignedMultiplyInvertNegativeMultiplier,
    SignedMultiplyInvertNegativeMultiplicand,
    MultiplyAdd,
    MultiplyShiftMultiplier,
    MultiplyShiftMultiplicand,
    SignedMultiplyInvertResultIfFlag,
    EndSignedMultiply,
    EndUnsignedMultiply,
}

/* CPU registers. Duh. */
 #[derive(Debug)]
 struct Registers {
    r1: u8,
    r2: u8,
    r3: u8,
    r4: u8,
    pc: u16,
    sp: u16,
    flags: u8,
}

/* Used to store various pieces of data required by the CPU state machine */
struct StateMachine {
    micro_ops: Vec<MicroOp>,
    index: usize,
    value_register: u8,
    dest_src_register: u8,
    address_register: u16,
    indirect_address_register: u16,
    multiply_negate: bool,
}

impl StateMachine {
    fn new() -> StateMachine {
        StateMachine {
            micro_ops: vec![],
            index: 0,
            value_register: 0,
            dest_src_register: 0,
            address_register: 0,
            indirect_address_register: 0,
            multiply_negate: false,
        }
    }
}

impl Registers {
    fn new() -> Registers {
        Registers {
            r1: 0,
            r2: 0,
            r3: 0,
            r4: 0,
            pc: 0,
            sp: SP_INITIAL_VALUE,
            flags: INTERRUPT_ENABLE_FLAG,
        }
    }

    fn zero_flag(&self) -> bool {
        self.flags & ZERO_FLAG != 0
    }

    fn negative_flag(&self) -> bool {
        self.flags & NEGATIVE_FLAG != 0
    }

    fn carry_flag(&self) -> bool {
        self.flags & CARRY_FLAG != 0
    }

    fn overflow_flag(&self) -> bool {
        self.flags & OVERFLOW_FLAG != 0
    }

    fn auto_increment_flag(&self) -> bool {
        self.flags & AUTO_INCREMENT_FLAG != 0
    }

    fn set_zero_negative_flags(&mut self, value: u8) {
        self.set_zero_flag_on_value(value);
        self.set_negative_flag_on_value(value);
    }

    fn set_zero_flag_on_value(&mut self, value: u8) {
        if value == 0 {
            self.set_zero_flag();
        } else {
            self.clear_zero_flag();
        }
    }

    fn set_zero_flag(&mut self) {
        self.flags |= ZERO_FLAG;
    }

    fn clear_zero_flag(&mut self) {
        self.flags &= !ZERO_FLAG;
    }

    fn set_negative_flag_on_value(&mut self, value: u8) {
        if value > 127 {
            self.flags |= NEGATIVE_FLAG;
        } else {
            self.flags &= !NEGATIVE_FLAG;
        }
    }

    fn set_carry_flag_on_value(&mut self, result: u16) {
        if result > 255 {
            self.set_carry_flag();
        } else {
            self.clear_carry_flag();
        }
    }

    fn set_carry_flag(&mut self) {
        self.flags |= CARRY_FLAG;
    }

    fn clear_carry_flag(&mut self) {
        self.flags &= !CARRY_FLAG;
    }

    /*
        overflow happens if both src values have same sign (8th bit set/unset in
        both), and the sign bit in in the result is different --> operation
        on two positive values resulted in negative number, or other way around
    */
    fn set_overflow_flag_on_value(
        &mut self,
        src1: u8,
        src2: u8,
        result: u16) {
        let same_sign = (0x80 & src1) == (0x80 & src2);
        let src_dst_differs = (0x80 & src1) != (0x80 & result) as u8;
        if same_sign && src_dst_differs {
            self.set_overflow_flag();
        } else {
            self.clear_overflow_flag();
        }
    }

    fn set_overflow_flag(&mut self) {
        self.flags |= OVERFLOW_FLAG;
    }

    fn clear_overflow_flag(&mut self) {
        self.flags &= !OVERFLOW_FLAG;
    }

    fn set_auto_increment_flag(&mut self) {
        self.flags |= AUTO_INCREMENT_FLAG;
    }

    fn clear_autoincrement_flag(&mut self) {
        self.flags &= !AUTO_INCREMENT_FLAG;
    }

}

pub struct Cpu {
    registers: Registers,
    state: StateMachine,
    memory_controller: Rc<RefCell<MemoryController>>,
}

impl Cpu {
    pub fn new(controller: Rc<RefCell<MemoryController>>) -> Cpu {
        // temp code: create test program for the cpu

      let temp_instructions = vec![
            // LDR r1, #$FF
            (LOAD_REGISTER << 2) | IMMEDIATE_ADDRESSING,
            ENCODING_R1,
            (8 as u8).wrapping_neg(),
            // LDR r1, #$03
            (LOAD_REGISTER << 2) | IMMEDIATE_ADDRESSING,
            ENCODING_R2,
            0x09,
            // MUL r4, r3, r2, r1
            (SIGNED_MULTIPLY << 2) | REGISTER_REGISTER_ADDRESSING,
            0b00011011
        ];

        for (i, v) in temp_instructions.iter().enumerate() {
            controller.borrow_mut().write(i as u16, *v);
        }


        controller.borrow_mut().write(0x1234, 14);
        controller.borrow_mut().write(0x1235, 42);
        controller.borrow_mut().write(0x1236, 21);

        controller.borrow_mut().write(0xABCD, 0x36);
        controller.borrow_mut().write(0xABCE, 0x12);

        Cpu {
            registers: Registers::new(),
            state: StateMachine::new(),
            memory_controller: controller,
        }
    }

    fn read(&self, address: u16) -> u8 {
        self.memory_controller.borrow().read(address)
    }

    fn read_pc(&mut self) -> u8 {
        let pc = self.registers.pc;
        let value = self.read(pc);
        self.registers.pc += 1;
        value
    }

    pub fn execute(&mut self) {
        if self.state.index == self.state.micro_ops.len() {
            println!();
            self.state.index = 0;
            self.state.micro_ops.clear();
            let value = self.read_pc();

            let opcode = value >> 2;
            let addressing = value & 0x03;
            match opcode {
                /* Data movement instructions */
                LOAD_REGISTER => self.decode_ldr(addressing),
                /* Arithmetic instructions */
                ADD_WITHOUT_CARRY => self.decode_add(addressing),
                SIGNED_MULTIPLY => self.decode_imul(addressing),
                UNSIGNED_MULTIPLY => self.decode_mul(addressing),
                /* Status flag instructions  */
                SET_AUTO_INCREMENT_FLAG => self.decode_sai(),
                CLEAR_AUTO_INCREMENT_FLAG => self.decode_cla(),

                e => panic!("Unimplemented opcode 0x{:0x}", e),
            }
        } else {
            self.execute_micro_op();
        }

        println!("Registers: {:?}", self.registers);
    }

    fn decode_ldr(&mut self, addressing: u8) {
        match addressing & 0x03 {
            IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::ImmedateToRegister);
            },
            ABSOLUTE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchLowAddress);
                self.state.micro_ops.push(MicroOp::FetchHighAddress);
                self.state.micro_ops.push(MicroOp::AbsoluteToRegister);
            },
            INDEXED_ABSOLUTE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchLowAddress);
                self.state.micro_ops.push(MicroOp::FetchHighAddress);
                self.state.micro_ops.push(MicroOp::IndexedAddress);
                self.state.micro_ops.push(MicroOp::AbsoluteToRegister);
            },
            INDIRECT_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchLowAddress);
                self.state.micro_ops.push(MicroOp::FetchHighAddress);
                self.state.micro_ops.push(MicroOp::FetchIndirectLowAddress);
                self.state.micro_ops.push(MicroOp::FetchIndirectHighAddress);
                self.state.micro_ops.push(MicroOp::AbsoluteToRegister);

                if self.registers.auto_increment_flag() {
                    self.state.micro_ops.push(
                        MicroOp::IncrementAndStoreIndirectLowByte);
                    self.state.micro_ops.push(
                        MicroOp::StoreIndirectHighByte);
                }
            },
            _ => unreachable!(),
        }
    }

    fn decode_add(&mut self, addressing: u8) {
        match addressing & 0x01 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::AddWithoutCarryRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(MicroOp::AddWithoutCarryImmediate);
            },
            _ => unreachable!(),
        }
    }

    fn decode_imul(&mut self, addressing: u8) {
        match addressing & 0x01 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginMultiply);
                self.state.micro_ops.push(
                    MicroOp::SignedMultiplyInvertNegativeMultiplicand);
                self.state.micro_ops.push(
                    MicroOp::SignedMultiplyInvertNegativeMultiplier);

                for _ in 0..7 {
                    self.state.micro_ops.push(
                        MicroOp::MultiplyAdd);
                    /*
                        Addition should take 2 cycles, as the cpu
                        has 8-bit alu and the target register is 16 bit.
                        I'm too lazy to actually implement this in microcode,
                        so the first op does whole addition and the second
                        one just executes nop for cycle timing.
                    */
                    self.state.micro_ops.push(MicroOp::Nop);
                    self.state.micro_ops.push(
                        MicroOp::MultiplyShiftMultiplier);
                    self.state.micro_ops.push(
                        MicroOp::MultiplyShiftMultiplicand);
                }
                self.state.micro_ops.push(MicroOp::MultiplyAdd);
                // as above
                self.state.micro_ops.push(MicroOp::Nop);
                self.state.micro_ops.push(
                    MicroOp::SignedMultiplyInvertResultIfFlag);

                self.state.micro_ops.push(MicroOp::EndSignedMultiply);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                unimplemented!();
            }
            _ => unreachable!(),
        }
    }

    fn decode_mul(&mut self, addressing: u8) {
        match addressing & 0x01 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginMultiply);

                for _ in 0..7 {
                    self.state.micro_ops.push(
                        MicroOp::MultiplyAdd);
                    // as with imul
                    self.state.micro_ops.push(MicroOp::Nop);
                    self.state.micro_ops.push(
                        MicroOp::MultiplyShiftMultiplier);
                    self.state.micro_ops.push(
                        MicroOp::MultiplyShiftMultiplicand);
                }
                self.state.micro_ops.push(MicroOp::MultiplyAdd);
                // as with imul
                self.state.micro_ops.push(MicroOp::Nop);
                self.state.micro_ops.push(MicroOp::EndUnsignedMultiply);

            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                unimplemented!();
            }
            _ => unreachable!(),
        }
    }

    fn decode_sai(&mut self) {
        self.state.micro_ops.push(MicroOp::SetAutoIncrementFlag);
    }

    fn decode_cla(&mut self) {
        self.state.micro_ops.push(MicroOp::ClearAutoIncrementFlag);
    }

    fn execute_micro_op(&mut self) {

        let current_op = self.state.micro_ops[self.state.index];
        match current_op {
            MicroOp::Nop => {
                // do nothing
            },
            MicroOp::SetAutoIncrementFlag => {
                self.registers.set_auto_increment_flag();
            },
            MicroOp::ClearAutoIncrementFlag => {
                self.registers.clear_autoincrement_flag();
            },
            MicroOp::FetchValue => {
                self.state.value_register = self.read_pc();
            }
            MicroOp::FetchDestSrc => {
                self.state.dest_src_register = self.read_pc();
            },
            MicroOp::FetchLowAddress => {
                self.state.address_register = self.read_pc() as u16;
            },
            MicroOp::FetchHighAddress => {
                self.state.address_register |= (self.read_pc() as u16) << 8;
            },
            MicroOp::FetchIndirectLowAddress => {
                self.state.indirect_address_register =
                    self.state.address_register;

                self.state.address_register =
                    self.read(self.state.indirect_address_register) as u16;
            },
            MicroOp::FetchIndirectHighAddress => {
                self.state.indirect_address_register += 1;
                self.state.address_register |=
                    (self.read(self.state.indirect_address_register) as u16)
                    << 8;
            },
            MicroOp::IncrementAndStoreIndirectLowByte => {
                self.state.address_register =
                    (self.state.address_register as u32 + 1) as u16;
                self.state.indirect_address_register -= 1;

                self.memory_controller.borrow_mut().write(
                    self.state.indirect_address_register,
                    self.state.address_register as u8);
            },
            MicroOp::StoreIndirectHighByte => {
                self.memory_controller.borrow_mut().write(
                    self.state.indirect_address_register + 1 ,
                    (self.state.address_register >> 8) as u8);
            },
            MicroOp::ImmedateToRegister => {
                let immediate = self.read_pc();
                let destination = self.state.dest_src_register & 0x03;
                self.store_register(destination, immediate);
            },
            MicroOp::AbsoluteToRegister => {
                let value = self.read(self.state.address_register);
                let destination = self.state.dest_src_register & 0x03;
                self.store_register(destination, value);
            },
            MicroOp::IndexedAddress => {
                let index_reg = (self.state.dest_src_register >> 2) & 0x03;
                let value = self.load_register(index_reg);
                self.state.address_register += value as u16;

                if self.registers.auto_increment_flag() {
                    self.store_register(
                        index_reg,
                        (value as u16 + 1) as u8);
                }
            },
            MicroOp::AddWithoutCarryRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);
                let result = src1val as u16 + src2val as u16;

                self.registers.set_carry_flag_on_value(result);
                self.registers.set_overflow_flag_on_value(src1val, src2val, result);

                self.store_register(destination, result as u8);
            },
            MicroOp::AddWithoutCarryImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.state.value_register;
                let result = src1val as u16 + src2val as u16;


                self.registers.set_carry_flag_on_value(result);
                self.registers.set_overflow_flag_on_value(src1val, src2val, result);

                self.store_register(destination, result as u8);
            },
            MicroOp::BeginMultiply => {
                let src1 = (self.state.dest_src_register >> 4) & 0x03;
                let src2 = (self.state.dest_src_register >> 6) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);

                // indirect & direct address registers are reused for
                // multiplication
                self.state.indirect_address_register = 0;
                self.state.address_register = src2val as u16;
                self.state.value_register = src1val;
                self.state.multiply_negate = false;
            },
            MicroOp::SignedMultiplyInvertNegativeMultiplicand => {
                if self.state.address_register > 127 {
                    self.state.address_register =
                        (self.state.address_register as u8)
                            .wrapping_neg() as u16;
                    self.state.multiply_negate = !self.state.multiply_negate;
                }
            },
            MicroOp::SignedMultiplyInvertNegativeMultiplier => {
                if self.state.value_register > 127 {
                    self.state.value_register =
                        self.state.value_register.wrapping_neg();
                    self.state.multiply_negate = !self.state.multiply_negate;
                }

            },
            MicroOp::MultiplyAdd => {
                if self.state.value_register & 0x01 != 0 {
                    self.state.indirect_address_register =
                        (self.state.indirect_address_register as u32 +
                        self.state.address_register as u32) as u16;
                }
            },
            MicroOp::MultiplyShiftMultiplier => {
                self.state.value_register >>= 1;
            },
            MicroOp::MultiplyShiftMultiplicand => {
                self.state.address_register <<= 1;
            },
            MicroOp::SignedMultiplyInvertResultIfFlag => {
                if self.state.multiply_negate {
                    self.state.indirect_address_register =
                        self.state.indirect_address_register.wrapping_neg();
                }
            },
            MicroOp::EndSignedMultiply => {
                let (_, high, low) = self.end_mul_common();
                let sign_bit = 0x80 & low;

                if (sign_bit != 0 && high == 0xFF)
                    || (sign_bit == 0 && high == 0) {
                    self.registers.clear_overflow_flag();
                    self.registers.clear_carry_flag();
                } else {
                    self.registers.set_overflow_flag();
                    self.registers.set_carry_flag();
                }

            }
            MicroOp::EndUnsignedMultiply => {
                let (res, high, _) = self.end_mul_common();
                if high == 0 {
                    self.registers.clear_overflow_flag();
                    self.registers.clear_carry_flag();
                } else {
                    self.registers.set_overflow_flag();
                    self.registers.set_carry_flag();
                }
            }
        }

        self.state.index += 1;
    }

    fn end_mul_common(&mut self) -> (u16, u8, u8) {
        let high_byte_reg = self.state.dest_src_register & 0x03;
        let low_byte_reg = (self.state.dest_src_register >> 2) & 0x03;
        let result = self.state.indirect_address_register;

        self.store_register(
            low_byte_reg,
            result as u8);

        self.store_register(
            high_byte_reg,
            (result >> 8) as u8);

        if result == 0 {
            self.registers.set_zero_flag();
        } else {
            self.registers.clear_zero_flag();
        }

        (result, (result >> 8) as u8, result as u8)
    }

    fn store_register(&mut self, register: u8, value: u8) {
        match register & 0x03 {
            ENCODING_R1 => self.registers.r1 = value,
            ENCODING_R2 => self.registers.r2 = value,
            ENCODING_R3 => self.registers.r3 = value,
            ENCODING_R4 => self.registers.r4 = value,
            _ => unreachable!(),
        };
        self.registers.set_zero_negative_flags(value);
    }


    fn load_register(&mut self, register: u8) -> u8 {
        match register & 0x03 {
            ENCODING_R1 => self.registers.r1,
            ENCODING_R2 => self.registers.r2,
            ENCODING_R3 => self.registers.r3,
            ENCODING_R4 => self.registers.r4,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;

    fn create_test_cpu() -> Cpu {
        let memory_controller = Rc::new(
            RefCell::new(
                MemoryController::new(128*1024)));
        Cpu::new(memory_controller)
    }

    fn emit_set_auto_increment(opcodes: &mut Vec<u8>) {
        opcodes.push(SET_AUTO_INCREMENT_FLAG << 2);
    }


    fn emit_clear_auto_increment(opcodes: &mut Vec<u8>) {
        opcodes.push(CLEAR_AUTO_INCREMENT_FLAG << 2);
    }

    fn emit_load_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        immediate: u8) {
        opcodes.push((LOAD_REGISTER << 2) | IMMEDIATE_ADDRESSING);
        opcodes.push(destination & 0x03);
        opcodes.push(immediate);
    }

    fn emit_load_absolute(
        opcodes: &mut Vec<u8>,
        destination: u8,
        address: u16) {
        opcodes.push((LOAD_REGISTER << 2) | ABSOLUTE_ADDRESSING);
        opcodes.push(destination & 0x03);
        opcodes.push(address as u8);
        opcodes.push((address >> 8)  as u8);
    }

    fn emit_load_indexed_absolute(
        opcodes: &mut Vec<u8>,
        destination: u8,
        address: u16,
        index: u8) {
        opcodes.push((LOAD_REGISTER << 2) | INDEXED_ABSOLUTE_ADDRESSING);
        opcodes.push(destination & 0x03 | ((index & 0x03) << 2));
        opcodes.push(address as u8);
        opcodes.push((address >> 8)  as u8);
    }

    fn emit_load_indirect(
        opcodes: &mut Vec<u8>,
        destination: u8,
        address: u16) {
        opcodes.push((LOAD_REGISTER << 2) | INDIRECT_ADDRESSING);
        opcodes.push(destination & 0x03);
        opcodes.push(address as u8);
        opcodes.push((address >> 8)  as u8);
    }

    fn emit_add_without_carry_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((ADD_WITHOUT_CARRY << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_add_without_carry_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((ADD_WITHOUT_CARRY << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_unsigned_multiply_reg_reg(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((UNSIGNED_MULTIPLY << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) <<  4) |
            ((src_2 & 0x03) << 6));
    }

    fn emit_signed_multiply_reg_reg(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((SIGNED_MULTIPLY << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) <<  4) |
            ((src_2 & 0x03) << 6));
    }


    fn update_program(cpu: &mut Cpu, program: Vec<u8>, load_address: u16) {
        cpu.registers.pc = load_address;
        let mut controller = cpu.memory_controller.borrow_mut();
        for (index, value) in program.iter().enumerate() {
            controller.write(load_address + index as u16, *value);
        }
    }

    fn execute_instruction(cpu: &mut Cpu) {
        cpu.execute();
        while cpu.state.index < cpu.state.micro_ops.len() {
            cpu.execute();
        }
    }

    #[test]
    fn set_auto_increment_flag_sets_the_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_set_auto_increment(&mut program);
        update_program(&mut cpu, program, 0x1000);

        execute_instruction(&mut cpu);

        assert!(cpu.registers.auto_increment_flag());
    }

    #[test]
    fn clear_auto_increment_flag_unsets_the_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_clear_auto_increment(&mut program);
        update_program(&mut cpu, program, 0x1000);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert!(!cpu.registers.auto_increment_flag());
    }

    #[test]
    fn load_immediate_sets_register_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_load_immediate(&mut program, ENCODING_R1, 0x83);
        update_program(&mut cpu, program, 0x1000);

        execute_instruction(&mut cpu);

        assert_eq!(0x83, cpu.registers.r1);
    }

    #[test]
    fn load_immediate_sets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_load_immediate(&mut program, ENCODING_R1, 0);
        update_program(&mut cpu, program, 0x1000);

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn load_immediate_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        cpu.registers.flags = ZERO_FLAG;
        emit_load_immediate(&mut program, ENCODING_R1, 5);
        update_program(&mut cpu, program, 0x1000);

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }


    #[test]
    fn load_immediate_sets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_immediate(&mut program, ENCODING_R1, 200);
        update_program(&mut cpu, program, 0x1000);

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn load_immediate_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        cpu.registers.flags = ZERO_FLAG;
        emit_load_immediate(&mut program, ENCODING_R1, 5);
        update_program(&mut cpu, program, 0x1000);

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }

    #[test]
    fn load_absolute_sets_register_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_load_absolute(&mut program, ENCODING_R2, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0xFF);
        execute_instruction(&mut cpu);
        assert_eq!(0xFF, cpu.registers.r2);
    }

    #[test]
    fn load_absolute_sets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_absolute(&mut program, ENCODING_R2, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0);
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn load_absolute_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_absolute(&mut program, ENCODING_R2, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0x23);
        cpu.registers.flags = ZERO_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn load_absolute_sets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_absolute(&mut program, ENCODING_R2, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0x86);
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn load_absolute_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_absolute(&mut program, ENCODING_R2, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0x70);
        cpu.registers.flags = NEGATIVE_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }

    #[test]
    fn load_indexed_absolute_sets_register_correctly() {

        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r4 = 0x04;
        cpu.memory_controller.borrow_mut().write(0x1238, 0xFA);
        execute_instruction(&mut cpu);
        assert_eq!(0xFA, cpu.registers.r3);
    }

    #[test]
    fn load_indexed_absolute_sets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0x04;

        cpu.memory_controller.borrow_mut().write(0x1238, 0);
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn load_indexed_absolute_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0x04;

        cpu.memory_controller.borrow_mut().write(0x1238, 23);
        cpu.registers.flags = ZERO_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn load_indexed_absolute_sets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0x04;

        cpu.memory_controller.borrow_mut().write(0x1238, 0x80);
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn load_indexed_absolute_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0x04;

        cpu.memory_controller.borrow_mut().write(0x1238, 0x7F);
        cpu.registers.flags = NEGATIVE_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }

    #[test]
    fn index_register_is_not_incremented_if_flag_is_unset() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0x04;

        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x04, cpu.registers.r4);
    }

    #[test]
    fn index_register_is_incremented_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0x04;

        cpu.registers.flags = AUTO_INCREMENT_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0x05, cpu.registers.r4);
    }

    #[test]
    fn index_register_overflows_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indexed_absolute(
            &mut program, ENCODING_R3, 0x1234, ENCODING_R4);
        update_program(&mut cpu, program, 0x2000);
        cpu.registers.r4 = 0xFF;

        cpu.registers.flags = AUTO_INCREMENT_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0x00, cpu.registers.r4);
    }

    #[test]
    fn load_indirect_sets_register_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0xD9);
        cpu.memory_controller.borrow_mut().write(0x1235, 0x38);

        cpu.memory_controller.borrow_mut().write(0x38D9, 0xA8);
        execute_instruction(&mut cpu);

        assert_eq!(0xA8, cpu.registers.r4);
    }

    #[test]
    fn load_indirect_address_is_not_incremented_if_flag_is_unset() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0xD9);
        cpu.memory_controller.borrow_mut().write(0x1235, 0x38);

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);

        assert_eq!(0xD9, cpu.memory_controller.borrow().read(0x1234));
        assert_eq!(0x38, cpu.memory_controller.borrow().read(0x1235));
    }

    #[test]
    fn load_indirect_address_is_incremented_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0xD9);
        cpu.memory_controller.borrow_mut().write(0x1235, 0x38);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0xDA, cpu.memory_controller.borrow().read(0x1234));
        assert_eq!(0x38, cpu.memory_controller.borrow().read(0x1235));
    }

    #[test]
    fn load_indirect_address_is_incremented_and_carry_handled_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0xFF);
        cpu.memory_controller.borrow_mut().write(0x1235, 0x38);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0x00, cpu.memory_controller.borrow().read(0x1234));
        assert_eq!(0x39, cpu.memory_controller.borrow().read(0x1235));
    }

    #[test]
    fn load_indirect_address_is_incremented_and_overflows_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.memory_controller.borrow_mut().write(0x1234, 0xFF);
        cpu.memory_controller.borrow_mut().write(0x1235, 0xFF);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0x00, cpu.memory_controller.borrow().read(0x1234));
        assert_eq!(0x00, cpu.memory_controller.borrow().read(0x1235));
    }

    #[test]
    fn add_without_carry_reg_reg_computes_correct_value_when_carry_unset() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.r3 = 0x02;
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x07, cpu.registers.r1);
    }

    #[test]
    fn add_without_carry_reg_reg_computes_correct_value_when_carry_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.r3 = 0x02;
        cpu.registers.flags = CARRY_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0x07, cpu.registers.r1);
    }

    #[test]
    fn add_without_carry_reg_reg_result_overflows_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x49;
        cpu.registers.r2 = 0xC8;
        cpu.registers.r3 = 0x8C;
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x54, cpu.registers.r1);
    }

    #[test]
    fn add_without_carry_reg_reg_sets_and_clears_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.r3 = 0x02;
        cpu.registers.flags = ZERO_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x0;
        cpu.registers.r2 = 0x0;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn add_without_carry_reg_reg_sets_and_clears_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.r3 = 0x02;
        cpu.registers.flags = NEGATIVE_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x5A;
        cpu.registers.r2 = 0x41;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn add_without_carry_reg_reg_sets_and_clears_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.r3 = 0x02;
        cpu.registers.flags = CARRY_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x8C;
        cpu.registers.r2 = 0xC8;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn add_without_carry_reg_reg_sets_and_clears_overflow_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        // clear flag
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        // overflow set when both values > 127
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2);
        // overflow set when both values < 127
        emit_add_without_carry_reg_reg(
            &mut program,
            ENCODING_R2,
            ENCODING_R3,
            ENCODING_R4);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x0;
        cpu.registers.r2 = 0x8C;
        cpu.registers.r3 = 0x19;
        cpu.registers.flags = OVERFLOW_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.overflow_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x8C;
        cpu.registers.r2 = 0xC8;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());


        cpu.registers.r2 = 0x0;
        cpu.registers.r3 = 0x7F;
        cpu.registers.r4 = 0x20;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());
    }

    #[test]
    fn add_without_carry_reg_immediate_computes_correct_value_when_carry_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x02);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.flags = CARRY_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0x07, cpu.registers.r1);
    }

    #[test]
    fn add_without_carry_reg_immediate_overflows_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x8C);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x9A;
        cpu.registers.flags = CARRY_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0x26, cpu.registers.r1);
    }



    #[test]
    fn add_without_carry_reg_immediate_sets_and_clears_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x02);
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            0x00);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.flags = ZERO_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x0;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn add_without_carry_reg_immediate_sets_and_clears_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x02);
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            0x41);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.flags = NEGATIVE_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x5A;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn add_without_carry_reg_immediate_sets_and_clears_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x02);
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            0xC8);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.flags = CARRY_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x8C;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn add_without_carry_reg_immediate_sets_and_clears_overflow_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        // clear flag
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x19);
        // overflow set when both values > 127
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            0xC8);
        // overflow set when both values < 127
        emit_add_without_carry_reg_immediate(
            &mut program,
            ENCODING_R2,
            ENCODING_R3,
            0x20);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x0;
        cpu.registers.r2 = 0x8C;
        cpu.registers.flags = OVERFLOW_FLAG;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.overflow_flag());

        cpu.registers.r4 = 0x0;
        cpu.registers.r3 = 0x8C;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());


        cpu.registers.r2 = 0x0;
        cpu.registers.r3 = 0x7F;

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());
    }

    #[test]
    fn unsigned_multiply_stores_values_correctly_in_registers() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        // clear flag
        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3,
            ENCODING_R4);
        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 5;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 8;
        cpu.registers.r4 = 9;

        execute_instruction(&mut cpu);
        assert_eq!(72, cpu.registers.r2);
        assert_eq!(0, cpu.registers.r1);


        cpu.registers.r1 = 98;
        cpu.registers.r2 = 65;
        cpu.registers.r3 = 8;
        cpu.registers.r4 = 9;

        execute_instruction(&mut cpu);
        // 98*65 = 6370 = 24*256 + 226
        assert_eq!(226, cpu.registers.r3);
        assert_eq!(24, cpu.registers.r4);


        cpu.registers.r1 = 129;
        cpu.registers.r2 = 129;
        cpu.registers.r3 = 8;
        cpu.registers.r4 = 9;
        execute_instruction(&mut cpu);
        // 129*129 = 16641 = 65*256 + 1
        assert_eq!(1, cpu.registers.r3);
        assert_eq!(65, cpu.registers.r4);

        cpu.registers.r1 = 200;
        cpu.registers.r2 = 75;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;
        execute_instruction(&mut cpu);
        // 200*75 = 35230 = 58*256 + 152
        assert_eq!(152, cpu.registers.r3);
        assert_eq!(58, cpu.registers.r4);
    }

    #[test]
    fn unsigned_multiply_sets_zero_flag_if_result_is_zero() {
         let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn unsigned_multiply_unsets_zero_flag_if_result_is_not_zero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 3;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(60, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn unsigned_multiply_set_carry_and_overflow_flags_if_high_byte_is_nonzero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 40;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());
        assert!(cpu.registers.carry_flag());
    }


    #[test]
    fn unsigned_multiply_unsets_carry_and_overflow_flags_if_high_byte_is_zero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = CARRY_FLAG | OVERFLOW_FLAG;
        cpu.registers.r1 = 2;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.overflow_flag());
        assert!(!cpu.registers.carry_flag());
    }

    #[test]
    fn signed_multiply_stores_values_correctly_in_registers() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        // clear flag
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3,
            ENCODING_R4);
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        // should work with small positive numbers
        cpu.registers.r1 = 6;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 8;
        cpu.registers.r4 = 9;

        execute_instruction(&mut cpu);
        assert_eq!(72, cpu.registers.r2);
        assert_eq!(0, cpu.registers.r1);

        // works with small negative numbers
        cpu.registers.r1 = 8u8.wrapping_neg(); // -8
        cpu.registers.r2 = 9;
        cpu.registers.r3 = 5;
        cpu.registers.r4 = 6;

        execute_instruction(&mut cpu);
        // -8*9 = -72
        assert_eq!(72u8.wrapping_neg(), cpu.registers.r3);
        assert_eq!(0xFF, cpu.registers.r4);


        // same as above, but negative number switched
        cpu.registers.r1 = 8;
        cpu.registers.r2 = 9u8.wrapping_neg(); // -9
        cpu.registers.r3 = 5;
        cpu.registers.r4 = 6;

        execute_instruction(&mut cpu);
        // 8*-9 = -72
        assert_eq!(72u8.wrapping_neg(), cpu.registers.r3);
        assert_eq!(0xFF, cpu.registers.r4);

        // works when both numbers negative
        cpu.registers.r1 = 8u8.wrapping_neg(); // -8
        cpu.registers.r2 = 9u8.wrapping_neg(); // -9
        cpu.registers.r3 = 5;
        cpu.registers.r4 = 6;

        execute_instruction(&mut cpu);
        // -8*-9 = 72
        assert_eq!(72, cpu.registers.r3);
        assert_eq!(0x00, cpu.registers.r4);

        // works with larger numbers
        cpu.registers.r1 = 126;
        cpu.registers.r2 = 92u8.wrapping_neg(); // -92
        cpu.registers.r3 = 5;
        cpu.registers.r4 = 6;

        execute_instruction(&mut cpu);
        // 126*-92 = -11592 = 0xD2B8
        assert_eq!(0xB8, cpu.registers.r3);
        assert_eq!(0xD2, cpu.registers.r4);
    }


    #[test]
    fn signed_multiply_sets_zero_flag_if_result_is_zero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn signed_multiply_unsets_zero_flag_if_result_is_not_zero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 3;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(60, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn signed_multiply_sets_negative_flag_if_result_is_negative() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 3u8.wrapping_neg();
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn signed_multiply_unsets_negative_flag_if_result_is_not_negative() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 3;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }

    #[test]
    fn signed_multiply_sets_carry_and_overflow_flags_if_result_does_not_fit_low_byte() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 30;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
        assert!(cpu.registers.overflow_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 30u8.wrapping_neg();
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
        assert!(cpu.registers.overflow_flag());
    }

    #[test]
    fn signed_multiply_unsets_carry_and_overflow_flags_if_result_fits_low_byte() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_signed_multiply_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = OVERFLOW_FLAG | CARRY_FLAG;
        cpu.registers.r1 = 3;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());
        assert!(!cpu.registers.overflow_flag());

        cpu.registers.flags = OVERFLOW_FLAG | CARRY_FLAG;
        cpu.registers.r1 = 3u8.wrapping_neg();
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());
        assert!(!cpu.registers.overflow_flag());
    }
}