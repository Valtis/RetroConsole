
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

/* Set/Clear flag addressing mode */
const IMPLICIT_ADDRESSING: u8 = 0x00;

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

const ADD_WITH_CARRY: u8 = 0x30;
const ADD_WITHOUT_CARRY: u8 = 0x31;

const SIGNED_MULTIPLY: u8 = 0x34;
const UNSIGNED_MULTIPLY: u8 = 0x35;

const SIGNED_DIVIDE: u8 = 0x36;
const UNSIGNED_DIVIDE: u8 = 0x37;

/* Bitwise operations */
const BITWISE_AND: u8 = 0x20;
const BITWISE_OR: u8 = 0x21;
const BITWISE_XOR: u8 = 0x22;
const BITWISE_NOT: u8 = 0x23;

const ARITHMETIC_SHIFT_LEFT: u8 = 0x24;
const ARITHMETIC_SHIFT_RIGHT: u8 = 0x25;

const LOGICAL_SHIFT_RIGHT: u8 = 0x26;
// logical shift right === arithmetic shift right

const ROTATE_LEFT: u8 = 0x27;
const ROTATE_RIGHT: u8 = 0x28;

/* Flag instructions */
const SET_AUTO_INCREMENT_FLAG: u8 = 0x07;
const CLEAR_AUTO_INCREMENT_FLAG: u8 = 0x06;

/* Initial value of the stack pointer */
const SP_INITIAL_VALUE: u16 = 0x200;


/* Interrupt vector table address locations */
const RESET_VECTOR: u16 = 0x00;
const ILLEGAL_OPCODE_VECTOR: u16 = 0x02;
const DIVIDE_ERROR_VECTOR: u16 = 0x04;
const DIVIDE_OVERFLOW_VECTOR: u16 = 0x06;

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
    ClearInterruptFlag,
    SetFaultFlag,
    FetchValue,
    FetchDestSrc,
    FetchLowAddress,
    FetchHighAddress,
    FetchIndirectLowAddress,
    FetchIndirectHighAddress,
    FetchInterruptVectorLowByte,
    FetchInterruptVectorHighByte,
    IncrementAndStoreIndirectLowByte,
    StoreIndirectHighByte,
    PushPCHighByte,
    PushPCLowByte,
    PushStatusFlags,
    ImmedateToRegister,
    AbsoluteToRegister,
    RegisterToAbsolute,
    IndexedAddress,
    AddWithCarryRegister,
    AddWithCarryImmediate,
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
    BeginDivision,
    DivideFetchImmediate,
    DivideCheckDenominator,
    SignedDivideInvertNegativeNumerator,
    SignedDivideInvertNegativeDenominator,
    DivideShiftRemainder,
    DivideUpdateRemainderBit,
    DivideShiftNumerator,
    DivideTestRemainderDenominator,
    DivideMaybeUpdateRemainder,
    DivideMaybeUpdatedQuotient,
    DivideShiftQuotinent,
    SignedDivideMaybeNegateRemainder,
    SignedDivideMaybeNegateQuotinent,
    EndDivision,
    ArithmeticShiftLeftRegister,
    ArithmeticShiftLeftImmediate,
    ArithmeticShiftRightRegister,
    ArithmeticShiftRightImmediate,
    LogicalShiftRightRegister,
    LogicalShiftRightImmediate,
    BitwiseAndRegister,
    BitwiseAndImmediate,
    BitwiseOrRegister,
    BitwiseOrImmediate,
    BitwiseXorRegister,
    BitwiseXorImmediate,
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
    value_register_2: u8,
    dest_src_register: u8,
    address_register: u16,
    indirect_address_register: u16,
    multiply_negate: bool,
    divide_flag: bool,
    negate_remainder: bool,
    negate_quotinent: bool,
}

impl StateMachine {
    fn new() -> StateMachine {
        StateMachine {
            micro_ops: vec![],
            index: 0,
            value_register: 0,
            value_register_2: 0,
            dest_src_register: 0,
            address_register: 0,
            indirect_address_register: 0,
            multiply_negate: false,
            divide_flag: false,
            negate_remainder: false,
            negate_quotinent: false,
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

    fn fault_flag(&self) -> bool {
        self.flags & FAULT_FLAG != 0
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

    fn set_interrupt_flag(&mut self) {
        self.flags |= INTERRUPT_ENABLE_FLAG;
    }

    fn clear_interrupt_flag(&mut self) {
        self.flags &= !INTERRUPT_ENABLE_FLAG;
    }

    fn set_auto_increment_flag(&mut self) {
        self.flags |= AUTO_INCREMENT_FLAG;
    }

    fn clear_autoincrement_flag(&mut self) {
        self.flags &= !AUTO_INCREMENT_FLAG;
    }

    fn set_fault_flag(&mut self) {
        self.flags |= FAULT_FLAG;
    }

    fn clear_fault_flag(&mut self) {
        self.flags &= !FAULT_FLAG;
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


    fn write(&mut self, address: u16, value: u8) {
        self.memory_controller.borrow_mut().write(address, value);
    }

    fn push(&mut self, value: u8) {
        let sp = self.registers.sp;
        self.write(sp, value);
        self.registers.sp -= 1;
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
                STORE_REGISTER => self.decode_str(addressing),
                /* Arithmetic instructions */
                ADD_WITH_CARRY => self.decode_adc(addressing),
                ADD_WITHOUT_CARRY => self.decode_add(addressing),
                SIGNED_MULTIPLY => self.decode_imul(addressing),
                UNSIGNED_MULTIPLY => self.decode_mul(addressing),
                SIGNED_DIVIDE => self.decode_idiv(addressing),
                UNSIGNED_DIVIDE => self.decode_div(addressing),
                /* Bitwise operations */
                ARITHMETIC_SHIFT_LEFT => self.decode_asl(addressing),
                ARITHMETIC_SHIFT_RIGHT => self.decode_asr(addressing),
                LOGICAL_SHIFT_RIGHT => self.decode_lsr(addressing),
                BITWISE_AND => self.decode_and(addressing),
                BITWISE_OR => self.decode_or(addressing),
                BITWISE_XOR => self.decode_xor(addressing),
                /* Status flag instructions  */
                SET_AUTO_INCREMENT_FLAG => self.decode_sai(addressing),
                CLEAR_AUTO_INCREMENT_FLAG => self.decode_cla(addressing),
                _ => self.illegal_opcode(),
            }
        } else {
            self.execute_micro_op();
        }

        println!("{:?}", self.registers);
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

    fn decode_str(&mut self, addressing: u8) {
        match addressing & 0x03 {
            IMMEDIATE_ADDRESSING => {
                self.illegal_opcode();
            },
            ABSOLUTE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchLowAddress);
                self.state.micro_ops.push(MicroOp::FetchHighAddress);
                self.state.micro_ops.push(MicroOp::RegisterToAbsolute);
            },
            INDEXED_ABSOLUTE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchLowAddress);
                self.state.micro_ops.push(MicroOp::FetchHighAddress);
                self.state.micro_ops.push(MicroOp::IndexedAddress);
                self.state.micro_ops.push(MicroOp::RegisterToAbsolute);
            },
            INDIRECT_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchLowAddress);
                self.state.micro_ops.push(MicroOp::FetchHighAddress);
                self.state.micro_ops.push(MicroOp::FetchIndirectLowAddress);
                self.state.micro_ops.push(MicroOp::FetchIndirectHighAddress);
                self.state.micro_ops.push(MicroOp::RegisterToAbsolute);

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

    fn decode_adc(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::AddWithCarryRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(MicroOp::AddWithCarryImmediate);
            },
            _ => self.illegal_opcode(),
        }
    }

    fn decode_add(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::AddWithoutCarryRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(MicroOp::AddWithoutCarryImmediate);
            },
            _ => self.illegal_opcode(),
        }
    }

    fn decode_imul(&mut self, addressing: u8) {
        match addressing & 0x03 {
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
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginMultiply);
                self.state.micro_ops.push(MicroOp::FetchValue);
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

            }
            _ => self.illegal_opcode(),
        }
    }

    fn decode_mul(&mut self, addressing: u8) {
        match addressing & 0x03 {
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
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginMultiply);
                self.state.micro_ops.push(MicroOp::FetchValue);

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

            }
            _ => self.illegal_opcode(),
        }
    }

    fn decode_idiv(&mut self, addressing: u8) {
        match addressing {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginDivision);
                self.state.micro_ops.push(MicroOp::DivideCheckDenominator);
                self.state.micro_ops.push(
                    MicroOp::SignedDivideInvertNegativeNumerator);
                self.state.micro_ops.push(
                    MicroOp::SignedDivideInvertNegativeDenominator);

                for _ in 0..8 {
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideUpdateRemainderBit);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftNumerator);
                    self.state.micro_ops.push(
                        MicroOp::DivideTestRemainderDenominator);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftQuotinent);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdateRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdatedQuotient);
                }
                self.state.micro_ops.push(
                    MicroOp::SignedDivideMaybeNegateRemainder);
                self.state.micro_ops.push(
                    MicroOp::SignedDivideMaybeNegateQuotinent);
                self.state.micro_ops.push(MicroOp::EndDivision);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginDivision);
                self.state.micro_ops.push(MicroOp::DivideFetchImmediate);
                self.state.micro_ops.push(MicroOp::DivideCheckDenominator);
                self.state.micro_ops.push(
                    MicroOp::SignedDivideInvertNegativeNumerator);
                self.state.micro_ops.push(
                    MicroOp::SignedDivideInvertNegativeDenominator);

                for _ in 0..8 {
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideUpdateRemainderBit);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftNumerator);
                    self.state.micro_ops.push(
                        MicroOp::DivideTestRemainderDenominator);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftQuotinent);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdateRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdatedQuotient);
                }
                self.state.micro_ops.push(
                    MicroOp::SignedDivideMaybeNegateRemainder);
                self.state.micro_ops.push(
                    MicroOp::SignedDivideMaybeNegateQuotinent);
                self.state.micro_ops.push(MicroOp::EndDivision);
            },
            _ => self.illegal_opcode(),

        }
    }

    fn decode_div(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {

                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginDivision);
                self.state.micro_ops.push(MicroOp::DivideCheckDenominator);


                for _ in 0..8 {
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideUpdateRemainderBit);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftNumerator);
                    self.state.micro_ops.push(
                        MicroOp::DivideTestRemainderDenominator);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftQuotinent);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdateRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdatedQuotient);
                }

                self.state.micro_ops.push(MicroOp::EndDivision);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BeginDivision);
                self.state.micro_ops.push(MicroOp::DivideFetchImmediate);
                self.state.micro_ops.push(MicroOp::DivideCheckDenominator);

                for _ in 0..8 {
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideUpdateRemainderBit);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftNumerator);
                    self.state.micro_ops.push(
                        MicroOp::DivideTestRemainderDenominator);
                    self.state.micro_ops.push(
                        MicroOp::DivideShiftQuotinent);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdateRemainder);
                    self.state.micro_ops.push(
                        MicroOp::DivideMaybeUpdatedQuotient);
                }

                self.state.micro_ops.push(MicroOp::EndDivision);
            },
            _ => self.illegal_opcode(),
        }
    }

    fn decode_asl(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(
                    MicroOp::ArithmeticShiftLeftRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(
                    MicroOp::ArithmeticShiftLeftImmediate);
            },
            _ => self.illegal_opcode(),
        }
    }

    fn decode_asr(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(
                    MicroOp::ArithmeticShiftRightRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(
                    MicroOp::ArithmeticShiftRightImmediate);
            },
            _ => self.illegal_opcode(),
        }
    }

    fn decode_lsr(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(
                    MicroOp::LogicalShiftRightRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(
                    MicroOp::LogicalShiftRightImmediate);
            },
            _ => self.illegal_opcode(),
        }
    }

    fn decode_and(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BitwiseAndRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(MicroOp::BitwiseAndImmediate);
            },
            _ =>  self.illegal_opcode(),
        }
    }

    fn decode_or(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BitwiseOrRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(MicroOp::BitwiseOrImmediate);
            },
            _ =>  self.illegal_opcode(),
        }
    }
    fn decode_xor(&mut self, addressing: u8) {
        match addressing & 0x03 {
            REGISTER_REGISTER_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::BitwiseXorRegister);
            },
            REGISTER_IMMEDIATE_ADDRESSING => {
                self.state.micro_ops.push(MicroOp::FetchDestSrc);
                self.state.micro_ops.push(MicroOp::FetchValue);
                self.state.micro_ops.push(MicroOp::BitwiseXorImmediate);
            },
            _ =>  self.illegal_opcode(),
        }
    }

    fn decode_sai(&mut self, addressing: u8) {
        match addressing & 0x03 {
            IMPLICIT_ADDRESSING =>
                self.state.micro_ops.push(MicroOp::SetAutoIncrementFlag),
            _ =>  self.illegal_opcode(),
        }
    }

    fn decode_cla(&mut self, addressing: u8) {
        match addressing & 0x03 {
            IMPLICIT_ADDRESSING =>
                self.state.micro_ops.push(MicroOp::ClearAutoIncrementFlag),
            _ =>  self.illegal_opcode(),
        }
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
            MicroOp::ClearInterruptFlag => {
                self.registers.clear_interrupt_flag();
            },
            MicroOp::SetFaultFlag => {
                self.registers.set_fault_flag();
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
            MicroOp::FetchInterruptVectorLowByte => {
                self.registers.pc =
                    self.read(self.state.address_register) as u16;
                self.state.address_register += 1;
            },
            MicroOp::FetchInterruptVectorHighByte => {
                self.registers.pc |=
                    (self.read(self.state.address_register) as u16)
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
            MicroOp::PushPCHighByte => {
                let byte = (self.registers.pc >> 8) as u8;
                self.push(byte);
            },
            MicroOp::PushPCLowByte => {
                let byte = self.registers.pc as u8;
                self.push(byte);
            },
            MicroOp::PushStatusFlags => {
                let flags = self.registers.flags;
                self.push(flags);
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
            MicroOp::RegisterToAbsolute => {
                let src = self.state.dest_src_register & 0x03;
                let value = self.load_register(src);
                let address = self.state.address_register;
                self.write(address, value);
            }
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
            MicroOp::AddWithCarryRegister => {

                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);
                let carry = if self.registers.carry_flag() {
                    1
                } else {
                    0
                };

                let result = src1val as u16 + src2val as u16 + carry;

                self.registers.set_carry_flag_on_value(result);
                self.registers.set_overflow_flag_on_value(src1val, src2val, result);

                self.store_register(destination, result as u8);
            },
            MicroOp::AddWithCarryImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.state.value_register;
                let carry = if self.registers.carry_flag() {
                    1
                } else {
                    0
                };

                let result = src1val as u16 + src2val as u16 + carry;

                self.registers.set_carry_flag_on_value(result);
                self.registers.set_overflow_flag_on_value(
                    src1val, src2val, result);

                self.store_register(destination, result as u8);
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
                self.registers.set_overflow_flag_on_value(
                    src1val, src2val, result);

                self.store_register(destination, result as u8);
            },
            MicroOp::BeginMultiply => {
                let src1 = (self.state.dest_src_register >> 6) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

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
            MicroOp::BeginDivision => {
                let denom_reg = (self.state.dest_src_register >> 6) & 0x03;
                let numer_reg = (self.state.dest_src_register >> 4) & 0x03;

                let denom = self.load_register(denom_reg);
                let numer = self.load_register(numer_reg);

                // quotinent
                self.state.value_register = 0;
                // remainder
                self.state.value_register_2 = 0;

                // reuse address register to store numerator
                self.state.address_register = numer as u16;
                // reuse reg to represent denominator
                self.state.indirect_address_register = denom as u16;
                self.state.divide_flag = false;
                self.state.negate_quotinent = false;
                self.state.negate_remainder = false;
            },
            MicroOp::DivideFetchImmediate => {
                self.state.indirect_address_register = self.read_pc() as u16;
            }
            MicroOp::DivideCheckDenominator => {
                 if self.state.indirect_address_register == 0 {
                    self.divide_error();
                }
            },
            MicroOp::SignedDivideInvertNegativeDenominator => {
                if self.state.indirect_address_register > 127 {
                    self.state.indirect_address_register =
                        self.state.indirect_address_register.wrapping_neg()
                        & 0xFF;;
                    self.state.negate_quotinent = !self.state.negate_quotinent;
                }
            },
            MicroOp::SignedDivideInvertNegativeNumerator => {
                if self.state.address_register > 127 {
                    self.state.address_register =
                        self.state.address_register.wrapping_neg() & 0xFF;
                    self.state.negate_quotinent = !self.state.negate_quotinent;
                    self.state.negate_remainder = true;
                }
            },
            MicroOp::DivideShiftRemainder => {
                self.state.value_register_2 <<= 1;
            },
            MicroOp::DivideUpdateRemainderBit => {
                self.state.value_register_2 |=
                    ((self.state.address_register >> 7) as u8) & 0x01;
            },
            MicroOp::DivideShiftNumerator => {
                self.state.address_register <<= 1;
            },
            MicroOp::DivideTestRemainderDenominator => {
                self.state.divide_flag =
                    self.state.value_register_2
                    >= self.state.indirect_address_register as u8;
            },
            MicroOp::DivideShiftQuotinent => {
                self.state.value_register <<= 1;
            },
            MicroOp::DivideMaybeUpdateRemainder => {
                if self.state.divide_flag {
                   self.state.value_register_2 -=
                        self.state.indirect_address_register as u8;
                }
            },
            MicroOp::DivideMaybeUpdatedQuotient => {
                if self.state.divide_flag {
                    self.state.value_register |= 1;
                }
            },
            MicroOp::SignedDivideMaybeNegateQuotinent => {
                if self.state.negate_quotinent {
                    self.state.value_register =
                        self.state.value_register.wrapping_neg();
                }

                let is_negative = self.state.value_register & 0x80 != 0;

                // negate flag is set if the result should be negative,
                // and result is negative only if negate flag is set.
                // if result is negative without the flag being set,
                // overflow has occured, e.g. we're trying to represent +128
                // with signed number
                if is_negative && !self.state.negate_quotinent {
                    self.divide_overflow();
                }
            },
            MicroOp::SignedDivideMaybeNegateRemainder => {
                if self.state.negate_remainder {
                    self.state.value_register_2 =
                        self.state.value_register_2.wrapping_neg();
                }
            },
            MicroOp::EndDivision => {
                let remainder_reg = self.state.dest_src_register & 0x03;
                let quotinent_reg = (self.state.dest_src_register >> 2) & 0x03;

                let q = self.state.value_register;
                let r = self.state.value_register_2;
                self.store_register(quotinent_reg, q);
                self.store_register(remainder_reg, r);
            },
            MicroOp::ArithmeticShiftLeftRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);
                let result = (src1val as u16) << ((src2val & 0x07) as u16);
                self.registers.set_carry_flag_on_value(result);

                self.store_register(destination, result as u8);
            },
            MicroOp::ArithmeticShiftLeftImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.state.value_register;
                let result = (src1val as u16) << ((src2val & 0x07) as u16);
                self.registers.set_carry_flag_on_value(result);

                self.store_register(destination, result as u8);
            },
            MicroOp::ArithmeticShiftRightRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = (self.load_register(src1) as i16) << 8;
                let src2val = self.load_register(src2) as i16;
                let result = src1val >> (src2val & 0x07);

                if result & 0x80 != 0 {
                    self.registers.set_carry_flag();
                } else {
                    self.registers.clear_carry_flag();
                }

                self.store_register(destination, (result >> 8) as u8);
            },
            MicroOp::ArithmeticShiftRightImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = (self.load_register(src1) as i16) << 8;
                let src2val = self.state.value_register as i16;
                let result = src1val >> (src2val & 0x07);

                if result & 0x80 != 0 {
                    self.registers.set_carry_flag();
                } else {
                    self.registers.clear_carry_flag();
                }

                self.store_register(destination, (result >> 8) as u8);
            },
            MicroOp::LogicalShiftRightRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = (self.load_register(src1) as u16) << 8;
                let src2val = self.load_register(src2) as u16;
                let result = src1val >> (src2val & 0x07);

                if result & 0x80 != 0 {
                    self.registers.set_carry_flag();
                } else {
                    self.registers.clear_carry_flag();
                }

                self.store_register(destination, (result >> 8) as u8);
            },
            MicroOp::LogicalShiftRightImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = (self.load_register(src1) as u16) << 8;
                let src2val = self.state.value_register as u16;
                let result = src1val >> (src2val & 0x07);

                if result & 0x80 != 0 {
                    self.registers.set_carry_flag();
                } else {
                    self.registers.clear_carry_flag();
                }

                self.store_register(destination, (result >> 8) as u8);
            },
            MicroOp::BitwiseAndRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);
                let result = src1val & src2val;

                self.store_register(destination, result);
            },
            MicroOp::BitwiseAndImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.state.value_register;
                let result = src1val & src2val;

                self.store_register(destination, result);
            },
            MicroOp::BitwiseOrRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);
                let result = src1val | src2val;

                self.store_register(destination, result);
            },
            MicroOp::BitwiseOrImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.state.value_register;
                let result = src1val | src2val;

                self.store_register(destination, result);
            },
            MicroOp::BitwiseXorRegister => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;
                let src2 = (self.state.dest_src_register >> 4) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.load_register(src2);
                let result = src1val ^ src2val;

                self.store_register(destination, result);
            },
            MicroOp::BitwiseXorImmediate => {
                let destination = self.state.dest_src_register & 0x03;
                let src1 = (self.state.dest_src_register >> 2) & 0x03;

                let src1val = self.load_register(src1);
                let src2val = self.state.value_register;
                let result = src1val ^ src2val;

                self.store_register(destination, result);
            },
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


    fn illegal_opcode(&mut self) {
        self.fault(ILLEGAL_OPCODE_VECTOR);
    }

    fn divide_error(&mut self) {
        self.fault(DIVIDE_ERROR_VECTOR);
    }

    fn divide_overflow(&mut self) {
        self.fault(DIVIDE_OVERFLOW_VECTOR);
    }


    fn fault(&mut self, vector: u16) {
        self.start_interrupt();
        self.state.address_register = vector;
        self.state.micro_ops.push(MicroOp::SetFaultFlag);
        self.state.index = 0;
    }

    fn start_interrupt(&mut self) {
        self.state.micro_ops.clear();
        self.state.micro_ops.push(MicroOp::PushPCHighByte);
        self.state.micro_ops.push(MicroOp::PushPCLowByte);
        self.state.micro_ops.push(MicroOp::PushStatusFlags);
        self.state.micro_ops.push(MicroOp::ClearInterruptFlag);
        self.state.micro_ops.push(MicroOp::FetchInterruptVectorLowByte);
        self.state.micro_ops.push(MicroOp::FetchInterruptVectorHighByte);
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


    // Invalid opcode, exists for testing.
    fn emit_store_immediate(
        opcodes: &mut Vec<u8>) {
        opcodes.push((STORE_REGISTER << 2) | IMMEDIATE_ADDRESSING);
    }

    fn emit_store_absolute(
        opcodes: &mut Vec<u8>,
        source: u8,
        address: u16) {
        opcodes.push((STORE_REGISTER << 2) | ABSOLUTE_ADDRESSING);
        opcodes.push(source & 0x03);
        opcodes.push(address as u8);
        opcodes.push((address >> 8)  as u8);
    }

    fn emit_store_indexed_absolute(
        opcodes: &mut Vec<u8>,
        source: u8,
        address: u16,
        index: u8) {
        opcodes.push((STORE_REGISTER << 2) | INDEXED_ABSOLUTE_ADDRESSING);
        opcodes.push(source & 0x03 | ((index & 0x03) << 2));
        opcodes.push(address as u8);
        opcodes.push((address >> 8)  as u8);
    }

    fn emit_store_indirect(
        opcodes: &mut Vec<u8>,
        source: u8,
        address: u16) {
        opcodes.push((STORE_REGISTER << 2) | INDIRECT_ADDRESSING);
        opcodes.push(source & 0x03);
        opcodes.push(address as u8);
        opcodes.push((address >> 8)  as u8);
    }

    fn emit_add_with_carry_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((ADD_WITH_CARRY << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_add_with_carry_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((ADD_WITH_CARRY << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
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

    fn emit_unsigned_multiply_reg_immediate(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        immediate: u8) {

        opcodes.push((UNSIGNED_MULTIPLY << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) <<  4));
        opcodes.push(immediate);
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

    fn emit_signed_multiply_reg_immediate(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        immediate: u8) {

        opcodes.push((SIGNED_MULTIPLY << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) <<  4));
        opcodes.push(immediate);
    }

    fn emit_unsigned_division_reg_reg(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((UNSIGNED_DIVIDE << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) << 4) |
            ((src_2 & 0x03) << 6));
    }

    fn emit_unsigned_division_reg_immediate(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((UNSIGNED_DIVIDE << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) << 4));
        opcodes.push(immediate);
    }

    fn emit_signed_division_reg_reg(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((SIGNED_DIVIDE << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) << 4) |
            ((src_2 & 0x03) << 6));
    }

    fn emit_signed_division_reg_immediate(
        opcodes: &mut Vec<u8>,
        high_reg: u8,
        low_reg: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((SIGNED_DIVIDE << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (high_reg & 0x03) |
            ((low_reg & 0x03) << 2) |
            ((src_1 & 0x03) << 4));
        opcodes.push(immediate);
    }

    fn emit_arithmetic_shift_left_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push(
            (ARITHMETIC_SHIFT_LEFT << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_arithmetic_shift_left_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push(
            (ARITHMETIC_SHIFT_LEFT << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_arithmetic_shift_right_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push(
            (ARITHMETIC_SHIFT_RIGHT << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_arithmetic_shift_right_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push(
            (ARITHMETIC_SHIFT_RIGHT << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_logical_shift_right_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push(
            (LOGICAL_SHIFT_RIGHT << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_logical_shift_right_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push(
            (LOGICAL_SHIFT_RIGHT << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_rotate_left_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((ROTATE_LEFT << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_rotate_left_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((ROTATE_LEFT << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_rotate_right_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((ROTATE_RIGHT << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_rotate_right_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((ROTATE_RIGHT << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_bitwise_and_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((BITWISE_AND << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_bitwise_and_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((BITWISE_AND << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_bitwise_or_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((BITWISE_OR << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_bitwise_or_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((BITWISE_OR << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
    }

    fn emit_bitwise_xor_reg_reg(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        src_2: u8) {
        opcodes.push((BITWISE_XOR << 2) | REGISTER_REGISTER_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2) |
            ((src_2 & 0x03) << 4));
    }

    fn emit_bitwise_xor_reg_immediate(
        opcodes: &mut Vec<u8>,
        destination: u8,
        src_1: u8,
        immediate: u8) {
        opcodes.push((BITWISE_XOR << 2) | REGISTER_IMMEDIATE_ADDRESSING);
        opcodes.push(
            (destination & 0x03) |
            ((src_1 & 0x03) << 2));
        opcodes.push(immediate);
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

        cpu.write(0x1234, 0xFF);
        execute_instruction(&mut cpu);
        assert_eq!(0xFF, cpu.registers.r2);
    }

    #[test]
    fn load_absolute_sets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_absolute(&mut program, ENCODING_R2, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0);
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

        cpu.write(0x1234, 0x23);
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

        cpu.write(0x1234, 0x86);
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

        cpu.write(0x1234, 0x70);
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
        cpu.write(0x1238, 0xFA);
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

        cpu.write(0x1238, 0);
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

        cpu.write(0x1238, 23);
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

        cpu.write(0x1238, 0x80);
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

        cpu.write(0x1238, 0x7F);
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

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.write(0x38D9, 0xA8);
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

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);

        assert_eq!(0xD9, cpu.read(0x1234));
        assert_eq!(0x38, cpu.read(0x1235));
    }

    #[test]
    fn load_indirect_address_is_incremented_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0xDA, cpu.read(0x1234));
        assert_eq!(0x38, cpu.read(0x1235));
    }

    #[test]
    fn load_indirect_address_is_incremented_and_carry_handled_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xFF);
        cpu.write(0x1235, 0x38);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0x00, cpu.read(0x1234));
        assert_eq!(0x39, cpu.read(0x1235));
    }

    #[test]
    fn load_indirect_address_is_incremented_and_overflows_if_flag_is_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_load_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xFF);
        cpu.write(0x1235, 0xFF);

        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0x00, cpu.read(0x1234));
        assert_eq!(0x00, cpu.read(0x1235));
    }

    #[test]
    fn store_immediate_generates_invalid_opcode_fault() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_immediate(
            &mut program);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(ILLEGAL_OPCODE_VECTOR, 0x40);
        cpu.write(ILLEGAL_OPCODE_VECTOR + 1, 0xF0);

        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0xF040, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn store_absolute_stores_value_into_memory() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_absolute(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r4 = 0x30;

        execute_instruction(&mut cpu);
        assert_eq!(0x30, cpu.read(0x1234));
     }

    #[test]
    fn store_absolute_does_not_modify_flags() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_absolute(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG | CARRY_FLAG | NEGATIVE_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(
            ZERO_FLAG | CARRY_FLAG | NEGATIVE_FLAG,
            cpu.registers.flags);
    }

    #[test]
    fn store_indexed_absolute_stores_value_into_memory() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indexed_absolute(
            &mut program,
            ENCODING_R4,
            0x1234,
            ENCODING_R2);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r4 = 0xFE;
        cpu.registers.r2 = 0xAB;

        execute_instruction(&mut cpu);
        assert_eq!(0xFE, cpu.read(0x1234 + 0xAB));
    }

    #[test]
    fn store_indexed_absolute_does_not_modify_flags() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indexed_absolute(
            &mut program,
            ENCODING_R4,
            0x1234,
            ENCODING_R2);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r4 = 0xFE;
        cpu.registers.r2 = 0xAB;
        cpu.registers.flags = ZERO_FLAG | CARRY_FLAG | NEGATIVE_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(
            ZERO_FLAG | CARRY_FLAG | NEGATIVE_FLAG,
            cpu.registers.flags);
    }

    #[test]
    fn store_indexed_absolute_does_not_increment_index_reg_if_flag_clear() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indexed_absolute(
            &mut program,
            ENCODING_R4,
            0x1234,
            ENCODING_R2);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r4 = 0xFE;
        cpu.registers.r2 = 0xAB;
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0xAB, cpu.registers.r2);
    }

    #[test]
    fn store_indexed_absolute_increments_index_reg_if_flag_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indexed_absolute(
            &mut program,
            ENCODING_R4,
            0x1234,
            ENCODING_R2);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r4 = 0xFE;
        cpu.registers.r2 = 0xAB;
        cpu.registers.flags = AUTO_INCREMENT_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0xAC, cpu.registers.r2);
    }

    #[test]
    fn store_indirect_stores_value_into_memory() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.write(0x38D9, 0x00);

        cpu.registers.r4 = 0x23;
        execute_instruction(&mut cpu);

        assert_eq!(0x23, cpu.read(0x38D9));
    }

    #[test]
    fn store_indirect_does_not_modify_flags() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.write(0x38D9, 0x00);

        cpu.registers.r4 = 0x23;
        cpu.registers.flags = CARRY_FLAG | ZERO_FLAG | NEGATIVE_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(
            CARRY_FLAG | ZERO_FLAG | NEGATIVE_FLAG,
            cpu.registers.flags);
    }

    #[test]
    fn store_indirect_does_not_increment_address_if_flag_is_cleared() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.write(0x38D9, 0x00);

        cpu.registers.r4 = 0x23;
        cpu.registers.flags = 0;
        execute_instruction(&mut cpu);

        assert_eq!(0xD9, cpu.read(0x1234));
        assert_eq!(0x38, cpu.read(0x1235));
    }

    #[test]
    fn store_indirect_increments_address_if_flag_is_set() {
                let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_store_indirect(
            &mut program, ENCODING_R4, 0x1234);
        update_program(&mut cpu, program, 0x2000);

        cpu.write(0x1234, 0xD9);
        cpu.write(0x1235, 0x38);

        cpu.write(0x38D9, 0x00);

        cpu.registers.r4 = 0x23;
        cpu.registers.flags = AUTO_INCREMENT_FLAG;
        execute_instruction(&mut cpu);

        assert_eq!(0xDA, cpu.read(0x1234));
        assert_eq!(0x38, cpu.read(0x1235));
    }

    #[test]
    fn add_with_carry_reg_reg_computes_correct_value_when_carry_unset() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_reg(
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
    fn add_with_carry_reg_reg_computes_correct_value_when_carry_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_reg(
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
        assert_eq!(0x08, cpu.registers.r1);
    }

    #[test]
    fn add_with_carry_reg_reg_sets_and_unsets_overflow_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 0x80;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());

        cpu.registers.flags = OVERFLOW_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.overflow_flag());
    }

    #[test]
    fn add_with_carry_reg_reg_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());
    }

    #[test]
    fn add_with_carry_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn add_with_carry_reg_reg_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);
        emit_add_with_carry_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            ENCODING_R3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;
        cpu.registers.r3 = 0x01;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }
    #[test]
    fn add_with_carry_reg_immediate_computes_correct_value_when_carry_unset() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x02);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.flags = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x07, cpu.registers.r1);
    }

    #[test]
    fn add_with_carry_reg_immediate_computes_correct_value_when_carry_set() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x02);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x05;
        cpu.registers.flags = CARRY_FLAG;

        execute_instruction(&mut cpu);
        assert_eq!(0x08, cpu.registers.r1);
    }

    #[test]
    fn add_with_carry_reg_immediate_sets_and_unsets_overflow_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x80);
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0xFF;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.overflow_flag());

        cpu.registers.flags = OVERFLOW_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.overflow_flag());
    }

    #[test]
    fn add_with_carry_reg_immediate_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0xFF;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());
    }

    #[test]
    fn add_with_carry_reg_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0xFF;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn add_with_carry_reg_immediate_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);
        emit_add_with_carry_reg_immediate(
            &mut program,
            ENCODING_R1,
            ENCODING_R2,
            0x01);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x7F;
        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 0x40;
        cpu.registers.r2 = 0x20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
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
    fn unsigned_multiply_reg_immediate_calculates_values_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_multiply_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            40);
        emit_unsigned_multiply_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            200);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(200, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(232, cpu.registers.r3);
        assert_eq!(3, cpu.registers.r4);
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


    #[test]
    fn signed_multiply_reg_immediate_calculates_values_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_multiply_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            40);
        emit_signed_multiply_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            127);
        emit_signed_multiply_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            2u8.wrapping_neg());
        emit_signed_multiply_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            100u8.wrapping_neg());


        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(200, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(123, cpu.registers.r3);
        assert_eq!(2, cpu.registers.r4);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(10u8.wrapping_neg(), cpu.registers.r3);
        assert_eq!(0xFF, cpu.registers.r4);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 5;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0x0C, cpu.registers.r3);
        assert_eq!(0xFE, cpu.registers.r4);
    }

    #[test]
    fn unsigned_division_calculates_even_division_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 5;
        cpu.registers.r2 = 200;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(40, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
    }


    #[test]
    fn unsigned_division_reg_reg_calculates_uneven_division_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 7;
        cpu.registers.r2 = 200;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(28, cpu.registers.r3);
        assert_eq!(4, cpu.registers.r4);
    }

    #[test]
    fn unsigned_division_reg_reg_by_zero_generates_fault() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0;
        cpu.registers.r2 = 200;

        cpu.write(DIVIDE_ERROR_VECTOR, 0x2F);
        cpu.write(DIVIDE_ERROR_VECTOR+1, 0x9A);

        execute_instruction(&mut cpu);
        assert_eq!(0x9A2F, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn unsigned_division_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_unsigned_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 0;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(cpu.registers.zero_flag());


        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 2;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(2, cpu.registers.r4);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn unsigned_division_reg_immediate_calculates_correct_value() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            6);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 16;

        execute_instruction(&mut cpu);
        assert_eq!(2, cpu.registers.r3);
        assert_eq!(4, cpu.registers.r4);
    }

    #[test]
    fn unsigned_division_reg_immediate_generates_fault_with_div_zero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            0);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 16;


        cpu.write(DIVIDE_ERROR_VECTOR, 0x2F);
        cpu.write(DIVIDE_ERROR_VECTOR+1, 0x9A);

        execute_instruction(&mut cpu);
        assert_eq!(0x9A2F, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn unsigned_division_reg_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_unsigned_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            7);
        emit_unsigned_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            7);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(cpu.registers.zero_flag());


        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 2;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(2, cpu.registers.r4);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn signed_division_reg_reg_calculates_positive_values_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 7;
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(2, cpu.registers.r3);
        assert_eq!(6, cpu.registers.r4);
    }

    #[test]
    fn signed_division_reg_reg_calculates_positive_negative_values_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 7u8.wrapping_neg();
        cpu.registers.r2 = 20;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(2u8.wrapping_neg(), cpu.registers.r3);
        assert_eq!(6, cpu.registers.r4);
    }

    #[test]
    fn signed_division_reg_reg_calculates_negative_positive_values_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 7;
        cpu.registers.r2 = 20u8.wrapping_neg();
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(2u8.wrapping_neg(), cpu.registers.r3);
        assert_eq!(6u8.wrapping_neg(), cpu.registers.r4);
    }

    #[test]
    fn signed_division_reg_reg_calculates_negative_negative_values_correctly() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 7u8.wrapping_neg();
        cpu.registers.r2 = 20u8.wrapping_neg();
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(2, cpu.registers.r3);
        assert_eq!(6u8.wrapping_neg(), cpu.registers.r4);
    }

    #[test]
    fn signed_division_reg_reg_by_zero_generates_fault() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0;
        cpu.registers.r2 = 200;

        cpu.write(DIVIDE_ERROR_VECTOR, 0x2F);
        cpu.write(DIVIDE_ERROR_VECTOR+1, 0x9A);

        execute_instruction(&mut cpu);
        assert_eq!(0x9A2F, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn signed_division_reg_reg_unrepresentable_div_generates_fault() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        // -128/-1 = 128, but signed 8-bit range is [-128, 127] --> invalid
        cpu.registers.r1 = 1u8.wrapping_neg();
        cpu.registers.r2 = 128u8.wrapping_neg();

        cpu.write(DIVIDE_OVERFLOW_VECTOR, 0x2F);
        cpu.write(DIVIDE_OVERFLOW_VECTOR+1, 0x9A);

        execute_instruction(&mut cpu);
        assert_eq!(0x9A2F, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn signed_division_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_signed_division_reg_reg(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 0;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(cpu.registers.zero_flag());


        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 2u8.wrapping_neg();
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(2u8.wrapping_neg(), cpu.registers.r4);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn signed_division_reg_immediate_calculates_correct_value() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            6u8.wrapping_neg());

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 16;

        execute_instruction(&mut cpu);
        assert_eq!(2u8.wrapping_neg(), cpu.registers.r3);
        assert_eq!(4, cpu.registers.r4);
    }

    #[test]
    fn signed_division_reg_immediate_generates_fault_with_div_zero() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            0);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 16;

        cpu.write(DIVIDE_ERROR_VECTOR, 0x2F);
        cpu.write(DIVIDE_ERROR_VECTOR+1, 0x9A);

        execute_instruction(&mut cpu);
        assert_eq!(0x9A2F, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn signed_division_reg_immediate_faults_with_unrepresentable_result() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            1u8.wrapping_neg());

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 128u8.wrapping_neg();

        cpu.write(DIVIDE_OVERFLOW_VECTOR, 0x2F);
        cpu.write(DIVIDE_OVERFLOW_VECTOR+1, 0x9A);

        execute_instruction(&mut cpu);
        assert_eq!(0x9A2F, cpu.registers.pc);
        assert!(cpu.registers.fault_flag());
    }

    #[test]
    fn signed_division_reg_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_signed_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            7);
        emit_signed_division_reg_immediate(
            &mut program,
            ENCODING_R4,
            ENCODING_R3,
            ENCODING_R2,
            7);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(0, cpu.registers.r4);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 2;
        cpu.registers.r3 = 9;
        cpu.registers.r4 = 8;

        execute_instruction(&mut cpu);
        assert_eq!(0, cpu.registers.r3);
        assert_eq!(2, cpu.registers.r4);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn arithmetic_shift_left_reg_reg_stores_correct_value_in_dst_reg() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 3;
        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(144, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_left_reg_reg_large_shift_wraps_around() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 8;
        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        // shift amount mod 8: 8 mod 8 = 0
        execute_instruction(&mut cpu);
        assert_eq!(18, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_left_reg_reg_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 8; // effectively 0 due to modulo
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 0x02;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn arithmetic_shift_left_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x80;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn arithmetic_shift_left_reg_reg_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_left_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x30;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn arithmetic_shift_left_reg_immediate_stores_correct_value_in_dst_reg() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(18 << 3, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_left_reg_immediate_large_shift_wraps_around() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            8);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        // shift amount mod 8: 8 mod 8 = 0
        execute_instruction(&mut cpu);
        assert_eq!(18, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_left_reg_immediate_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            8);
        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            7);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0x02;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn arithmetic_shift_left_reg_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x80;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn arithmetic_shift_left_reg_immediate_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_left_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r2 = 0x30;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn arithmetic_shift_right_reg_reg_stores_correct_value_in_dst_reg() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 3;
        cpu.registers.r2 = 0xF4;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0xFE, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_right_reg_reg_large_shift_wraps_around() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 8;
        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        // shift amount mod 8: 8 mod 8 = 0
        execute_instruction(&mut cpu);
        assert_eq!(18, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_right_reg_reg_preserves_sign_bit() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);


        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 0xAF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0xEB, cpu.registers.r3);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x1F, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_right_reg_reg_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x74;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 8; // effectively 0 due to modulo
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 0x40;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn arithmetic_shift_right_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn arithmetic_shift_right_reg_reg_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_arithmetic_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x30;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn arithmetic_shift_right_reg_immediate_stores_correct_value_in_dst_reg() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xF4;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0xFE, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_right_reg_immediate_large_shift_wraps_around() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            8);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        // shift amount mod 8: 8 mod 8 = 0
        execute_instruction(&mut cpu);
        assert_eq!(18, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_right_reg_immediate_preserves_sign_bit() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            2);
        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            2);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xAF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0xEB, cpu.registers.r3);

        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x1F, cpu.registers.r3);
    }

    #[test]
    fn arithmetic_shift_right_reg_immediate_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            8);
        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            7);

       update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0x74;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0x40;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn arithmetic_shift_right_reg_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn arithmetic_shift_right_reg_immediate_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_arithmetic_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r2 = 0x30;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }


    #[test]
    fn logical_shift_right_reg_reg_stores_correct_value_in_dst_reg() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 3;
        cpu.registers.r2 = 0xF4;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x1E, cpu.registers.r3);
    }

    #[test]
    fn logical_shift_right_reg_reg_large_shift_wraps_around() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 16;
        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        // shift amount mod 8: 16 mod 8 = 0
        execute_instruction(&mut cpu);
        assert_eq!(18, cpu.registers.r3);
    }

    #[test]
    fn logical_shift_right_reg_reg_zero_fills_from_left() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);


        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 0xAF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x2B, cpu.registers.r3);

        cpu.registers.r1 = 2;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x1F, cpu.registers.r3);
    }

    #[test]
    fn logical_shift_right_reg_reg_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x74;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 24; // effectively 0 due to modulo
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r1 = 7;
        cpu.registers.r2 = 0x40;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn logical_shift_right_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn logical_shift_right_reg_reg_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 1;
        cpu.registers.r2 = 0x30;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }

    #[test]
    fn logical_shift_right_reg_immediate_stores_correct_value_in_dst_reg() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            3);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xF4;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x1E, cpu.registers.r3);
    }

    #[test]
    fn logical_shift_right_reg_immediate_large_shift_wraps_around() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            32);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 18;
        cpu.registers.r3 = 20;

        // shift amount mod 8: 8 mod 8 = 0
        execute_instruction(&mut cpu);
        assert_eq!(18, cpu.registers.r3);
    }

    #[test]
    fn logical_shift_right_reg_immediate_zero_fills_from_left() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            2);
        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            2);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xAF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x2B, cpu.registers.r3);

        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert_eq!(0x1F, cpu.registers.r3);
    }

    #[test]
    fn logical_shift_right_reg_immediate_sets_and_unsets_carry_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            8);
        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            7);

       update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0x74;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0xFF;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.carry_flag());

        cpu.registers.flags = CARRY_FLAG;
        cpu.registers.r2 = 0x40;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.carry_flag());
    }

    #[test]
    fn logical_shift_right_reg_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);
        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x01;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 0x7F;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());
    }

    #[test]
    fn logical_shift_right_reg_immediate_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_logical_shift_right_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r2 = 0x30;
        cpu.registers.r3 = 20;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());
    }

    #[test]
    fn bitwise_and_reg_reg_stores_correct_result_in_destination_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_and_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0xAF;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x8E, cpu.registers.r3);
    }

    #[test]
    fn bitwise_and_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_and_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_bitwise_and_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 0xAF;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x10;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn bitwise_and_reg_reg_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_and_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_bitwise_and_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 0x0F;
        cpu.registers.r2 = 0x0E;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0xF0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }


    #[test]
    fn bitwise_and_immediate_stores_correct_result_in_destination_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_and_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xAF);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x8E, cpu.registers.r3);
    }

    #[test]
    fn bitwise_and_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_and_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xAF);
        emit_bitwise_and_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x10);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn bitwise_and_immediate_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_and_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x0F);
        emit_bitwise_and_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xF0);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r2 = 0x0E;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn bitwise_or_reg_reg_stores_correct_result_in_destination_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_or_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0xAF;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0xEF, cpu.registers.r3);
    }

    #[test]
    fn bitwise_or_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_or_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_bitwise_or_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 0xAF;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x00;
        cpu.registers.r2 = 0x00;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn bitwise_or_reg_reg_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_or_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_bitwise_or_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 0x0F;
        cpu.registers.r2 = 0x0E;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0xF0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn bitwise_or_immediate_stores_correct_result_in_destination_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_or_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xAF);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0xEF, cpu.registers.r3);
    }

    #[test]
    fn bitwise_or_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_or_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xAF);
        emit_bitwise_or_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x00);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x00;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn bitwise_or_immediate_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_or_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x0F);
        emit_bitwise_or_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xF0);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r2 = 0x0E;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn bitwise_xor_reg_reg_stores_correct_result_in_destination_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0xAF;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x61, cpu.registers.r3);
    }

    #[test]
    fn bitwise_xor_reg_reg_stores_with_itself_clears_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_reg(
            &mut program,
            ENCODING_R1,
            ENCODING_R1,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r1 = 0xAF;

        execute_instruction(&mut cpu);
        assert_eq!(0x00, cpu.registers.r1);
    }

    #[test]
    fn bitwise_xor_reg_reg_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_bitwise_xor_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r1 = 0xAF;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x8F;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn bitwise_xor_reg_reg_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);
        emit_bitwise_xor_reg_reg(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            ENCODING_R1);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r1 = 0x8F;
        cpu.registers.r2 = 0x8E;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r1 = 0x70;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }

    #[test]
    fn bitwise_xor_immediate_stores_correct_result_in_destination_register() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xAF);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert_eq!(0x61, cpu.registers.r3);
    }

    #[test]
    fn bitwise_xor_immediate_sets_and_unsets_zero_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0xAF);
        emit_bitwise_xor_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x8F);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = ZERO_FLAG;
        cpu.registers.r2 = 0xCE;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.zero_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.zero_flag());
    }

    #[test]
    fn bitwise_xor_immediate_sets_and_unsets_negative_flag() {
        let mut cpu = create_test_cpu();
        let mut program = vec![];

        emit_bitwise_xor_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x0F);
        emit_bitwise_xor_reg_immediate(
            &mut program,
            ENCODING_R3,
            ENCODING_R2,
            0x70);

        update_program(&mut cpu, program, 0x2000);

        cpu.registers.flags = NEGATIVE_FLAG;
        cpu.registers.r2 = 0x0E;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(!cpu.registers.negative_flag());

        cpu.registers.flags = 0;
        cpu.registers.r2 = 0x8F;
        cpu.registers.r3 = 0;

        execute_instruction(&mut cpu);
        assert!(cpu.registers.negative_flag());
    }
}