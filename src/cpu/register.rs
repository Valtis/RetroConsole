
/* CPU flags */
pub const ZERO_FLAG: u8 = 0x01;
pub const NEGATIVE_FLAG: u8 = 0x02;
pub const CARRY_FLAG: u8 = 0x04;
pub const OVERFLOW_FLAG: u8 = 0x08;
pub const INTERRUPT_ENABLE_FLAG: u8 = 0x10;
pub const BREAK_FLAG: u8  = 0x20;
pub const AUTO_INCREMENT_FLAG: u8 = 0x40;
pub const FAULT_FLAG: u8 = 0x80;


/* Initial value of the stack pointer */
const SP_INITIAL_VALUE: u16 = 0x200;

#[derive(Debug)]
pub struct Registers {
    pub r1: u8,
    pub r2: u8,
    pub r3: u8,
    pub r4: u8,
    pub pc: u16,
    pub sp: u16,
    pub flags: u8,
 }

impl Registers {
    pub fn new() -> Registers {
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

    pub fn zero_flag(&self) -> bool {
        self.flags & ZERO_FLAG != 0
    }

    pub fn negative_flag(&self) -> bool {
        self.flags & NEGATIVE_FLAG != 0
    }

    pub fn carry_flag(&self) -> bool {
        self.flags & CARRY_FLAG != 0
    }

    pub fn overflow_flag(&self) -> bool {
        self.flags & OVERFLOW_FLAG != 0
    }

    pub fn interrupt_flag(&self) -> bool {
        self.flags & INTERRUPT_ENABLE_FLAG != 0
    }

    pub fn auto_increment_flag(&self) -> bool {
        self.flags & AUTO_INCREMENT_FLAG != 0
    }

    pub fn fault_flag(&self) -> bool {
        self.flags & FAULT_FLAG != 0
    }

    pub fn set_zero_negative_flags(&mut self, value: u8) {
        self.set_zero_flag_on_value(value);
        self.set_negative_flag_on_value(value);
    }

    pub fn set_zero_flag_on_value(&mut self, value: u8) {
        if value == 0 {
            self.set_zero_flag();
        } else {
            self.clear_zero_flag();
        }
    }

    pub fn set_zero_flag(&mut self) {
        self.flags |= ZERO_FLAG;
    }

    pub fn clear_zero_flag(&mut self) {
        self.flags &= !ZERO_FLAG;
    }

    pub fn set_negative_flag_on_value(&mut self, value: u8) {
        if value > 127 {
            self.flags |= NEGATIVE_FLAG;
        } else {
            self.flags &= !NEGATIVE_FLAG;
        }
    }

    pub fn set_carry_flag_on_value(&mut self, result: u16) {
        if result > 255 {
            self.set_carry_flag();
        } else {
            self.clear_carry_flag();
        }
    }

    pub fn set_carry_flag(&mut self) {
        self.flags |= CARRY_FLAG;
    }

    pub fn clear_carry_flag(&mut self) {
        self.flags &= !CARRY_FLAG;
    }

    /*
        overflow happens if both src values have same sign (8th bit set/unset in
        both), and the sign bit in in the result is different --> operation
        on two positive values resulted in negative number, or other way around
    */
    pub fn set_overflow_flag_on_value(
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

    pub fn set_overflow_flag(&mut self) {
        self.flags |= OVERFLOW_FLAG;
    }

    pub fn clear_overflow_flag(&mut self) {
        self.flags &= !OVERFLOW_FLAG;
    }

    pub fn set_interrupt_flag(&mut self) {
        self.flags |= INTERRUPT_ENABLE_FLAG;
    }

    pub fn clear_interrupt_flag(&mut self) {
        self.flags &= !INTERRUPT_ENABLE_FLAG;
    }

    pub fn set_auto_increment_flag(&mut self) {
        self.flags |= AUTO_INCREMENT_FLAG;
    }

    pub fn clear_autoincrement_flag(&mut self) {
        self.flags &= !AUTO_INCREMENT_FLAG;
    }

    pub fn set_fault_flag(&mut self) {
        self.flags |= FAULT_FLAG;
    }

    pub fn clear_fault_flag(&mut self) {
        self.flags &= !FAULT_FLAG;
    }
}

