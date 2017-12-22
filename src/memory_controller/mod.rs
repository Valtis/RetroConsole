

pub struct MemoryController {
    memory: Vec<u8>
}




impl MemoryController {
    pub fn new(size: usize) -> MemoryController {
        let mut controller = MemoryController {
            memory: vec![],
        };

        controller.memory.resize(size, 0);

        controller
    }


    pub fn read(&self, address: u16) -> u8 {
        self.memory[address as usize]
    }

    pub fn write(&mut self, address: u16, value: u8) {
        self.memory[address as usize] = value;
    }
}