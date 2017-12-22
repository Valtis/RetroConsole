extern crate sdl2;
extern crate time;

use self::sdl2::keyboard::Keycode;
use self::sdl2::event::Event;

use std::rc::Rc;
use std::cell::RefCell;

use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use graphics::{ScreenBuffer};

use self::sdl2::pixels::PixelFormatEnum;
use self::sdl2::rect::Rect;

use cpu::Cpu;
use memory_controller::MemoryController;

// hz
const CPU_FREQUENCY: u32 = 2764800;
const CPU_CYCLES_PER_EMULATION_TICK: u32 = 10;


pub struct AsyncConsole {
    cpu: Cpu,
    cpu_cycle_time: u64,
    time: u64,
    buffer_ready: Arc<AtomicBool>,
    debug_ctr: u32,
    debug_index: usize,
    debug_scanline: u32,
    debug_column: u32,
    debug_pixel_index: usize,
    debug_buffer: ScreenBuffer,
    debug_external_buffer: Arc<Mutex<ScreenBuffer>>,
}

impl AsyncConsole {

    pub fn new(
        buffer_ready: Arc<AtomicBool>,
        external_buffer: Arc<Mutex<ScreenBuffer>>) -> AsyncConsole {
        let cycle_time = (1_000_000_000.0/(CPU_FREQUENCY as f64)) as u64;
        println!("Cycle time: {}", cycle_time);

        let controller =
            Rc::new(
                RefCell::new(
                    MemoryController::new(128*1024)));

        AsyncConsole {
            cpu: Cpu::new(controller.clone()),
            cpu_cycle_time: cycle_time,
            time: time::precise_time_ns(),
            buffer_ready: buffer_ready,
            debug_ctr: 0,
            debug_index: 0,
            debug_scanline: 0,
            debug_column: 0,
            debug_pixel_index: 0,
            debug_buffer: ScreenBuffer::new(),
            debug_external_buffer: external_buffer,
        }
    }

    fn execute(&mut self) {

        let current_time = time::precise_time_ns();
        let time_taken = current_time - self.time;
        let cycle_time =
            self.cpu_cycle_time*CPU_CYCLES_PER_EMULATION_TICK as u64;

        if time_taken > cycle_time {
            for _ in 0..CPU_CYCLES_PER_EMULATION_TICK {
                self.run_emulation_tick();
            }

            self.time = current_time - (time_taken - cycle_time);
        }

    }

    fn run_emulation_tick(&mut self) {
        self.debug_ctr += 1;
        if self.debug_ctr > CPU_FREQUENCY {
            self.debug_ctr = 0;
            self.debug_pixel_index = (self.debug_pixel_index + 1) & 0x0F;
        }

        for _ in 0..2 {
            if self.debug_scanline < 240 {
                if self.debug_column < 320 {
                    self
                        .debug_buffer
                        .pixels[self.debug_index] = self.debug_pixel_index;
                    self.debug_index += 1;
                }
            } else if self.debug_scanline == 240 && self.debug_column == 0 {
                self.debug_index = 0;
                let mut ptr = self.debug_external_buffer.lock().unwrap();
                *ptr = self.debug_buffer.clone();
                self.buffer_ready.store(true, Ordering::SeqCst);
            }

            self.debug_column += 1;
            if self.debug_column == 360 {
                self.debug_column = 0;
                self.debug_scanline += 1;
                if self.debug_scanline == 256 {
                    self.debug_scanline = 0;
                }
            }
        }

        self.cpu.execute();
/*        self.gpu.execute();
        self.apu.execute();*/
    }

}


pub struct RetroConsole {
    run: Arc<AtomicBool>,
    buffer_ready: Arc<AtomicBool>,
    buffer: Arc<Mutex<ScreenBuffer>>,
}

impl RetroConsole {
    fn new() -> RetroConsole {
        RetroConsole {
            run: Arc::new(AtomicBool::new(false)),
            buffer_ready: Arc::new(AtomicBool::new(false)),
            buffer: Arc::new(Mutex::new(ScreenBuffer::new()))
        }
    }

    pub fn stop(&mut self) {
        self.run.store(false, Ordering::SeqCst);
    }

    pub fn run_async(&mut self) {
        let run = self.run.clone();
        run.store(true, Ordering::SeqCst);
        let buffer_ready = self.buffer_ready.clone();
        let buffer = self.buffer.clone();

        thread::spawn(move || {
            let mut console = AsyncConsole::new(
                buffer_ready,
                buffer);

            while run.load(Ordering::SeqCst) {
                console.execute();
            }

        });

    }

    pub fn screen_buffer(&self) -> Option<ScreenBuffer> {
        if self.buffer_ready.load(Ordering::SeqCst) {
            self.buffer_ready.store(false, Ordering::SeqCst);
            Some(self.buffer.lock().unwrap().clone())
        } else {
            None
        }
    }

}


pub fn run() {

    let mut console = RetroConsole::new();

    let sdl_context = sdl2::init()
        .unwrap_or_else(|e| panic!("Failed to initialize SDL context: {}", e));

    let video_subsystem = sdl_context.video().unwrap_or_else(
        |e| panic!("Failed to initialize SDL video subsystem: {}", e));


    // hardcoded resolution for now.
    // TODO: Implement arbitrary resolution & scaling
    let window = video_subsystem.window("RustNes", 320*2, 240*2)
        .position_centered()
        .opengl()
        .build()
        .unwrap_or_else(|e| panic!("Failed to create window: {}", e));


    let mut canvas = window
        .into_canvas()
        .build()
        .unwrap_or_else(|e| panic!("Failed to create canvas: {}", e));
    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
            .create_texture_streaming(
                PixelFormatEnum::RGB888, 320, 240)
            .unwrap_or_else(|e| panic!("Failed to create texture: {}", e));


    console.run_async();

    'main_loop: loop {
        let mut event_pump = sdl_context.event_pump().unwrap();
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    console.stop();
                    break 'main_loop;
                },
                _ => {}
            }
        }

        /* move into separate function/module*/
        if let Some(ext_buffer) = console.screen_buffer() {
            texture
                .with_lock(None, |buffer: &mut [u8], pitch: usize| {
                    for y in 0..240 {
                         for x in 0..320 {
                             let index = ext_buffer.pixels[y * 320 + x];
                             let pixel = ext_buffer.color(index);
                             let offset = y*pitch + 4*x;
                             buffer[offset + 0] = pixel.blue;
                             buffer[offset + 1] = pixel.green;
                             buffer[offset + 2] = pixel.red;
                             buffer[offset + 3] = 255 as u8;
                         }
                     }
                 })
                .unwrap();

            canvas.clear();
            canvas.copy(&texture, None, Rect::new(0, 0, 320*2, 240*2));
            canvas.present();
        }
    }
}



