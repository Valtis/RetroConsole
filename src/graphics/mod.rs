

const PIXELS: usize = 320*240;

const PALETTE: [Pixel; 16] = [
    Pixel { red: 0, green: 0, blue: 0 },
    Pixel { red: 255, green: 255, blue: 255 },
    Pixel { red: 179, green: 0, blue: 0 },
    Pixel { red: 204, green: 41, blue: 0 },

    Pixel { red: 255, green: 102, blue: 102 },
    Pixel { red: 0, green: 153, blue: 51 },
    Pixel { red: 102, green: 255, blue: 102 },
    Pixel { red: 57, green: 230, blue: 0 },

    Pixel { red: 0, green: 0, blue: 230 },
    Pixel { red: 0, green: 153, blue: 255 },
    Pixel { red: 46, green: 92, blue: 184 },
    Pixel { red: 179, green: 179, blue: 0 },

    Pixel { red: 255, green: 255, blue: 102 },
    Pixel { red: 204, green: 204, blue: 0 },
    Pixel { red: 153, green: 153, blue: 153 },
    Pixel { red: 255, green: 26, blue: 255 },
];

/**/

#[derive(Clone)]
pub struct Pixel {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}


#[derive(Clone)]
pub struct ScreenBuffer {
    pub pixels: Vec<usize>
}

impl ScreenBuffer {
    pub fn color(&self, index: usize) -> Pixel {
        PALETTE[self.pixels[index]].clone()
    }
}



impl ScreenBuffer {
    pub fn new() -> ScreenBuffer {
        let mut buffer = ScreenBuffer {
            pixels: vec![]
        };

        buffer.pixels.resize(PIXELS, 0);

        buffer
    }
}

