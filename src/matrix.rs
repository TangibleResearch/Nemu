use crate::clock::Clock;
use crate::mac::Mac;

pub const MATRIX_SIZE: usize = 4;
pub type Matrix = [[i32; MATRIX_SIZE]; MATRIX_SIZE];

/// A 4x4 matrix engine backed by 16 parallel MAC units.
#[derive(Debug, Clone, Copy)]
pub struct MatrixEngine {
    macs: [[Mac; MATRIX_SIZE]; MATRIX_SIZE],
    a_tile: Matrix,
    b_tile: Matrix,
    output: Matrix,
    k: usize,
    busy: bool,
    done: bool,
    clock: Clock,
}

impl MatrixEngine {
    pub fn new() -> Self {
        Self {
            macs: [[Mac::new(); MATRIX_SIZE]; MATRIX_SIZE],
            a_tile: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            b_tile: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            output: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            k: 0,
            busy: false,
            done: false,
            clock: Clock::new(),
        }
    }

    pub fn load_tiles(&mut self, a: Matrix, b: Matrix) {
        assert!(!self.busy, "Cannot load Matrix Engine while it is busy");
        self.a_tile = a;
        self.b_tile = b;
        self.done = false;
    }

    pub fn start(&mut self) {
        assert!(!self.busy, "Matrix Engine is already busy");

        for row in &mut self.macs {
            for mac in row {
                mac.reset();
            }
        }

        self.output = [[0; MATRIX_SIZE]; MATRIX_SIZE];
        self.k = 0;
        self.busy = true;
        self.done = false;
    }

    /// Simulates one matrix-engine cycle.
    ///
    /// All 16 MACs conceptually execute in parallel. The emulator loops over
    /// them sequentially, but one call represents one hardware cycle.
    pub fn step_cycle(&mut self) {
        if !self.busy || self.done {
            return;
        }

        self.clock.tick();

        for row in 0..MATRIX_SIZE {
            for column in 0..MATRIX_SIZE {
                let a = i8::try_from(self.a_tile[row][self.k])
                    .expect("Matrix Engine input A must fit in INT8");
                let b = i8::try_from(self.b_tile[self.k][column])
                    .expect("Matrix Engine input B must fit in INT8");
                self.macs[row][column].step(a, b);
            }
        }

        self.k += 1;

        if self.k == MATRIX_SIZE {
            for row in 0..MATRIX_SIZE {
                for column in 0..MATRIX_SIZE {
                    self.output[row][column] = self.macs[row][column].accumulator();
                }
            }
            self.busy = false;
            self.done = true;
        }
    }

    pub const fn is_busy(&self) -> bool {
        self.busy
    }

    pub const fn is_done(&self) -> bool {
        self.done
    }

    pub const fn read_output(&self) -> Option<Matrix> {
        if self.done { Some(self.output) } else { None }
    }

    pub const fn clock_ticks(&self) -> u64 {
        self.clock.get_tick()
    }
}

impl Default for MatrixEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{MATRIX_SIZE, MatrixEngine};

    #[test]
    fn multiplies_a_tile_by_an_identity_tile() {
        let input = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ];
        let identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        let mut engine = MatrixEngine::new();

        engine.load_tiles(input, identity);
        engine.start();
        for _ in 0..MATRIX_SIZE {
            engine.step_cycle();
        }

        assert!(engine.is_done());
        assert_eq!(engine.read_output(), Some(input));
        assert_eq!(engine.clock_ticks(), MATRIX_SIZE as u64);
    }
}
