#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Clock {
    tick: u64,
}

impl Clock {
    pub const fn new() -> Self {
        Self { tick: 0 }
    }

    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    pub const fn get_tick(&self) -> u64 {
        self.tick
    }
}

#[cfg(test)]
mod tests {
    use super::Clock;

    #[test]
    fn counts_ticks() {
        let mut clock = Clock::new();

        clock.tick();
        clock.tick();

        assert_eq!(clock.get_tick(), 2);
    }
}
