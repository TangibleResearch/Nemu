#[derive(Debug, Clone, Copy)]
pub struct AluResult {
    pub value: u32,
    pub zero: bool,
    pub negative: bool,
    pub carry: bool,
    pub overflow: bool,
}

#[derive(Debug, Default)]
pub struct ScalarAlu;

impl ScalarAlu {
    pub const fn new() -> Self {
        Self
    }

    pub fn add(&self, a: u32, b: u32) -> AluResult {
        let (value, carry) = a.overflowing_add(b);

        let signed_a = a as i32;
        let signed_b = b as i32;
        let signed_value = value as i32;

        let overflow = (signed_a > 0 && signed_b > 0 && signed_value < 0)
            || (signed_a < 0 && signed_b < 0 && signed_value >= 0);

        AluResult {
            value,
            zero: value == 0,
            negative: (value & 0x8000_0000) != 0,
            carry,
            overflow,
        }
    }

    pub fn sub(&self, a: u32, b: u32) -> AluResult {
        let (value, borrow) = a.overflowing_sub(b);

        let signed_a = a as i32;
        let signed_b = b as i32;
        let signed_value = value as i32;

        let overflow = (signed_a >= 0 && signed_b < 0 && signed_value < 0)
            || (signed_a < 0 && signed_b >= 0 && signed_value >= 0);

        AluResult {
            value,
            zero: value == 0,
            negative: (value & 0x8000_0000) != 0,
            carry: !borrow,
            overflow,
        }
    }

    pub fn mul(&self, a: u32, b: u32) -> AluResult {
        let (value, overflow) = a.overflowing_mul(b);

        AluResult {
            value,
            zero: value == 0,
            negative: (value & 0x8000_0000) != 0,
            carry: overflow,
            overflow,
        }
    }

    pub fn div(&self, a: u32, b: u32) -> AluResult {
        if b == 0 {
            panic!("Neuron divide-by-zero exception");
        }

        let value = a / b;

        AluResult {
            value,
            zero: value == 0,
            negative: (value & 0x8000_0000) != 0,
            carry: false,
            overflow: false,
        }
    }

    pub fn modulo(&self, a: u32, b: u32) -> AluResult {
        if b == 0 {
            panic!("Neuron modulo-by-zero exception");
        }

        let value = a % b;

        AluResult {
            value,
            zero: value == 0,
            negative: (value & 0x8000_0000) != 0,
            carry: false,
            overflow: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ScalarAlu;

    #[test]
    fn reports_addition_flags() {
        let result = ScalarAlu.add(u32::MAX, 1);

        assert_eq!(result.value, 0);
        assert!(result.zero);
        assert!(result.carry);
        assert!(!result.negative);
    }

    #[test]
    fn reports_signed_overflow() {
        let result = ScalarAlu.add(i32::MAX as u32, 1);

        assert_eq!(result.value, i32::MIN as u32);
        assert!(result.negative);
        assert!(result.overflow);
    }
}
