use crate::clock::Clock;
use crate::cpu::FLAG_ZERO;
use crate::isa::*;
use crate::mac::Mac;
use crate::scalar_alu::{AluResult, ScalarAlu};

/// Dispatches decoded non-AI operations to Neuron execution units.
///
/// AI/DNN instructions such as MMUL and RELU are handled directly
/// by the CPU's AI accelerator path and must not execute here.
pub struct Issuer {
    scalar_alu: ScalarAlu,
    mac: Mac,
}

/// Result returned by the issuer after executing an instruction.
pub enum IssueResult {
    Scalar(AluResult),
    Value(u32),
    Branch(u32),
    Output(char),
    Halt,
    None,
}

impl IssueResult {
    pub(crate) fn scalar(self) -> AluResult {
        match self {
            Self::Scalar(result) => result,
            _ => panic!("issuer did not return a scalar result"),
        }
    }

    pub(crate) fn value(self) -> u32 {
        match self {
            Self::Value(value) => value,
            _ => panic!("issuer did not return a value"),
        }
    }

    pub(crate) fn branch(self) -> Option<u32> {
        match self {
            Self::Branch(address) => Some(address),
            Self::None => None,
            _ => panic!("issuer did not return a branch result"),
        }
    }

    pub(crate) fn output(self) -> char {
        match self {
            Self::Output(character) => character,
            _ => panic!("issuer did not return output"),
        }
    }

    pub(crate) fn expect_none(self) {
        if !matches!(self, Self::None) {
            panic!("issuer unexpectedly returned a result");
        }
    }

    pub(crate) fn expect_halt(self) {
        if !matches!(self, Self::Halt) {
            panic!("issuer did not return HALT");
        }
    }
}

impl Issuer {
    pub const fn new() -> Self {
        Self {
            scalar_alu: ScalarAlu::new(),
            mac: Mac::new(),
        }
    }

    /// Issues one instruction and advances the supplied architectural clock
    /// after successful non-output compute or state work.
    pub fn issue_clocked(
        &mut self,
        operation: u8,
        arguments: &[u32],
        flags: u32,
        clock: &mut Clock,
    ) -> IssueResult {
        let result = self.issue(operation, arguments, flags);

        if !matches!(operation, OP_OUT | OP_HALT) {
            clock.tick();
        }

        result
    }

    /// Issues one decoded non-AI instruction.
    ///
    /// `arguments` contains values already resolved by the CPU.
    ///
    /// MMUL and RELU deliberately panic here because they must be routed
    /// through the CPU's AI accelerator instead.
    pub fn issue(&mut self, operation: u8, arguments: &[u32], flags: u32) -> IssueResult {
        match operation {
            //
            // Data movement
            //
            OP_MOVI | OP_MOV | OP_LOAD | OP_POP => {
                IssueResult::Value(Self::arg(arguments, 0, operation))
            }

            OP_STORE | OP_PUSH => IssueResult::None,

            //
            // Output
            //
            OP_OUT => {
                let value = Self::arg(arguments, 0, operation);

                IssueResult::Output((value & 0xFF) as u8 as char)
            }

            //
            // Scalar ALU
            //
            OP_ADD => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Scalar(self.scalar_alu.add(a, b))
            }

            OP_SUB => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Scalar(self.scalar_alu.sub(a, b))
            }

            OP_MUL => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Scalar(self.scalar_alu.mul(a, b))
            }

            OP_DIV => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Scalar(self.scalar_alu.div(a, b))
            }

            OP_MOD => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Scalar(self.scalar_alu.modulo(a, b))
            }

            //
            // Bitwise operations
            //
            OP_AND => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Value(a & b)
            }

            OP_OR => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Value(a | b)
            }

            OP_XOR => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Value(a ^ b)
            }

            OP_NOT => {
                let value = Self::arg(arguments, 0, operation);

                IssueResult::Value(!value)
            }

            OP_SHL => {
                let value = Self::arg(arguments, 0, operation);
                let shift = Self::arg(arguments, 1, operation);

                IssueResult::Value(value.wrapping_shl(shift & 31))
            }

            OP_SHR => {
                let value = Self::arg(arguments, 0, operation);
                let shift = Self::arg(arguments, 1, operation);

                IssueResult::Value(value.wrapping_shr(shift & 31))
            }

            //
            // Comparison
            //
            OP_CMP => {
                let a = Self::arg(arguments, 0, operation);
                let b = Self::arg(arguments, 1, operation);

                IssueResult::Value(a.wrapping_sub(b))
            }

            //
            // Branching
            //
            OP_JMP | OP_CALL | OP_RET => {
                let address = Self::arg(arguments, 0, operation);

                IssueResult::Branch(address)
            }

            OP_JZ => {
                let address = Self::arg(arguments, 0, operation);

                if (flags & FLAG_ZERO) != 0 {
                    IssueResult::Branch(address)
                } else {
                    IssueResult::None
                }
            }

            OP_JNZ => {
                let address = Self::arg(arguments, 0, operation);

                if (flags & FLAG_ZERO) == 0 {
                    IssueResult::Branch(address)
                } else {
                    IssueResult::None
                }
            }

            //
            // Standalone MAC coprocessor
            //
            OP_MAC => {
                let a = Self::arg(arguments, 0, operation) as i8;
                let b = Self::arg(arguments, 1, operation) as i8;

                self.mac.step(a, b);

                IssueResult::None
            }

            OP_MACCLR => {
                self.mac.reset();

                IssueResult::None
            }

            OP_MACREAD => IssueResult::Value(self.mac.accumulator() as u32),

            //
            // AI / DNN accelerator operations
            //
            OP_MMUL | OP_RELU => {
                panic!("AI opcode {operation:#04X} must be dispatched to the CPU accelerator")
            }

            //
            // Processor control
            //
            OP_HALT => IssueResult::Halt,

            //
            // Unknown opcode
            //
            _ => {
                panic!("Issuer received unsupported opcode: {operation:#04X}")
            }
        }
    }

    /// Returns one argument or panics with a useful message.
    ///
    /// This is better than scattering arguments[0], arguments[1], etc.
    /// everywhere because malformed decoded instructions now produce
    /// an understandable error instead of a generic slice panic.
    fn arg(arguments: &[u32], index: usize, operation: u8) -> u32 {
        arguments.get(index).copied().unwrap_or_else(|| {
            panic!(
                "Opcode {operation:#04X} expected argument {index}, \
                 but only {} argument(s) were supplied",
                arguments.len()
            )
        })
    }

    /// Current value of the standalone MAC accumulator.
    pub const fn mac_accumulator(&self) -> i32 {
        self.mac.accumulator()
    }
}

impl Default for Issuer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::FLAG_ZERO;

    #[test]
    fn add_is_dispatched_to_scalar_alu() {
        let mut issuer = Issuer::default();

        let result = issuer.issue(OP_ADD, &[5, 7], 0).scalar();

        assert_eq!(result.value, 12);
        assert!(!result.zero);
    }

    #[test]
    fn movi_returns_value() {
        let mut issuer = Issuer::default();

        let result = issuer.issue(OP_MOVI, &[123], 0).value();

        assert_eq!(result, 123);
    }

    #[test]
    fn jz_branches_when_zero_flag_is_set() {
        let mut issuer = Issuer::default();

        let result = issuer.issue(OP_JZ, &[0x1234], FLAG_ZERO);

        assert_eq!(result.branch(), Some(0x1234));
    }

    #[test]
    fn jz_does_not_branch_without_zero_flag() {
        let mut issuer = Issuer::default();

        let result = issuer.issue(OP_JZ, &[0x1234], 0);

        assert_eq!(result.branch(), None);
    }

    #[test]
    fn jnz_branches_when_zero_flag_is_clear() {
        let mut issuer = Issuer::default();

        let result = issuer.issue(OP_JNZ, &[0x5678], 0);

        assert_eq!(result.branch(), Some(0x5678));
    }

    #[test]
    fn mac_accumulates() {
        let mut issuer = Issuer::default();

        issuer.issue(OP_MACCLR, &[], 0).expect_none();

        issuer.issue(OP_MAC, &[2, 5], 0).expect_none();

        issuer.issue(OP_MAC, &[3, 4], 0).expect_none();

        assert_eq!(issuer.mac_accumulator(), 22);

        assert_eq!(issuer.issue(OP_MACREAD, &[], 0).value(), 22);
    }

    #[test]
    fn output_returns_low_byte_as_character() {
        let mut issuer = Issuer::default();

        let output = issuer.issue(OP_OUT, &['A' as u32], 0).output();

        assert_eq!(output, 'A');
    }

    #[test]
    #[should_panic(expected = "must be dispatched to the CPU accelerator")]
    fn mmul_cannot_be_sent_to_normal_issuer() {
        let mut issuer = Issuer::default();

        issuer.issue(OP_MMUL, &[], 0);
    }

    #[test]
    #[should_panic(expected = "must be dispatched to the CPU accelerator")]
    fn relu_cannot_be_sent_to_normal_issuer() {
        let mut issuer = Issuer::default();

        issuer.issue(OP_RELU, &[5], 0);
    }

    #[test]
    #[should_panic(expected = "expected argument 1")]
    fn malformed_instruction_reports_missing_argument() {
        let mut issuer = Issuer::default();

        issuer.issue(OP_ADD, &[5], 0);
    }

    #[test]
    fn clocked_issue_counts_compute_but_not_output_or_halt() {
        let mut issuer = Issuer::default();
        let mut clock = Clock::new();

        issuer.issue_clocked(OP_ADD, &[2, 3], 0, &mut clock);
        issuer.issue_clocked(OP_OUT, &['A' as u32], 0, &mut clock);
        issuer.issue_clocked(OP_HALT, &[], 0, &mut clock);

        assert_eq!(clock.get_tick(), 1);
    }
}
