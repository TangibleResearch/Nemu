use crate::accelerator::{AcceleratorInstruction, AiAccelerator};
use crate::clock::Clock;
use crate::debug::trace::TraceEvent;
use crate::isa::*;
use crate::issuer::{IssueResult, Issuer};
use crate::matrix::{MATRIX_SIZE, Matrix, MatrixEngine};

pub const FLAG_ZERO: u32 = 1 << 0;
pub const FLAG_NEGATIVE: u32 = 1 << 1;
pub const FLAG_CARRY: u32 = 1 << 2;
pub const FLAG_OVERFLOW: u32 = 1 << 3;

pub struct NeuronCpu {
    issuer: Issuer,
    clock: Clock,
    pub accelerator: AiAccelerator,
    pub matrix_engine: MatrixEngine,
    memory_size: u32,

    // =========================
    // Scalar General Registers
    // =========================
    r0: u32,
    r1: u32,
    r2: u32,
    r3: u32,
    r4: u32,
    r5: u32,
    r6: u32,
    r7: u32,
    r8: u32,
    r9: u32,
    r10: u32,
    r11: u32,
    r12: u32,
    r13: u32,
    r14: u32,
    r15: u32,

    // =========================
    // Vector Registers
    // =========================
    v0: [u32; 8],
    v1: [u32; 8],
    v2: [u32; 8],
    v3: [u32; 8],
    v4: [u32; 8],
    v5: [u32; 8],
    v6: [u32; 8],
    v7: [u32; 8],

    // =========================
    // Matrix Registers
    // =========================
    m0: Matrix,
    m1: Matrix,
    m2: Matrix,
    m3: Matrix,

    // =========================
    // Predicate Registers
    // =========================
    p0: u8,
    p1: u8,
    p2: u8,
    p3: u8,

    // =========================
    // Special Registers
    // =========================
    pc: u32,
    sp: u32,
    fp: u32,
    status: u32,

    // =========================
    // AI Control Registers
    // =========================
    ai_mode: u32,
    quant_ctrl: u32,
    sparse_ctrl: u32,
    tensor_ctrl: u32,
    tensor_status: u32,

    halted: bool,
}
impl NeuronCpu {
    pub fn new(memory_size: u32) -> Self {
        Self::with_accelerator(memory_size, AiAccelerator::default())
    }

    /// Creates a CPU with a caller-provided accelerator configuration.
    pub fn with_accelerator(memory_size: u32, accelerator: AiAccelerator) -> Self {
        Self {
            issuer: Issuer::new(),
            clock: Clock::new(),
            accelerator,
            matrix_engine: MatrixEngine::new(),
            memory_size,
            r0: 0,
            r1: 0,
            r2: 0,
            r3: 0,
            r4: 0,
            r5: 0,
            r6: 0,
            r7: 0,
            r8: 0,
            r9: 0,
            r10: 0,
            r11: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            v0: [0; 8],
            v1: [0; 8],
            v2: [0; 8],
            v3: [0; 8],
            v4: [0; 8],
            v5: [0; 8],
            v6: [0; 8],
            v7: [0; 8],
            m0: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            m1: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            m2: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            m3: [[0; MATRIX_SIZE]; MATRIX_SIZE],
            p0: 0,
            p1: 0,
            p2: 0,
            p3: 0,
            pc: 0,
            sp: memory_size,
            fp: 0,
            status: 0,
            ai_mode: 0,
            quant_ctrl: 0,
            sparse_ctrl: 0,
            tensor_ctrl: 0,
            tensor_status: 0,
            halted: false,
        }
    }

    /// Restores the processor to the same fresh state as [`Self::new`].
    ///
    /// Program and system RAM are owned by the caller and are intentionally
    /// unaffected by a CPU reset.
    pub fn reset(&mut self) {
        println!("NC Alert: Neuron System Reset");
        *self = Self::new(self.memory_size);
    }

    /// Returns the AI accelerator integrated into this CPU.
    pub const fn accelerator(&self) -> &AiAccelerator {
        &self.accelerator
    }

    /// Returns mutable access to the integrated AI accelerator.
    pub const fn accelerator_mut(&mut self) -> &mut AiAccelerator {
        &mut self.accelerator
    }

    /// Queues work on the CPU's AI accelerator.
    pub fn queue_accelerator_instruction(&mut self, instruction: AcceleratorInstruction) {
        self.accelerator.add_instruction(instruction);
    }

    /// Executes the next queued instruction on the CPU's AI accelerator.
    pub fn process_accelerator_instruction(&mut self) -> Result<(), String> {
        self.accelerator
            .process_instruction_clocked(&mut self.clock)
            .map(|_| ())
    }

    fn execute_accelerator_instruction(&mut self, instruction: AcceleratorInstruction) -> f32 {
        self.accelerator
            .execute_instruction_clocked(instruction, &mut self.clock)
            .unwrap_or_else(|error| panic!("Neuron AI accelerator execution failed: {error}"))
    }

    // ============================================================
    // SCALAR REGISTER ACCESS
    // ============================================================

    pub fn read_scalar(&self, register: u8) -> u32 {
        match register {
            0 => self.r0,
            1 => self.r1,
            2 => self.r2,
            3 => self.r3,
            4 => self.r4,
            5 => self.r5,
            6 => self.r6,
            7 => self.r7,
            8 => self.r8,
            9 => self.r9,
            10 => self.r10,
            11 => self.r11,
            12 => self.r12,
            13 => self.r13,
            14 => self.r14,
            15 => self.r15,

            _ => {
                panic!("Invalid Neuron scalar register: R{}", register);
            }
        }
    }

    pub fn write_scalar(&mut self, register: u8, value: u32) {
        match register {
            0 => self.r0 = value,
            1 => self.r1 = value,
            2 => self.r2 = value,
            3 => self.r3 = value,
            4 => self.r4 = value,
            5 => self.r5 = value,
            6 => self.r6 = value,
            7 => self.r7 = value,
            8 => self.r8 = value,
            9 => self.r9 = value,
            10 => self.r10 = value,
            11 => self.r11 = value,
            12 => self.r12 = value,
            13 => self.r13 = value,
            14 => self.r14 = value,
            15 => self.r15 = value,

            _ => {
                panic!("Invalid Neuron scalar register: R{}", register);
            }
        }
    }

    pub fn read_vector(&self, register: u8) -> [u32; 8] {
        match register {
            0 => self.v0,
            1 => self.v1,
            2 => self.v2,
            3 => self.v3,
            4 => self.v4,
            5 => self.v5,
            6 => self.v6,
            7 => self.v7,
            _ => panic!("Invalid Neuron vector register: V{}", register),
        }
    }

    pub fn write_vector(&mut self, register: u8, value: [u32; 8]) {
        match register {
            0 => self.v0 = value,
            1 => self.v1 = value,
            2 => self.v2 = value,
            3 => self.v3 = value,
            4 => self.v4 = value,
            5 => self.v5 = value,
            6 => self.v6 = value,
            7 => self.v7 = value,
            _ => panic!("Invalid Neuron vector register: V{}", register),
        }
    }

    // ============================================================
    // MATRIX REGISTER ACCESS
    // ============================================================

    pub fn read_matrix(&self, register: u8) -> Matrix {
        match register {
            0 => self.m0,
            1 => self.m1,
            2 => self.m2,
            3 => self.m3,
            _ => panic!("Invalid Neuron matrix register: M{}", register),
        }
    }

    pub fn write_matrix(&mut self, register: u8, value: Matrix) {
        match register {
            0 => self.m0 = value,
            1 => self.m1 = value,
            2 => self.m2 = value,
            3 => self.m3 = value,
            _ => panic!("Invalid Neuron matrix register: M{}", register),
        }
    }

    pub fn read_predicate(&self, register: u8) -> u8 {
        match register {
            0 => self.p0,
            1 => self.p1,
            2 => self.p2,
            3 => self.p3,
            _ => panic!("Invalid Neuron predicate register: P{}", register),
        }
    }

    pub fn write_predicate(&mut self, register: u8, value: u8) {
        match register {
            0 => self.p0 = value,
            1 => self.p1 = value,
            2 => self.p2 = value,
            3 => self.p3 = value,
            _ => panic!("Invalid Neuron predicate register: P{}", register),
        }
    }

    // ============================================================
    // MEMORY HELPERS
    // ============================================================

    fn read_u32(memory: &[u8], address: u32) -> u32 {
        let start = address as usize;
        let end = start + 4;

        let bytes: [u8; 4] = memory[start..end]
            .try_into()
            .expect("Unable to read 32-bit value from Neuron memory");

        u32::from_le_bytes(bytes)
    }

    fn write_u32(memory: &mut [u8], address: u32, value: u32) {
        let start = address as usize;
        let end = start + 4;

        memory[start..end].copy_from_slice(&value.to_le_bytes());
    }

    fn fetch_u8(&mut self, memory: &[u8]) -> u8 {
        let value = memory[self.pc as usize];
        self.pc += 1;
        value
    }

    fn fetch_u32(&mut self, memory: &[u8]) -> u32 {
        let value = Self::read_u32(memory, self.pc);
        self.pc += 4;
        value
    }

    // ============================================================
    // STATUS FLAGS
    // ============================================================

    fn set_flag(&mut self, flag: u32, enabled: bool) {
        if enabled {
            self.status |= flag;
        } else {
            self.status &= !flag;
        }
    }

    fn update_zero_negative(&mut self, value: u32) {
        self.set_flag(FLAG_ZERO, value == 0);

        self.set_flag(FLAG_NEGATIVE, (value & 0x8000_0000) != 0);
    }

    // ============================================================
    // STACK
    // ============================================================

    fn push_u32(&mut self, memory: &mut [u8], value: u32) {
        self.sp = self.sp.checked_sub(4).expect("Neuron stack underflow");

        Self::write_u32(memory, self.sp, value);
    }

    fn pop_u32(&mut self, memory: &[u8]) -> u32 {
        let value = Self::read_u32(memory, self.sp);

        self.sp = self
            .sp
            .checked_add(4)
            .expect("Neuron stack pointer overflow");

        value
    }

    // ============================================================
    // CPU EXECUTION
    // ============================================================

    fn issue(&mut self, operation: u8, arguments: &[u32]) -> IssueResult {
        self.issuer
            .issue_clocked(operation, arguments, self.status, &mut self.clock)
    }

    pub fn step(&mut self, memory: &mut [u8]) -> Option<TraceEvent> {
        if self.halted {
            return None;
        }

        let instruction_pc = self.pc;

        let opcode = self.fetch_u8(memory);

        match opcode {
            // ====================================================
            // 0x10 - MOVI
            //
            // MOVI destination, immediate32
            //
            // [10][dst][imm32]
            // ====================================================
            OP_MOVI => {
                let destination = self.fetch_u8(memory);

                let immediate = self.fetch_u32(memory);

                let value = self.issue(OP_MOVI, &[immediate]).value();

                self.write_scalar(destination, value);

                self.update_zero_negative(value);
            }

            // ====================================================
            // 0x11 - MOV
            //
            // MOV destination, source
            //
            // [11][dst][src]
            // ====================================================
            OP_MOV => {
                let destination = self.fetch_u8(memory);
                let source = self.fetch_u8(memory);

                let source_value = self.read_scalar(source);
                let value = self.issue(OP_MOV, &[source_value]).value();

                self.write_scalar(destination, value);
            }
            // 0xA0
            OP_OUT => {
                let source = self.fetch_u8(memory);
                let value = self.read_scalar(source);

                let character = self.issue(OP_OUT, &[value]).output();

                print!("{}", character);
            }
            // ====================================================
            // 0x20 - ADD
            //
            // ADD destination, source_a, source_b
            // ====================================================
            OP_ADD => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);

                // Send the operands to the physical Scalar ALU.
                let alu_result = self.issue(OP_ADD, &[a, b]).scalar();

                // Write the ALU result back to the register file.
                self.write_scalar(destination, alu_result.value);

                // Copy the ALU's flags into STATUS.
                self.set_flag(FLAG_ZERO, alu_result.zero);
                self.set_flag(FLAG_NEGATIVE, alu_result.negative);
                self.set_flag(FLAG_CARRY, alu_result.carry);
                self.set_flag(FLAG_OVERFLOW, alu_result.overflow);
            }

            // ====================================================
            // 0x21 - SUB
            // ====================================================
            OP_SUB => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);

                let alu_result = self.issue(OP_SUB, &[a, b]).scalar();
                self.write_scalar(destination, alu_result.value);

                self.update_zero_negative(alu_result.value);

                self.set_flag(FLAG_CARRY, alu_result.carry);

                self.set_flag(FLAG_OVERFLOW, alu_result.overflow);
            }

            // ====================================================
            // 0x22 - MUL
            // ====================================================
            OP_MUL => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);

                let alu_result = self.issue(OP_MUL, &[a, b]).scalar();

                self.write_scalar(destination, alu_result.value);

                self.set_flag(FLAG_ZERO, alu_result.zero);
                self.set_flag(FLAG_NEGATIVE, alu_result.negative);
                self.set_flag(FLAG_CARRY, alu_result.carry);
                self.set_flag(FLAG_OVERFLOW, alu_result.overflow);
            }

            // ====================================================
            // 0x23 - DIV
            // ====================================================
            OP_DIV => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);

                let alu_result = self.issue(OP_DIV, &[a, b]).scalar();

                self.write_scalar(destination, alu_result.value);

                self.set_flag(FLAG_ZERO, alu_result.zero);
                self.set_flag(FLAG_NEGATIVE, alu_result.negative);
                self.set_flag(FLAG_CARRY, alu_result.carry);
                self.set_flag(FLAG_OVERFLOW, alu_result.overflow);
            }

            // ====================================================
            // 0x24 - MOD
            // ====================================================
            OP_MOD => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);

                let alu_result = self.issue(OP_MOD, &[a, b]).scalar();

                self.write_scalar(destination, alu_result.value);

                self.set_flag(FLAG_ZERO, alu_result.zero);
                self.set_flag(FLAG_NEGATIVE, alu_result.negative);
                self.set_flag(FLAG_CARRY, alu_result.carry);
                self.set_flag(FLAG_OVERFLOW, alu_result.overflow);
            }
            // ====================================================
            // 0x30 - AND
            // ====================================================
            OP_AND => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);
                let result = self.issue(OP_AND, &[a, b]).value();

                self.write_scalar(destination, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0x31 - OR
            // ====================================================
            OP_OR => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);
                let result = self.issue(OP_OR, &[a, b]).value();

                self.write_scalar(destination, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0x32 - XOR
            // ====================================================
            OP_XOR => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);
                let b = self.read_scalar(source_b);
                let result = self.issue(OP_XOR, &[a, b]).value();

                self.write_scalar(destination, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0x33 - NOT
            //
            // NOT destination, source
            // ====================================================
            OP_NOT => {
                let destination = self.fetch_u8(memory);
                let source = self.fetch_u8(memory);

                let value = self.read_scalar(source);
                let result = self.issue(OP_NOT, &[value]).value();

                self.write_scalar(destination, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0x34 - SHL
            // ====================================================
            OP_SHL => {
                let destination = self.fetch_u8(memory);
                let source = self.fetch_u8(memory);
                let amount_register = self.fetch_u8(memory);

                let value = self.read_scalar(source);

                let amount = self.read_scalar(amount_register);
                let result = self.issue(OP_SHL, &[value, amount]).value();

                self.write_scalar(destination, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0x35 - SHR
            // ====================================================
            OP_SHR => {
                let destination = self.fetch_u8(memory);
                let source = self.fetch_u8(memory);
                let amount_register = self.fetch_u8(memory);

                let value = self.read_scalar(source);

                let amount = self.read_scalar(amount_register);
                let result = self.issue(OP_SHR, &[value, amount]).value();

                self.write_scalar(destination, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0x40 - LOAD
            //
            // LOAD destination, address_register
            // ====================================================
            OP_LOAD => {
                let destination = self.fetch_u8(memory);
                let address_register = self.fetch_u8(memory);

                let address = self.read_scalar(address_register);

                let loaded = Self::read_u32(memory, address);
                let value = self.issue(OP_LOAD, &[loaded]).value();

                self.write_scalar(destination, value);
            }

            // ====================================================
            // 0x41 - STORE
            //
            // STORE address_register, source
            // ====================================================
            OP_STORE => {
                let address_register = self.fetch_u8(memory);

                let source = self.fetch_u8(memory);

                let address = self.read_scalar(address_register);

                let value = self.read_scalar(source);

                self.issue(OP_STORE, &[address, value]).expect_none();

                Self::write_u32(memory, address, value);
            }

            // ====================================================
            // 0x50 - PUSH
            // ====================================================
            OP_PUSH => {
                let source = self.fetch_u8(memory);

                let value = self.read_scalar(source);

                self.issue(OP_PUSH, &[value]).expect_none();

                self.push_u32(memory, value);
            }

            // ====================================================
            // 0x51 - POP
            // ====================================================
            OP_POP => {
                let destination = self.fetch_u8(memory);

                let popped = self.pop_u32(memory);
                let value = self.issue(OP_POP, &[popped]).value();

                self.write_scalar(destination, value);
            }

            // ====================================================
            // 0x60 - CMP
            //
            // CMP source_a, source_b
            //
            // Updates STATUS.
            // ====================================================
            OP_CMP => {
                let source_a = self.fetch_u8(memory);

                let source_b = self.fetch_u8(memory);

                let a = self.read_scalar(source_a);

                let b = self.read_scalar(source_b);

                let result = self.issue(OP_CMP, &[a, b]).value();

                self.update_zero_negative(result);

                self.set_flag(FLAG_CARRY, a >= b);
            }

            // ====================================================
            // 0x70 - JMP
            //
            // JMP absolute_address
            // ====================================================
            OP_JMP => {
                let address = self.fetch_u32(memory);

                self.pc = self
                    .issue(OP_JMP, &[address])
                    .branch()
                    .expect("JMP must produce a branch target");
            }

            // ====================================================
            // 0x71 - JZ
            // ====================================================
            OP_JZ => {
                let address = self.fetch_u32(memory);

                if let Some(target) = self.issue(OP_JZ, &[address]).branch() {
                    self.pc = target;
                }
            }

            // ====================================================
            // 0x72 - JNZ
            // ====================================================
            OP_JNZ => {
                let address = self.fetch_u32(memory);

                if let Some(target) = self.issue(OP_JNZ, &[address]).branch() {
                    self.pc = target;
                }
            }

            // ====================================================
            // 0x80 - CALL
            //
            // CALL absolute_address
            // ====================================================
            OP_CALL => {
                let address = self.fetch_u32(memory);

                let return_address = self.pc;

                self.push_u32(memory, return_address);

                self.pc = self
                    .issue(OP_CALL, &[address])
                    .branch()
                    .expect("CALL must produce a branch target");
            }

            // ====================================================
            // 0x81 - RET
            // ====================================================
            OP_RET => {
                let return_address = self.pop_u32(memory);

                self.pc = self
                    .issue(OP_RET, &[return_address])
                    .branch()
                    .expect("RET must produce a branch target");
            }
            // ====================================================
            // 0x82 - MAC (Multiply-Accumulate)
            //
            // Encoding:
            // 82 <register A> <register B>
            //
            // Operation:
            // ACC = ACC + (A * B)
            // ====================================================
            OP_MAC => {
                let address_a = self.fetch_u8(memory);
                let address_b = self.fetch_u8(memory);

                let value_a = self.read_scalar(address_a) as i8;
                let value_b = self.read_scalar(address_b) as i8;

                self.issue(OP_MAC, &[value_a as u32, value_b as u32])
                    .expect_none();
            }

            // ====================================================
            // 0x83 - MACCLR
            //
            // Operation:
            // ACC = 0
            // ====================================================
            OP_MACCLR => {
                self.issue(OP_MACCLR, &[]).expect_none();
            }

            // ====================================================
            // 0x84 - MACREAD
            //
            // Encoding:
            // 84 <destination register>
            //
            // Operation:
            // destination = ACC
            // ====================================================
            OP_MACREAD => {
                let destination = self.fetch_u8(memory);

                let accumulator = self.issue(OP_MACREAD, &[]).value();

                self.write_scalar(destination, accumulator);

                self.update_zero_negative(accumulator);
            }

            // ====================================================
            // 0x92 - MSET
            //
            // Encoding:
            // 92 <matrix> <row> <column> <signed INT8 immediate>
            // ====================================================
            OP_MSET => {
                let register = self.fetch_u8(memory);
                let row = self.fetch_u8(memory) as usize;
                let column = self.fetch_u8(memory) as usize;
                let value = self.fetch_u8(memory) as i8;

                if register > 3 {
                    panic!("MSET: invalid matrix register M{register}");
                }
                if row >= MATRIX_SIZE {
                    panic!("MSET: matrix row {row} is outside 0..={}", MATRIX_SIZE - 1);
                }
                if column >= MATRIX_SIZE {
                    panic!(
                        "MSET: matrix column {column} is outside 0..={}",
                        MATRIX_SIZE - 1
                    );
                }

                let mut matrix = self.read_matrix(register);
                matrix[row][column] = i32::from(value);
                self.write_matrix(register, matrix);
                self.clock.tick();
            }

            // ====================================================
            // 0x90 - MMUL
            //
            // Encoding:
            // 90 <destination matrix> <source A matrix> <source B matrix>
            //
            // Example:
            // MMUL M2, M0, M1
            //
            // Operation:
            // M2 = M0 × M1
            //
            // 4x4 INT8 inputs
            // INT32 accumulated output
            // ====================================================
            OP_MMUL => {
                let destination = self.fetch_u8(memory);
                let source_a = self.fetch_u8(memory);
                let source_b = self.fetch_u8(memory);

                // Read source matrix registers.
                let a = self.read_matrix(source_a);
                let b = self.read_matrix(source_b);

                let mut result: Matrix = [[0; MATRIX_SIZE]; MATRIX_SIZE];

                // C[row][column] =
                //     A[row][0] * B[0][column]
                //   + A[row][1] * B[1][column]
                //   + ...
                //
                // For a 4x4 matrix this produces:
                //
                // 16 output elements
                // × 4 products each
                // = 64 accelerator multiply operations.
                for row in 0..MATRIX_SIZE {
                    for column in 0..MATRIX_SIZE {
                        let mut accumulator: i32 = 0;

                        for k in 0..MATRIX_SIZE {
                            // Neuron matrix multiplication currently uses
                            // signed INT8 inputs.
                            let a_value = i8::try_from(a[row][k]).unwrap_or_else(|_| {
                                panic!(
                                    "MMUL: M{}[{}][{}] value {} does not fit in INT8",
                                    source_a, row, k, a[row][k]
                                )
                            });

                            let b_value = i8::try_from(b[k][column]).unwrap_or_else(|_| {
                                panic!(
                                    "MMUL: M{}[{}][{}] value {} does not fit in INT8",
                                    source_b, k, column, b[k][column]
                                )
                            });

                            // Send the actual multiplication to the
                            // MatrixMultiply processing element.
                            let product = self.execute_accelerator_instruction(
                                AcceleratorInstruction::MatrixMultiply {
                                    a: f32::from(a_value),
                                    b: f32::from(b_value),
                                    output_slot: 0,
                                },
                            );

                            // INT8 × INT8 is exactly representable as f32,
                            // so converting this product back to i32 is safe
                            // for the current accelerator implementation.
                            accumulator = accumulator.wrapping_add(product as i32);
                        }

                        result[row][column] = accumulator;
                    }
                }

                // Write the complete 4x4 result back to the destination
                // matrix register.
                self.write_matrix(destination, result);

                // MMUL zero flag:
                // set ZERO only when every output element is zero.
                let result_is_zero = result.iter().flatten().all(|&value| value == 0);

                self.set_flag(FLAG_ZERO, result_is_zero);

                // A matrix doesn't have one meaningful scalar sign,
                // carry, or overflow result.
                self.set_flag(FLAG_NEGATIVE, false);
                self.set_flag(FLAG_CARRY, false);
                self.set_flag(FLAG_OVERFLOW, false);
            }
            OP_RELU => {
                let register = self.fetch_u8(memory);

                let value = self.read_scalar(register);
                let signed_value = value as i32;
                self.execute_accelerator_instruction(AcceleratorInstruction::ReLU {
                    input: signed_value as f32,
                    output_slot: 0,
                });
                // Preserve the exact INT32 ISA value even when its f32 representation
                // on the accelerator output bus is rounded.
                let result = if signed_value < 0 { 0 } else { value };

                self.write_scalar(register, result);

                self.update_zero_negative(result);
            }

            // ====================================================
            // 0xFF - HALT
            // ====================================================
            OP_HALT => {
                self.issue(OP_HALT, &[]).expect_halt();
                self.halted = true;
            }

            _ => {
                panic!("Unknown Neuron opcode: {:#04X}", opcode);
            }
        }

        Some(TraceEvent {
            pc: instruction_pc,
            opcode,
            status: self.status,
        })
    }

    pub const fn is_halted(&self) -> bool {
        self.halted
    }

    pub const fn program_counter(&self) -> u32 {
        self.pc
    }

    pub const fn stack_pointer(&self) -> u32 {
        self.sp
    }

    pub const fn frame_pointer(&self) -> u32 {
        self.fp
    }

    pub fn set_frame_pointer(&mut self, value: u32) {
        self.fp = value;
    }

    pub const fn status(&self) -> u32 {
        self.status
    }

    pub const fn total_ticks(&self) -> u64 {
        self.clock.get_tick()
    }

    pub const fn mac_accumulator(&self) -> i32 {
        self.issuer.mac_accumulator()
    }

    pub const fn ai_mode(&self) -> u32 {
        self.ai_mode
    }

    pub fn set_ai_mode(&mut self, value: u32) {
        self.ai_mode = value;
    }

    pub const fn quantization_control(&self) -> u32 {
        self.quant_ctrl
    }

    pub fn set_quantization_control(&mut self, value: u32) {
        self.quant_ctrl = value;
    }

    pub const fn sparsity_control(&self) -> u32 {
        self.sparse_ctrl
    }

    pub fn set_sparsity_control(&mut self, value: u32) {
        self.sparse_ctrl = value;
    }

    pub const fn tensor_control(&self) -> u32 {
        self.tensor_ctrl
    }

    pub fn set_tensor_control(&mut self, value: u32) {
        self.tensor_ctrl = value;
    }

    pub const fn tensor_status(&self) -> u32 {
        self.tensor_status
    }
}

impl Default for NeuronCpu {
    fn default() -> Self {
        Self::new(0)
    }
}
