use neuron::assembler::assemble;
use neuron::cpu::{FLAG_CARRY, FLAG_NEGATIVE, FLAG_OVERFLOW, FLAG_ZERO};
use neuron::{AcceleratorInstruction, AiAccelerator, HardwareUnit, NeuronCpu};

fn run(program: &[u8]) -> NeuronCpu {
    let mut memory = vec![0; 1024];
    memory[..program.len()].copy_from_slice(program);

    let mut cpu = NeuronCpu::new(memory.len() as u32);
    while !cpu.is_halted() {
        cpu.step(&mut memory);
    }
    cpu
}

#[test]
fn executes_scalar_alu_instructions_through_the_cpu() {
    let program = [
        0x10, 0x01, 100, 0, 0, 0, // MOVI R1, 100
        0x10, 0x02, 40, 0, 0, 0, // MOVI R2, 40
        0x20, 0x03, 0x01, 0x02, // ADD R3, R1, R2
        0x21, 0x04, 0x01, 0x02, // SUB R4, R1, R2
        0x22, 0x05, 0x01, 0x02, // MUL R5, R1, R2
        0x23, 0x06, 0x01, 0x02, // DIV R6, R1, R2
        0x24, 0x07, 0x01, 0x02, // MOD R7, R1, R2
        0xFF, // HALT
    ];

    let cpu = run(&program);

    assert!(cpu.is_halted());
    assert_eq!(cpu.read_scalar(3), 140);
    assert_eq!(cpu.read_scalar(4), 60);
    assert_eq!(cpu.read_scalar(5), 4_000);
    assert_eq!(cpu.read_scalar(6), 2);
    assert_eq!(cpu.read_scalar(7), 20);
    assert_eq!(
        cpu.status() & (FLAG_ZERO | FLAG_NEGATIVE | FLAG_CARRY | FLAG_OVERFLOW),
        0
    );
}

#[test]
fn executes_mac_and_matrix_instructions_through_the_cpu() {
    let program = [
        0x10, 0x01, 2, 0, 0, 0, // MOVI R1, 2
        0x10, 0x02, 5, 0, 0, 0,    // MOVI R2, 5
        0x83, // MACCLR
        0x82, 0x01, 0x02, // MAC R1, R2
        0x82, 0x01, 0x02, // MAC R1, R2
        0x84, 0x03, // MACREAD R3
        0x90, 0x02, 0x00, 0x01, // MMUL M2, M0, M1
        0xFF, // HALT
    ];
    let mut memory = vec![0; 1024];
    memory[..program.len()].copy_from_slice(&program);
    let input = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];
    let identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    let mut cpu = NeuronCpu::new(memory.len() as u32);
    cpu.write_matrix(0, input);
    cpu.write_matrix(1, identity);

    while !cpu.is_halted() {
        cpu.step(&mut memory);
    }

    assert_eq!(cpu.read_scalar(3), 20);
    assert_eq!(cpu.mac_accumulator(), 20);
    assert_eq!(cpu.read_matrix(2), input);
    assert_eq!(cpu.accelerator().completed_instruction_count(), 64);
    assert_eq!(cpu.total_ticks(), 70);
}

#[test]
fn assembly_initializes_matrices_before_mmul() {
    let program = assemble(
        r#"
        MSET M0, 0, 0, 1
        MSET M0, 0, 1, 2
        MSET M0, 0, 2, 3
        MSET M0, 0, 3, 4
        MSET M0, 1, 0, 5
        MSET M0, 1, 1, 6
        MSET M0, 1, 2, 7
        MSET M0, 1, 3, 8
        MSET M0, 2, 0, 9
        MSET M0, 2, 1, 10
        MSET M0, 2, 2, 11
        MSET M0, 2, 3, 12
        MSET M0, 3, 0, 13
        MSET M0, 3, 1, 14
        MSET M0, 3, 2, 15
        MSET M0, 3, 3, 16

        MSET M1, 0, 0, 1
        MSET M1, 1, 1, 1
        MSET M1, 2, 2, 1
        MSET M1, 3, 3, 1

        MMUL M2, M0, M1
        HALT
        "#,
    )
    .unwrap();

    let cpu = run(&program);
    let expected = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];

    assert_eq!(cpu.read_matrix(0), expected);
    assert_eq!(cpu.read_matrix(2), expected);
    assert_eq!(cpu.total_ticks(), 84);
}

#[test]
fn mset_preserves_signed_int8_values() {
    let program = assemble("MSET M3, 0, 0, -128\nMSET M3, 3, 3, 127\nHALT").unwrap();

    let cpu = run(&program);

    assert_eq!(cpu.read_matrix(3)[0][0], -128);
    assert_eq!(cpu.read_matrix(3)[3][3], 127);
    assert_eq!(cpu.total_ticks(), 2);
}

#[test]
fn cpu_dispatches_relu_to_the_ai_accelerator() {
    let program = assemble(
        r#"
        MOVI R1, 0xFFFFFFFF
        RELU R1
        MOVI R2, 2147483647
        RELU R2
        HALT
        "#,
    )
    .unwrap();

    let cpu = run(&program);

    assert_eq!(cpu.read_scalar(1), 0);
    assert_eq!(cpu.read_scalar(2), 2_147_483_647);
    assert_eq!(cpu.accelerator().completed_instruction_count(), 2);
    assert_eq!(cpu.total_ticks(), 4);
}

#[test]
fn issuer_preserves_data_memory_stack_and_control_flow_behavior() {
    let program = assemble(
        r#"
        MOVI R1, 240
        MOVI R2, 15
        AND R3, R1, R2
        OR R4, R1, R2
        XOR R5, R1, R2
        NOT R6, R2
        MOVI R7, 4
        SHL R8, R2, R7
        SHR R9, R1, R7
        MOV R10, R9
        MOVI R11, 512
        STORE R11, R4
        LOAD R12, R11
        PUSH R12
        POP R13
        CMP R1, R8
        JNZ failure
        JZ equal
    failure:
        MOVI R14, 99
    equal:
        CALL function
        JMP done
    function:
        MOVI R14, 42
        RET
    done:
        HALT
        "#,
    )
    .unwrap();
    let mut memory = vec![0; 1024];
    memory[..program.len()].copy_from_slice(&program);
    let mut cpu = NeuronCpu::new(memory.len() as u32);

    while !cpu.is_halted() {
        cpu.step(&mut memory);
    }

    assert_eq!(cpu.read_scalar(3), 0);
    assert_eq!(cpu.read_scalar(4), 255);
    assert_eq!(cpu.read_scalar(5), 255);
    assert_eq!(cpu.read_scalar(6), !15);
    assert_eq!(cpu.read_scalar(8), 240);
    assert_eq!(cpu.read_scalar(9), 15);
    assert_eq!(cpu.read_scalar(10), 15);
    assert_eq!(cpu.read_scalar(12), 255);
    assert_eq!(cpu.read_scalar(13), 255);
    assert_eq!(cpu.read_scalar(14), 42);
    assert_eq!(&memory[512..516], &255_u32.to_le_bytes());
}

#[test]
fn cpu_owns_and_executes_ai_accelerator_work() {
    let mut cpu = NeuronCpu::new(1024);

    assert!(
        cpu.accelerator()
            .processing_grid
            .iter()
            .any(|pe| pe.unit_type == HardwareUnit::ReLUActivation)
    );

    cpu.queue_accelerator_instruction(AcceleratorInstruction::ReLU {
        input: -2.0,
        output_slot: 0,
    });
    cpu.process_accelerator_instruction().unwrap();

    assert_eq!(cpu.accelerator().output_bus[0], Some(0.0));
    assert_eq!(cpu.total_ticks(), 1);
}

#[test]
fn cpu_accepts_a_custom_ai_accelerator() {
    let accelerator = AiAccelerator::new(6, 16);
    let cpu = NeuronCpu::with_accelerator(1024, accelerator);

    assert_eq!(cpu.stack_pointer(), 1024);
    assert_eq!(cpu.accelerator().processing_grid.len(), 6);
    assert_eq!(cpu.accelerator().output_bus.len(), 16);
}

#[test]
fn reset_restores_fresh_cpu_state_without_modifying_memory() {
    let program = assemble(
        r#"
        MOVI R1, 42
        MOVI R2, 512
        STORE R2, R1
        MOVI R3, 0
        MAC R1, R1
        PUSH R1
        HALT
        "#,
    )
    .unwrap();
    let mut memory = vec![0; 1024];
    memory[..program.len()].copy_from_slice(&program);
    let mut cpu = NeuronCpu::new(memory.len() as u32);

    for register in 0..16 {
        cpu.write_scalar(register, u32::from(register) + 1);
    }
    for register in 0..8 {
        cpu.write_vector(register, [u32::from(register) + 1; 8]);
    }
    for register in 0..4 {
        cpu.write_matrix(register, [[i32::from(register) + 1; 4]; 4]);
        cpu.write_predicate(register, register + 1);
    }
    cpu.set_frame_pointer(123);
    cpu.set_ai_mode(1);
    cpu.set_quantization_control(2);
    cpu.set_sparsity_control(3);
    cpu.set_tensor_control(4);

    cpu.queue_accelerator_instruction(AcceleratorInstruction::ReLU {
        input: -1.0,
        output_slot: 0,
    });
    cpu.process_accelerator_instruction().unwrap();
    cpu.queue_accelerator_instruction(AcceleratorInstruction::VectorAdd {
        a: 1.0,
        b: 2.0,
        output_slot: 1,
    });

    let identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    cpu.matrix_engine.load_tiles(identity, identity);
    cpu.matrix_engine.start();
    cpu.matrix_engine.step_cycle();

    while !cpu.is_halted() {
        cpu.step(&mut memory);
    }

    assert_ne!(cpu.program_counter(), 0);
    assert_ne!(cpu.stack_pointer(), memory.len() as u32);
    assert_ne!(cpu.status(), 0);
    assert_ne!(cpu.mac_accumulator(), 0);
    assert_ne!(cpu.total_ticks(), 0);
    assert_ne!(cpu.accelerator().completed_instruction_count(), 0);
    assert!(cpu.matrix_engine.is_busy());

    let memory_before_reset = memory.clone();
    cpu.reset();

    assert_eq!(memory, memory_before_reset);
    assert!(!cpu.is_halted());
    assert_eq!(cpu.program_counter(), 0);
    assert_eq!(cpu.stack_pointer(), memory.len() as u32);
    assert_eq!(cpu.frame_pointer(), 0);
    assert_eq!(cpu.status(), 0);
    assert_eq!(cpu.total_ticks(), 0);
    assert_eq!(cpu.mac_accumulator(), 0);
    assert_eq!(cpu.ai_mode(), 0);
    assert_eq!(cpu.quantization_control(), 0);
    assert_eq!(cpu.sparsity_control(), 0);
    assert_eq!(cpu.tensor_control(), 0);
    assert_eq!(cpu.tensor_status(), 0);

    for register in 0..16 {
        assert_eq!(cpu.read_scalar(register), 0);
    }
    for register in 0..8 {
        assert_eq!(cpu.read_vector(register), [0; 8]);
    }
    for register in 0..4 {
        assert_eq!(cpu.read_matrix(register), [[0; 4]; 4]);
        assert_eq!(cpu.read_predicate(register), 0);
    }

    assert!(cpu.accelerator().queue_empty());
    assert_eq!(cpu.accelerator().completed_instruction_count(), 0);
    assert!(cpu.accelerator().output_bus.iter().all(Option::is_none));
    assert!(!cpu.matrix_engine.is_busy());
    assert!(!cpu.matrix_engine.is_done());
    assert_eq!(cpu.matrix_engine.read_output(), None);
    assert_eq!(cpu.matrix_engine.clock_ticks(), 0);
}
