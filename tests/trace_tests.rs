use neuron::NeuronCpu;
use neuron::cpu::{FLAG_CARRY, FLAG_ZERO};
use neuron::debug::{
    etb::{ETB, ETB_CAPACITY},
    trace::{TraceEvent, opcode_name},
};

#[test]
fn cpu_emits_one_post_execution_event_per_instruction() {
    let program = [
        0x10, 0x01, 2, 0, 0, 0, // MOVI R1, 2
        0x10, 0x02, 2, 0, 0, 0, // MOVI R2, 2
        0x20, 0x03, 0x01, 0x02, // ADD R3, R1, R2
        0x21, 0x04, 0x01, 0x02, // SUB R4, R1, R2
        0xFF, // HALT
    ];
    let mut memory = vec![0; 64];
    memory[..program.len()].copy_from_slice(&program);
    let mut cpu = NeuronCpu::new(memory.len() as u32);

    let movi = cpu.step(&mut memory).expect("MOVI should emit an event");
    assert_eq!(
        movi,
        TraceEvent {
            pc: 0,
            opcode: 0x10,
            status: 0,
        }
    );

    cpu.step(&mut memory).expect("second MOVI should emit");

    let add = cpu.step(&mut memory).expect("ADD should emit an event");
    assert_eq!(add.pc, 12);
    assert_eq!(add.opcode, 0x20);

    let sub = cpu.step(&mut memory).expect("SUB should emit an event");
    assert_eq!(sub.pc, 16);
    assert_eq!(sub.opcode, 0x21);
    assert_eq!(sub.status, cpu.status());
    assert_eq!(
        sub.status & (FLAG_ZERO | FLAG_CARRY),
        FLAG_ZERO | FLAG_CARRY
    );

    let halt = cpu.step(&mut memory).expect("HALT should emit an event");
    assert_eq!(halt.pc, 20);
    assert_eq!(halt.opcode, 0xFF);
    assert_eq!(halt.status, cpu.status());
    assert!(cpu.is_halted());
    assert_eq!(cpu.step(&mut memory), None);
    assert_eq!(cpu.total_ticks(), 4);
}

#[test]
fn event_status_reflects_the_completed_instruction() {
    let program = [
        0x10, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, // MOVI R1, u32::MAX
        0x10, 0x02, 1, 0, 0, 0, // MOVI R2, 1
        0x20, 0x03, 0x01, 0x02, // ADD R3, R1, R2
        0xFF,
    ];
    let mut memory = vec![0; 64];
    memory[..program.len()].copy_from_slice(&program);
    let mut cpu = NeuronCpu::new(memory.len() as u32);

    cpu.step(&mut memory);
    cpu.step(&mut memory);
    let event = cpu.step(&mut memory).expect("ADD should emit an event");

    assert_eq!(cpu.read_scalar(3), 0);
    assert_eq!(event.status, cpu.status());
    assert_eq!(
        event.status & (FLAG_ZERO | FLAG_CARRY),
        FLAG_ZERO | FLAG_CARRY
    );
}

#[test]
fn etb_stores_events_in_execution_order() {
    let mut etb = ETB::new();
    let first = TraceEvent {
        pc: 0,
        opcode: 0x10,
        status: 0,
    };
    let second = TraceEvent {
        pc: 6,
        opcode: 0x20,
        status: FLAG_ZERO,
    };

    etb.record(first);
    etb.record(second);

    let mut events = etb.iter();
    assert_eq!(events.next(), Some(&first));
    assert_eq!(events.next(), Some(&second));
    assert_eq!(events.next(), None);
}

#[test]
fn etb_retains_the_latest_256_events() {
    assert_eq!(ETB_CAPACITY, 256);
    let mut etb = ETB::new();

    for pc in 0..=ETB_CAPACITY as u32 {
        etb.record(TraceEvent {
            pc,
            opcode: 0x10,
            status: pc,
        });
    }

    assert_eq!(etb.len(), ETB_CAPACITY);
    assert_eq!(etb.iter().next().map(|event| event.pc), Some(1));
    assert_eq!(etb.iter().last().map(|event| event.pc), Some(256));
}

#[test]
fn cpu_executes_without_an_etb() {
    let mut memory = vec![0; 32];
    let program = [0x10, 0x01, 42, 0, 0, 0, 0xFF];
    memory[..program.len()].copy_from_slice(&program);
    let mut cpu = NeuronCpu::new(memory.len() as u32);

    while !cpu.is_halted() {
        let _ = cpu.step(&mut memory);
    }

    assert_eq!(cpu.read_scalar(1), 42);
    assert_eq!(opcode_name(0x10), "MOVI");
    assert_eq!(cpu.total_ticks(), 1);
}
