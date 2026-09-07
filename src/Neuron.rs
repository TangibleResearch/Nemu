use dotenvy::dotenv;
use neuron::debug::ic::{INSTRUCTION_LOG, Instruction};
use neuron::{NeuronCpu, debug::etb::ETB};
use std::env;
use std::fs;
use std::process::ExitCode;

const MEMORY_SIZE: usize = 1024;

fn main() -> ExitCode {
    dotenv().ok();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let program_path = arguments
        .next()
        .ok_or_else(|| "usage: neuron <program.bin>".to_owned())?;
    if arguments.next().is_some() {
        return Err("usage: neuron <program.bin>".to_owned());
    }

    let program = fs::read(&program_path)
        .map_err(|error| format!("failed to read {}: {error}", program_path.to_string_lossy()))?;
    if program.len() > MEMORY_SIZE {
        return Err(format!(
            "program is {} bytes but Neuron memory is only {MEMORY_SIZE} bytes",
            program.len()
        ));
    }

    let mut memory = vec![0_u8; MEMORY_SIZE];
    memory[..program.len()].copy_from_slice(&program);

    let mut cpu = NeuronCpu::new(MEMORY_SIZE as u32);
    let mut etb = ETB::new();

    println!("--- Neuron32 Boot ---");
    dotenv().ok();
    println!("© Copyright 2026 Tangible Research Inc. All rights reserved.");
    println!("Checking System Specifications...");
    match env::var("VERSION") {
        Ok(val) => println!("Version no: {}", val),
        Err(e) => println!("Couldn't read VERSION: {}", e),
    }
    match env::var("CODENAME") {
        Ok(val) => println!("Codename {}", val),
        Err(e) => println!("Couldn't read CODENAME: {}", e),
    }
    match env::var("SYSTEM") {
        Ok(val) => println!("System: {}", val),
        Err(e) => println!("Couldn't read SYSTEM: {}", e),
    }
    match env::var("SERIAL") {
        Ok(val) => println!("Serial : {}", val),
        Err(e) => println!("Couldn't read SERIAL: {}", e),
    }
    match env::var("COMPILE_DATE") {
        Ok(val) => println!("Compile on: {}", val),
        Err(e) => println!("Couldn't read COMPILE_DATE: {}", e),
    }
    while !cpu.is_halted() {
        if let Some(event) = cpu.step(&mut memory) {
            Instruction::log(event.opcode);
            etb.record(event);
        }
    }

    println!("Debugger Output:");
    etb.dump(&mut std::io::stdout().lock())
        .map_err(|error| format!("failed to write execution trace: {error}"))?;
    println!(
        "PC={} SP={} STATUS={:#010X}",
        cpu.program_counter(),
        cpu.stack_pointer(),
        cpu.status()
    );
    for register in 1..=5 {
        println!("R{register} = {}", cpu.read_scalar(register));
    }
    let stats = INSTRUCTION_LOG.lock().unwrap();

    println!("--- Instruction Statistics ---");
    println!("Total: {}", stats.total);
    println!("MOVI: {}", stats.movi);
    println!("MOV: {}", stats.mov);
    println!("ADD: {}", stats.add);
    println!("SUB: {}", stats.sub);
    println!("MUL: {}", stats.mul);
    println!("DIV: {}", stats.div);
    println!("MOD: {}", stats.mod_);
    println!("AND: {}", stats.and);
    println!("OR: {}", stats.or);
    println!("XOR: {}", stats.xor);
    println!("NOT: {}", stats.not);
    println!("SHL: {}", stats.shl);
    println!("SHR: {}", stats.shr);
    println!("LOAD: {}", stats.load);
    println!("STORE: {}", stats.store);
    println!("PUSH: {}", stats.push);
    println!("POP: {}", stats.pop);
    println!("CMP: {}", stats.cmp);
    println!("JMP: {}", stats.jmp);
    println!("JZ: {}", stats.jz);
    println!("JNZ: {}", stats.jnz);
    println!("CALL: {}", stats.call);
    println!("RET: {}", stats.ret);
    println!("MAC: {}", stats.mac);
    println!("MACCLR: {}", stats.macclr);
    println!("MACREAD: {}", stats.macread);
    println!("MMUL: {}", stats.mmul);
    println!("MSET: {}", stats.mset);
    println!("OUT: {}", stats.out);
    println!("RELU: {}", stats.relu);
    println!("HALT: {}", stats.halt);
    println!("Total clock ticks: {}", cpu.total_ticks());
    cpu.reset();
    Ok(())
}
