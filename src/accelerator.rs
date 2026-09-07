use std::collections::VecDeque;

use crate::clock::Clock;

pub const DEFAULT_GRID_SIZE: usize = 3;
pub const DEFAULT_OUTPUT_BUS_SIZE: usize = 8;

/// The kind of physical compute unit represented by a processing element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareUnit {
    MatrixMultiply,
    VectorAdd,
    ReLUActivation,
}

/// Work that can be submitted to the AI accelerator.
#[derive(Debug, Clone, PartialEq)]
pub enum AcceleratorInstruction {
    MatrixMultiply { a: f32, b: f32, output_slot: usize },

    VectorAdd { a: f32, b: f32, output_slot: usize },

    ReLU { input: f32, output_slot: usize },
}

/// A physical processing element inside the accelerator.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessingElement {
    pub unit_type: HardwareUnit,

    pub input_buffer_a: Option<f32>,
    pub input_buffer_b: Option<f32>,

    pub target_node_id: Option<usize>,

    /// `0` routes to input A and `1` routes to input B.
    pub target_buffer_slot: u8,

    pub busy: bool,
}

/// Snapshot of currently available accelerator hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareStatus {
    pub matrix_multiply_free: usize,
    pub vector_add_free: usize,
    pub relu_free: usize,
}

/// Neuron AI/DNN accelerator.
#[derive(Debug, Clone, PartialEq)]
pub struct AiAccelerator {
    pub processing_grid: Vec<ProcessingElement>,

    pub output_bus: Vec<Option<f32>>,

    pub instructqueue: VecDeque<AcceleratorInstruction>,

    completed_instruction_count: u64,
}

impl AiAccelerator {
    pub fn new(grid_size: usize, output_bus_size: usize) -> Self {
        let processing_grid = (0..grid_size)
            .map(|index| ProcessingElement {
                unit_type: match index % 3 {
                    0 => HardwareUnit::MatrixMultiply,
                    1 => HardwareUnit::VectorAdd,
                    _ => HardwareUnit::ReLUActivation,
                },

                input_buffer_a: None,
                input_buffer_b: None,

                target_node_id: None,
                target_buffer_slot: 0,

                busy: false,
            })
            .collect();

        Self {
            processing_grid,
            output_bus: vec![None; output_bus_size],
            instructqueue: VecDeque::new(),
            completed_instruction_count: 0,
        }
    }

    /// Add one instruction to the accelerator queue.
    pub fn add_instruction(&mut self, instruction: AcceleratorInstruction) {
        self.instructqueue.push_back(instruction);
    }

    /// Number of instructions currently waiting.
    pub fn queue_len(&self) -> usize {
        self.instructqueue.len()
    }

    /// Returns true when there is no queued accelerator work.
    pub fn queue_empty(&self) -> bool {
        self.instructqueue.is_empty()
    }

    /// Query the currently available hardware resources.
    pub fn hardware_status(&self) -> HardwareStatus {
        let mut status = HardwareStatus {
            matrix_multiply_free: 0,
            vector_add_free: 0,
            relu_free: 0,
        };

        for pe in &self.processing_grid {
            if pe.busy {
                continue;
            }

            match pe.unit_type {
                HardwareUnit::MatrixMultiply => {
                    status.matrix_multiply_free += 1;
                }

                HardwareUnit::VectorAdd => {
                    status.vector_add_free += 1;
                }

                HardwareUnit::ReLUActivation => {
                    status.relu_free += 1;
                }
            }
        }

        status
    }

    /// Process exactly ONE complete instruction from the queue.
    ///
    /// If execution fails, the instruction is returned to the front
    /// of the queue so it is not silently lost.
    pub fn process_instruction(&mut self) -> Result<f32, String> {
        let instruction = self
            .instructqueue
            .pop_front()
            .ok_or_else(|| "No accelerator instructions waiting.".to_string())?;

        println!(
            "[ACCEL] dispatch {:?} | queue remaining={}",
            instruction,
            self.instructqueue.len()
        );

        match self.execute_instruction(instruction.clone()) {
            Ok(result) => Ok(result),

            Err(error) => {
                // Put failed work back into the queue.
                self.instructqueue.push_front(instruction);
                Err(error)
            }
        }
    }

    /// Processes one queued instruction and advances the CPU clock only when
    /// the accelerator completes the work successfully.
    pub fn process_instruction_clocked(&mut self, clock: &mut Clock) -> Result<f32, String> {
        let result = self.process_instruction()?;
        clock.tick();
        Ok(result)
    }

    /// Execute one complete accelerator instruction.
    ///
    /// This does NOT modify the instruction queue.
    pub fn execute_instruction(
        &mut self,
        instruction: AcceleratorInstruction,
    ) -> Result<f32, String> {
        let required_unit = Self::required_hardware(&instruction);
        let output_slot = Self::instruction_output_slot(&instruction);

        if output_slot >= self.output_bus.len() {
            return Err(format!("Invalid accelerator output bus slot {output_slot}"));
        }

        let pe_index = self
            .find_available_resource(required_unit)
            .ok_or_else(|| format!("No available {:?} hardware resource", required_unit))?;

        self.reserve_resource(pe_index)?;

        println!("[ACCEL] allocated PE {} ({:?})", pe_index, required_unit);

        //
        // Execute the entire instruction while the PE is reserved.
        //
        let result = match instruction {
            AcceleratorInstruction::MatrixMultiply { a, b, output_slot } => {
                self.processing_grid[pe_index].input_buffer_a = Some(a);
                self.processing_grid[pe_index].input_buffer_b = Some(b);

                let result = a * b;

                self.output_bus[output_slot] = Some(result);

                result
            }

            AcceleratorInstruction::VectorAdd { a, b, output_slot } => {
                self.processing_grid[pe_index].input_buffer_a = Some(a);
                self.processing_grid[pe_index].input_buffer_b = Some(b);

                let result = a + b;

                self.output_bus[output_slot] = Some(result);

                result
            }

            AcceleratorInstruction::ReLU { input, output_slot } => {
                self.processing_grid[pe_index].input_buffer_a = Some(input);
                self.processing_grid[pe_index].input_buffer_b = None;

                let result = input.max(0.0);

                self.output_bus[output_slot] = Some(result);

                result
            }
        };

        println!(
            "[ACCEL] PE {} ({:?}) result={} -> output_bus[{}]",
            pe_index, required_unit, result, output_slot
        );

        self.release_resource(pe_index);

        self.completed_instruction_count = self.completed_instruction_count.wrapping_add(1);

        Ok(result)
    }

    /// Executes immediate accelerator work and advances the supplied clock
    /// after successful completion.
    pub fn execute_instruction_clocked(
        &mut self,
        instruction: AcceleratorInstruction,
        clock: &mut Clock,
    ) -> Result<f32, String> {
        let result = self.execute_instruction(instruction)?;
        clock.tick();
        Ok(result)
    }

    /// Determine which physical unit an instruction requires.
    fn required_hardware(instruction: &AcceleratorInstruction) -> HardwareUnit {
        match instruction {
            AcceleratorInstruction::MatrixMultiply { .. } => HardwareUnit::MatrixMultiply,

            AcceleratorInstruction::VectorAdd { .. } => HardwareUnit::VectorAdd,

            AcceleratorInstruction::ReLU { .. } => HardwareUnit::ReLUActivation,
        }
    }

    /// Determine which output bus slot the instruction targets.
    fn instruction_output_slot(instruction: &AcceleratorInstruction) -> usize {
        match instruction {
            AcceleratorInstruction::MatrixMultiply { output_slot, .. }
            | AcceleratorInstruction::VectorAdd { output_slot, .. }
            | AcceleratorInstruction::ReLU { output_slot, .. } => *output_slot,
        }
    }

    /// Search the resource pool for an idle PE of the requested type.
    fn find_available_resource(&self, required_unit: HardwareUnit) -> Option<usize> {
        self.processing_grid
            .iter()
            .position(|pe| pe.unit_type == required_unit && !pe.busy)
    }

    /// Reserve a processing element.
    fn reserve_resource(&mut self, pe_index: usize) -> Result<(), String> {
        let pe = self
            .processing_grid
            .get_mut(pe_index)
            .ok_or_else(|| format!("Invalid processing element {pe_index}"))?;

        if pe.busy {
            return Err(format!("Processing element {pe_index} is already busy"));
        }

        pe.busy = true;

        Ok(())
    }

    /// Release a processing element back into the resource pool.
    fn release_resource(&mut self, pe_index: usize) {
        if let Some(pe) = self.processing_grid.get_mut(pe_index) {
            pe.busy = false;

            pe.input_buffer_a = None;
            pe.input_buffer_b = None;
        }
    }

    /// Read an accelerator output bus slot.
    pub fn read_output(&self, output_slot: usize) -> Option<f32> {
        self.output_bus.get(output_slot).copied().flatten()
    }

    /// Clear an accelerator output bus slot.
    pub fn clear_output(&mut self, output_slot: usize) -> Result<(), String> {
        let slot = self
            .output_bus
            .get_mut(output_slot)
            .ok_or_else(|| format!("Invalid accelerator output bus slot {output_slot}"))?;

        *slot = None;

        Ok(())
    }

    pub const fn completed_instruction_count(&self) -> u64 {
        self.completed_instruction_count
    }
}

impl Default for AiAccelerator {
    fn default() -> Self {
        Self::new(DEFAULT_GRID_SIZE, DEFAULT_OUTPUT_BUS_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_accelerator_contains_all_hardware_types() {
        let accelerator = AiAccelerator::default();

        let status = accelerator.hardware_status();

        assert_eq!(status.matrix_multiply_free, 1);
        assert_eq!(status.vector_add_free, 1);
        assert_eq!(status.relu_free, 1);
    }

    #[test]
    fn accelerator_executes_every_hardware_type() {
        let mut accelerator = AiAccelerator::default();

        accelerator.add_instruction(AcceleratorInstruction::MatrixMultiply {
            a: 3.0,
            b: 4.0,
            output_slot: 0,
        });

        accelerator.add_instruction(AcceleratorInstruction::VectorAdd {
            a: 2.5,
            b: 1.5,
            output_slot: 1,
        });

        accelerator.add_instruction(AcceleratorInstruction::ReLU {
            input: -8.0,
            output_slot: 2,
        });

        assert_eq!(accelerator.process_instruction().unwrap(), 12.0);

        assert_eq!(accelerator.process_instruction().unwrap(), 4.0);

        assert_eq!(accelerator.process_instruction().unwrap(), 0.0);

        assert_eq!(accelerator.output_bus[0], Some(12.0));

        assert_eq!(accelerator.output_bus[1], Some(4.0));

        assert_eq!(accelerator.output_bus[2], Some(0.0));

        assert!(accelerator.instructqueue.is_empty());

        assert!(accelerator.processing_grid.iter().all(|pe| !pe.busy));

        assert_eq!(accelerator.completed_instruction_count(), 3);
    }

    #[test]
    fn process_instruction_only_processes_one_job() {
        let mut accelerator = AiAccelerator::default();

        accelerator.add_instruction(AcceleratorInstruction::MatrixMultiply {
            a: 2.0,
            b: 3.0,
            output_slot: 0,
        });

        accelerator.add_instruction(AcceleratorInstruction::ReLU {
            input: -5.0,
            output_slot: 1,
        });

        assert_eq!(accelerator.queue_len(), 2);

        accelerator.process_instruction().unwrap();

        assert_eq!(accelerator.queue_len(), 1);

        assert_eq!(accelerator.completed_instruction_count(), 1);

        assert_eq!(accelerator.output_bus[0], Some(6.0));

        assert_eq!(accelerator.output_bus[1], None);
    }

    #[test]
    fn failed_instruction_remains_queued() {
        let mut accelerator = AiAccelerator::default();

        accelerator.add_instruction(AcceleratorInstruction::ReLU {
            input: 1.0,
            output_slot: DEFAULT_OUTPUT_BUS_SIZE,
        });

        let result = accelerator.process_instruction();

        assert_eq!(
            result,
            Err(format!(
                "Invalid accelerator output bus slot {}",
                DEFAULT_OUTPUT_BUS_SIZE
            ))
        );

        assert_eq!(accelerator.queue_len(), 1);

        assert_eq!(accelerator.completed_instruction_count(), 0);

        assert!(accelerator.processing_grid.iter().all(|pe| !pe.busy));
    }

    #[test]
    fn relu_works_for_positive_and_negative_values() {
        let mut accelerator = AiAccelerator::default();

        let positive = accelerator
            .execute_instruction(AcceleratorInstruction::ReLU {
                input: 8.0,
                output_slot: 0,
            })
            .unwrap();

        let negative = accelerator
            .execute_instruction(AcceleratorInstruction::ReLU {
                input: -8.0,
                output_slot: 1,
            })
            .unwrap();

        assert_eq!(positive, 8.0);
        assert_eq!(negative, 0.0);
    }

    #[test]
    fn clocked_execution_ticks_only_after_success() {
        let mut accelerator = AiAccelerator::default();
        let mut clock = Clock::new();

        accelerator
            .execute_instruction_clocked(
                AcceleratorInstruction::VectorAdd {
                    a: 1.0,
                    b: 2.0,
                    output_slot: 0,
                },
                &mut clock,
            )
            .unwrap();
        let failed = accelerator.execute_instruction_clocked(
            AcceleratorInstruction::ReLU {
                input: 1.0,
                output_slot: DEFAULT_OUTPUT_BUS_SIZE,
            },
            &mut clock,
        );

        assert!(failed.is_err());
        assert_eq!(clock.get_tick(), 1);
    }
}
