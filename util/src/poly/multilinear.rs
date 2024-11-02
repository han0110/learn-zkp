use p3_field::{ExtensionField, Field};

/// Returns `[(1 - x_i) * evals[0] + x_i * evals[1], (1 - x_i) * evals[2] + x_i * evals[3], ...]`.
pub fn fix_var<F: Field, E: ExtensionField<F>>(evals: &[F], x_i: &E) -> Vec<E> {
    debug_assert!(evals.len().is_power_of_two());
    evals
        .chunks(2)
        .map(|evals: &[_]| *x_i * (evals[1] - evals[0]) + evals[0])
        .collect()
}
