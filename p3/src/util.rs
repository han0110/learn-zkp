use util::rayon::prelude::*;

pub use p3_util::*;

pub fn bit_rev<T>(slice: &[T]) -> impl Iterator<Item = &T> {
    debug_assert!(slice.len().is_power_of_two());
    let log_n = slice.len().ilog2() as usize;
    (0..1 << log_n).map(move |i| &slice[reverse_bits_len(i, log_n)])
}

pub fn par_bit_rev<T: Sync>(slice: &[T]) -> impl IndexedParallelIterator<Item = &T> {
    debug_assert!(slice.len().is_power_of_two());
    let log_n = slice.len().ilog2() as usize;
    (0..1 << log_n)
        .into_par_iter()
        .map(move |i| &slice[reverse_bits_len(i, log_n)])
}
