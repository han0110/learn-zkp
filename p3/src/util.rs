pub use p3_util::*;

pub fn bit_rev<T>(slice: &[T]) -> impl Iterator<Item = &T> {
    debug_assert!(slice.len().is_power_of_two());
    let log_n = slice.len().ilog2() as usize;
    (0..1 << log_n).map(move |i| &slice[reverse_bits_len(i, log_n)])
}
