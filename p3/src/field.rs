use crate::poly::univariate::horner;
use core::borrow::Borrow;

mod array;
mod ext_packing;
mod from_uniform_bytes;
mod slice;

pub use array::FieldArray;
pub use ext_packing::ExtPackedValue;
pub use from_uniform_bytes::FromUniformBytes;
pub use slice::FieldSlice;

pub use p3_field::*;

pub use p3_baby_bear::BabyBear;
pub use p3_goldilocks::Goldilocks;
pub use p3_mersenne_31::Mersenne31;

/// Returns `values[0] + alpha * values[1] + alpha^2 * values[2] + ...`
pub fn random_linear_combine<F: Field>(
    values: impl IntoIterator<IntoIter: DoubleEndedIterator<Item: Borrow<F>>>,
    alpha: &F,
) -> F {
    horner(values, alpha)
}

#[inline]
pub fn dit_butterfly<F: AbstractField + Copy, P: PackedField<Scalar = F> + Copy>(
    a: &mut P,
    b: &mut P,
    t: &F,
) {
    let b_t = *b * *t;
    (*a, *b) = (*a + b_t, *a - b_t);
}
