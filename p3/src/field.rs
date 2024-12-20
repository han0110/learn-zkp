use util::izip;

mod array;
mod ext_packing;
mod from_uniform_bytes;
mod slice;

pub use array::FieldArray;
pub use ext_packing::ExtPackedValue;
pub use from_uniform_bytes::FromUniformBytes;
pub use slice::FieldSlice;

pub use p3_field::*;

pub fn dot_product<F: FieldAlgebra, E: Clone + FieldExtensionAlgebra<F>>(a: &[E], b: &[F]) -> E {
    izip!(a, b).map(|(a, b)| a.clone() * b.clone()).sum()
}

#[inline]
pub fn dit_butterfly<F: FieldAlgebra + Copy, P: PackedField<Scalar = F> + Copy>(
    a: &mut P,
    b: &mut P,
    t: &F,
) {
    let b_t = *b * *t;
    (*a, *b) = (*a + b_t, *a - b_t);
}
