use core::borrow::Borrow;
use p3_field::Field;

/// Returns `coeffs[0] + x * coeffs[1] + x^2 * coeffs[2] + ...`
pub fn horner<F: Field>(
    coeffs: impl IntoIterator<IntoIter: DoubleEndedIterator<Item: Borrow<F>>>,
    x: &F,
) -> F {
    coeffs
        .into_iter()
        .rfold(F::ZERO, |eval, coeff| eval * *x + *coeff.borrow())
}
