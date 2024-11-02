use core::borrow::Borrow;
use p3_field::Field;
use poly::univariate::horner;

pub use itertools::{chain, izip, Itertools};

pub mod challenger;
pub mod collection;
pub mod field;
pub mod poly;

/// Returns `values[0] + alpha * values[1] + alpha^2 * values[2] + ...`
pub fn random_linear_combine<F: Field>(
    values: impl IntoIterator<IntoIter: DoubleEndedIterator<Item: Borrow<F>>>,
    alpha: &F,
) -> F {
    horner(values, alpha)
}

#[macro_export]
macro_rules! izip_eq {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip_eq!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        core::iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {{
        $crate::Itertools::zip_eq($crate::izip_eq!($first), $second)
    }};
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::izip_eq!($first);
        $(let t = $crate::izip_eq!(t, $rest);)*
        t.map($crate::izip_eq!(@closure a => (a) $(, $rest)*))
    }};
}
