use core::borrow::Borrow;
use p3_field::Field;
use poly::univariate::horner;

pub use itertools::{chain, enumerate, Itertools};
pub use p3_maybe_rayon as rayon;

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
macro_rules! izip {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        core::iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {{
        #[cfg(debug_assertions)]
        { $crate::Itertools::zip_eq($crate::izip!($first), $second) }
        #[cfg(not(debug_assertions))]
        { Iterator::zip($crate::izip!($first), $second) }
    }};
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::izip!($first);
        $(let t = $crate::izip!(t, $rest);)*
        t.map($crate::izip!(@closure a => (a) $(, $rest)*))
    }};
}

#[macro_export]
macro_rules! izip_par {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip_par!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        $crate::rayon::prelude::IntoParallelIterator::into_par_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {{
        #[cfg(debug_assertions)]
        { $crate::rayon::prelude::IndexedParallelIterator::zip_eq($crate::izip_par!($first), $second) }
        #[cfg(not(debug_assertions))]
        { $crate::rayon::prelude::IndexedParallelIterator::zip($crate::izip_par!($first), $second) }
    }};
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::izip_par!($first);
        $(let t = $crate::izip_par!(t, $rest);)*
        t.map($crate::izip_par!(@closure a => (a) $(, $rest)*))
    }};
}
