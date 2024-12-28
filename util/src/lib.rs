pub use itertools::{chain, cloned, enumerate, rev, Itertools};
pub use rayon;

#[macro_export]
macro_rules! zip {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::zip!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        core::iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {{
        #[cfg(debug_assertions)]
        { $crate::Itertools::zip_eq($crate::zip!($first), $second) }
        #[cfg(not(debug_assertions))]
        { Iterator::zip($crate::zip!($first), $second) }
    }};
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::zip!($first);
        $(let t = $crate::zip!(t, $rest);)*
        t.map($crate::zip!(@closure a => (a) $(, $rest)*))
    }};
}

#[macro_export]
macro_rules! par_zip {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::par_zip!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        $crate::rayon::prelude::IntoParallelIterator::into_par_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {{
        #[cfg(debug_assertions)]
        { $crate::rayon::prelude::IndexedParallelIterator::zip_eq($crate::par_zip!($first), $second) }
        #[cfg(not(debug_assertions))]
        { $crate::rayon::prelude::IndexedParallelIterator::zip($crate::par_zip!($first), $second) }
    }};
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::par_zip!($first);
        $(let t = $crate::par_zip!(t, $rest);)*
        t.map($crate::par_zip!(@closure a => (a) $(, $rest)*))
    }};
}
