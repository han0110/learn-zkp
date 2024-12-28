use crate::field::{dit_butterfly, Field, PackedValue};
use util::zip;

macro_rules! pack_for_each {
    (@ &mut $buf:ident $(, $($(&$mut:ident)? $tail:ident),*)?) => {
        let $buf = F::Packing::pack_slice_with_suffix_mut($buf);
        $(pack_for_each!(@ $($(&$mut)? $tail),*);)?
    };
    (@ $buf:ident $(, $($(&$mut:ident)? $tail:ident),*)?) => {
        let $buf = F::Packing::pack_slice_with_suffix($buf);
        $(pack_for_each!(@ $($(&$mut)? $tail),*);)?
    };
    (|$($(&$mut:ident)? $buf:ident),*| $f:expr) => {
        #[cfg(debug_assertions)]
        assert!(util::Itertools::tuple_windows([$(&*$buf),*].iter()).all(|(a, b)| a.len() == b.len()));

        pack_for_each!(@ $($(&$mut)? $buf),*);
        zip!($($buf.0,)*).for_each(|($($buf,)*)| $f);
        zip!($($buf.1,)*).for_each(|($($buf,)*)| $f);
    };
}

pub trait FieldSlice<F: Field>: AsRef<[F]> + AsMut<[F]> {
    fn slice_add_assign(&mut self, rhs: &[F]) {
        let lhs = self.as_mut();
        pack_for_each!(|&mut lhs, rhs| *lhs += *rhs);
    }

    fn slice_sub_assign(&mut self, rhs: &[F]) {
        let lhs = self.as_mut();
        pack_for_each!(|&mut lhs, rhs| *lhs -= *rhs);
    }

    fn slice_mul_assign(&mut self, rhs: &[F]) {
        let lhs = self.as_mut();
        pack_for_each!(|&mut lhs, rhs| *lhs *= *rhs);
    }

    fn slice_add(&mut self, lhs: &[F], rhs: &[F]) {
        let out = self.as_mut();
        pack_for_each!(|&mut out, lhs, rhs| *out = *lhs + *rhs);
    }

    fn slice_sub(&mut self, lhs: &[F], rhs: &[F]) {
        let out = self.as_mut();
        pack_for_each!(|&mut out, lhs, rhs| *out = *lhs - *rhs);
    }

    fn slice_mul(&mut self, lhs: &[F], rhs: &[F]) {
        let out = self.as_mut();
        pack_for_each!(|&mut out, lhs, rhs| *out = *lhs * *rhs);
    }

    fn slice_add_scaled_assign(&mut self, rhs: &[F], scalar: F) {
        let lhs = self.as_mut();
        pack_for_each!(|&mut lhs, rhs| *lhs += *rhs * scalar);
    }

    fn slice_dit_butterfly(&mut self, rhs: &mut [F], t: &F) {
        let lhs = self.as_mut();
        pack_for_each!(|&mut lhs, &mut rhs| dit_butterfly(lhs, rhs, t));
    }
}

impl<F: Field, T: AsRef<[F]> + AsMut<[F]>> FieldSlice<F> for T {}
