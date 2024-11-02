use p3_field::{ExtensionField, Field};
use std::borrow::Cow::{self, Owned};
use util::poly::multilinear::fix_var;

#[derive(Clone, Debug)]
pub enum Poly<'a, F: Field, E: ExtensionField<F>> {
    Base(Cow<'a, [F]>),
    Extension(Cow<'a, [E]>),
}

impl<'a, F: Field, E: ExtensionField<F>> Poly<'a, F, E> {
    pub fn num_vars(&self) -> usize {
        let a = self;
        op_poly!(|a| a.len().ilog2() as usize)
    }

    pub fn fix_var(&mut self, x_i: &E) {
        use Poly::*;
        match self {
            Base(evals) => *self = Extension(Owned(fix_var(evals, x_i))),
            Extension(evals) => *evals = Owned(fix_var(evals, x_i)),
        }
    }

    pub fn op<T>(
        a: &Self,
        b: &Self,
        bb: &impl Fn(&[F], &[F]) -> T,
        be: &impl Fn(&[E], &[F]) -> T,
        ee: &impl Fn(&[E], &[E]) -> T,
    ) -> T {
        match (a, b) {
            (Self::Base(a), Self::Base(b)) => bb(a, b),
            (Self::Base(b), Self::Extension(a)) | (Self::Extension(a), Self::Base(b)) => be(a, b),
            (Self::Extension(a), Self::Extension(b)) => ee(a, b),
        }
    }
}

macro_rules! op_polys {
    (|$a:ident, $b:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr $(,)?) => {
        $crate::poly::Poly::op(
            $a,
            $b,
            &|a, b| {
                let $a = a;
                let $b = b;
                let $bb_out = $op;
                $op_bb_out
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                $op
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                $op
            },
        )
    };
    (|$a:ident, $b:ident| $op:expr, $op_bb_out:expr $(,)?) => {
        op_polys!(|$a, $b| $op, |out| $op_bb_out(out))
    };
    (|$a:ident, $b:ident| $op:expr $(,)?) => {
        op_polys!(|$a, $b| $op, |out| out)
    };
}

macro_rules! op_poly {
    (|$a:ident| $op:expr, |$b_out:ident| $op_b_out:expr $(,)?) => {
        match $a {
            $crate::poly::Poly::Base(a) => {
                let $a = a;
                let $b_out = $op;
                $op_b_out
            }
            $crate::poly::Poly::Extension(a) => {
                let $a = a;
                $op
            }
        }
    };
    (|$a:ident| $op:expr, $op_b_out:expr $(,)?) => {
        op_poly!(|$a| $op, |out| $op_b_out(out))
    };
    (|$a:ident| $op:expr $(,)?) => {
        op_poly!(|$a| $op, |out| out)
    };
}

pub(crate) use {op_poly, op_polys};
