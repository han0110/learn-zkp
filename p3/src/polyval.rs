use crate::field::{BinaryField, FromUniformBytes};
use core::{
    fmt::{self, Debug, Display},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};
use num_bigint::BigUint;
use p3_field::{Field, FieldAlgebra, Packable};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use serde::{Deserialize, Serialize};

mod portable;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use x86_64_avx2::{from_canonical, invert_or_zero, montgomery_multiply, to_canonical};

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use portable::{from_canonical, invert_or_zero, montgomery_multiply, to_canonical};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Polyval {
    value: u128,
}

impl Polyval {
    #[inline]
    const fn new(value: u128) -> Self {
        Self { value }
    }

    #[inline]
    pub fn from_canonical_u128(value: u128) -> Self {
        Self::new(from_canonical(value))
    }

    #[inline]
    pub fn to_canonical_u128(&self) -> u128 {
        to_canonical(self.value)
    }
}

impl Debug for Polyval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.to_canonical_u128(), f)
    }
}

impl Display for Polyval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.to_canonical_u128(), f)
    }
}

impl Distribution<Polyval> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Polyval {
        Polyval::new(rng.gen())
    }
}

impl FromUniformBytes for Polyval {
    type Bytes = [u8; 16];

    fn try_from_uniform_bytes(bytes: [u8; 16]) -> Option<Self> {
        Some(Self::new(u128::from_le_bytes(bytes)))
    }
}

impl Packable for Polyval {}

impl FieldAlgebra for Polyval {
    type F = Self;

    const ZERO: Self = Self::new(0);
    const ONE: Self = Self::new(portable::from_canonical(1));
    const TWO: Self = Self::ZERO;
    const NEG_ONE: Self = Self::ONE;

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self::from_canonical_u128(b.into())
    }

    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        Self::from_canonical_u128(n.into())
    }

    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        Self::from_canonical_u128(n.into())
    }

    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        Self::from_canonical_u128(n.into())
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        Self::from_canonical_u128(n.into())
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        Self::from_canonical_u128(n as u128)
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        Self::from_canonical_u128(n.into())
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        Self::from_canonical_u128(n.into())
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        vec![Self::ZERO; len]
    }
}

impl Field for Polyval {
    type Packing = Polyval;

    const GENERATOR: Self = Self::new(2);

    #[inline]
    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn try_inverse(&self) -> Option<Self> {
        (!self.is_zero()).then(|| {
            let inv = invert_or_zero(self.value);
            Self::new(inv)
        })
    }

    fn halve(&self) -> Self {
        unreachable!()
    }

    #[inline]
    fn order() -> BigUint {
        BigUint::from(1u64) << 128
    }
}

impl BinaryField for Polyval {
    #[inline]
    fn basis(i: usize) -> Self {
        Self::from_canonical_u128(1 << i)
    }
}

impl Neg for Polyval {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self
    }
}

impl Add for Polyval {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.value ^ rhs.value)
    }
}

impl AddAssign for Polyval {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Polyval {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl SubAssign for Polyval {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Polyval {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(montgomery_multiply(self.value, rhs.value))
    }
}

impl MulAssign for Polyval {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Polyval {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl Sum for Polyval {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Product for Polyval {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        field::FieldAlgebra,
        field_testing::{test_inv_div, test_inverse, test_multiplicative_group_factors},
        polyval::Polyval,
    };
    use rand::{thread_rng, Rng};

    type F = Polyval;

    #[test]
    pub fn polyval() {
        let mut rng = thread_rng();
        let x = rng.gen::<F>();
        let y = rng.gen::<F>();
        let z = rng.gen::<F>();
        assert_eq!(F::ONE + F::NEG_ONE, F::ZERO);
        assert_eq!(x + (-x), F::ZERO);
        assert_eq!(F::ONE + F::ONE, F::TWO);
        assert_eq!(-x, F::ZERO - x);
        assert_eq!(x + x, x * F::TWO);
        assert_eq!(x * F::TWO, x.double());
        // assert_eq!(x, x.halve() * F::TWO);
        assert_eq!(x * (-x), -x.square());
        assert_eq!(x + y, y + x);
        assert_eq!(x * F::ZERO, F::ZERO);
        assert_eq!(x * F::ONE, x);
        assert_eq!(x * y, y * x);
        assert_eq!(x * (y * z), (x * y) * z);
        assert_eq!(x - (y + z), (x - y) - z);
        assert_eq!((x + y) - z, x + (y - z));
        assert_eq!(x * (y + z), x * y + x * z);
        assert_eq!(
            x + y + z + x + y + z,
            [x, x, y, y, z, z].iter().cloned().sum()
        );

        test_inv_div::<F>();
        test_inverse::<F>();
        test_multiplicative_group_factors::<F>();
    }
}
